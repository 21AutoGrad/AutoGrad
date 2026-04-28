"""
Inference pipeline.

Loads ONLY the trained CriterionWiseScoringSystem checkpoint (best_model_v5.pt)
locally. The CodeT5 encoder lives in a separate Hugging Face Space — we call
it over HTTP for hidden states. The Qwen explanation LLM is called via the
Hugging Face Inference API. See model/encoder_client.py and model/explainer.py.

Net effect: the deployment server only needs the small .pt checkpoint plus
the cross-attention layers in memory (~150–200 MB). No transformers / no
CodeT5 / no Qwen on the server.
"""
import logging
import os
import time

import torch

from .architecture import CriterionWiseScoringSystem
from .signals import extract_signals_from_attn
from .explainer import build_explanation, normalize_signals
from .encoder_client import encode_text

logger = logging.getLogger("inference")


# ── Config (overridable via env vars) ──────────────────────────────────────
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "best_model_v5.pt")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
N_HEADS = int(os.getenv("N_HEADS", "8"))
N_CROSS_LAYERS = int(os.getenv("N_CROSS_LAYERS", "2"))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


# ── Module-level singletons (initialised by load_pipeline) ─────────────────
_model = None
_device = None


def load_pipeline():
    """Load the trained scoring model. Call once at boot."""
    global _model, _device

    if _model is not None:
        return  # already loaded

    t0 = time.perf_counter()
    device = torch.device(DEVICE)
    logger.info("Loading scoring head on device: %s", device)

    model = CriterionWiseScoringSystem(
        embedding_dim=EMBEDDING_DIM,
        n_heads=N_HEADS,
        n_cross_layers=N_CROSS_LAYERS,
        dropout=0.1,
    ).to(device)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at '{CHECKPOINT_PATH}'. "
            f"Set CHECKPOINT_PATH env var or place best_model_v5.pt in the project root."
        )

    logger.info("Loading checkpoint: %s", CHECKPOINT_PATH)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    _model = model
    _device = device
    logger.info("Pipeline ready in %.1fs total.", time.perf_counter() - t0)


def _validate_payload(question, answer, criteria):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("'question' must be a non-empty string")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("'answer' must be a non-empty string")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("'criteria' must be a non-empty list")
    for i, c in enumerate(criteria):
        if not isinstance(c, dict):
            raise ValueError(f"criteria[{i}] must be an object")
        if "name" not in c:
            raise ValueError(f"criteria[{i}] missing 'name'")
        if "max_score" not in c:
            raise ValueError(f"criteria[{i}] missing 'max_score'")
        try:
            float(c["max_score"])
        except (TypeError, ValueError):
            raise ValueError(f"criteria[{i}].max_score must be numeric")


# ── Public API ─────────────────────────────────────────────────────────────
def predict(question, answer, criteria):
    """
    Score one (question, answer) pair against a list of rubric criteria.

    Same signature and return shape as the original implementation; the
    encoder is now remote and the LLM uses the HF Inference API.
    """
    if _model is None:
        raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")

    _validate_payload(question, answer, criteria)

    t_total = time.perf_counter()
    logger.info(
        "Starting prediction: question=%d chars, answer=%d chars, %d criteria",
        len(question), len(answer), len(criteria),
    )

    # Encode question and answer ONCE — they are shared across criteria
    t_enc = time.perf_counter()
    q_hidden, q_mask, question_tokens = encode_text(question, _device)
    a_hidden, a_mask, answer_tokens = encode_text(answer, _device)
    logger.info("Encoded question + answer via Space (%.2fs)", time.perf_counter() - t_enc)

    criterion_results = []
    total_score = 0.0
    max_total = 0.0
    n_crit = len(criteria)

    for idx, crit in enumerate(criteria, start=1):
        t_crit = time.perf_counter()
        name = crit["name"]
        desc = crit.get("description") or name
        max_s = float(crit["max_score"])
        logger.info("[%d/%d] Criterion '%s' (max=%.0f) → scoring…", idx, n_crit, name, max_s)

        # Notebook always prepends the <criterion> sentinel — the Space's
        # tokenizer knows about it (we add it during Space boot).
        c_text = "<criterion> " + desc
        c_hidden, c_mask, criterion_tokens = encode_text(c_text, _device)

        t_fwd = time.perf_counter()
        with torch.no_grad():
            pred_norm, attn_dict = _model(
                q_hidden, a_hidden, c_hidden,
                A_mask=a_mask, c_mask=c_mask,
                return_attn=True,
            )

        pred_norm = float(pred_norm.item())
        score_val = pred_norm * max_s
        logger.info(
            "[%d/%d] score=%.2f/%.0f (pred_norm=%.4f) in %.2fs",
            idx, n_crit, score_val, max_s, pred_norm, time.perf_counter() - t_fwd,
        )

        signals = extract_signals_from_attn(
            attn_dict,
            answer_tokens=answer_tokens,
            criterion_tokens=criterion_tokens,
            question_tokens=question_tokens,
        )
        signals = normalize_signals(signals)

        t_expl = time.perf_counter()
        logger.info("[%d/%d] Generating explanation via HF Inference API…", idx, n_crit)
        explanation = build_explanation(
            criterion_name=name,
            pred_norm=pred_norm,
            score=score_val,
            max_score=max_s,
            signals=signals,
            question=question,
            answer=answer,
            criterion_desc=desc,
        )
        logger.info(
            "[%d/%d] Explanation ready in %.2fs (criterion total %.2fs)",
            idx, n_crit, time.perf_counter() - t_expl, time.perf_counter() - t_crit,
        )

        criterion_results.append({
            "criterion": name,
            "criterion_description": desc,
            "score": round(score_val, 3),
            "max_score": max_s,
            "pred_norm": round(pred_norm, 4),
            "explanation": explanation,
        })

        total_score += score_val
        max_total += max_s

    percentage = (total_score / max_total * 100.0) if max_total > 0 else 0.0
    overall_explanation = _build_overall_explanation(
        total_score, max_total, criterion_results
    )

    logger.info(
        "Prediction done: total=%.2f/%.0f (%.1f%%) in %.2fs",
        total_score, max_total, percentage, time.perf_counter() - t_total,
    )

    return {
        "question": question,
        "answer": answer,
        "total_score": round(total_score, 3),
        "max_total_score": max_total,
        "percentage": round(percentage, 2),
        "results": criterion_results,
        "overall_explanation": overall_explanation,
    }


def _build_overall_explanation(total, max_total, criterion_results):
    """Tiny narrative stitched together from the per-criterion results."""
    if max_total == 0:
        return "No criteria were scorable."

    pct = total / max_total * 100.0
    if pct >= 85:
        band = "excellent"
    elif pct >= 70:
        band = "strong"
    elif pct >= 50:
        band = "partial"
    else:
        band = "weak"

    strong = [r["criterion"] for r in criterion_results if r["pred_norm"] >= 0.8]
    weak = [r["criterion"] for r in criterion_results if r["pred_norm"] < 0.5]

    parts = [f"Overall {band} performance: {total:.2f} / {max_total:.0f} ({pct:.1f}%)."]
    if strong:
        parts.append("Strongest on: " + ", ".join(strong) + ".")
    if weak:
        parts.append("Needs work on: " + ", ".join(weak) + ".")
    return " ".join(parts)
