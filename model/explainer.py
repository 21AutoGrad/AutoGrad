"""
LLM-based explanation generator — calls the Hugging Face Inference API
instead of running Qwen locally.

Why
---
The deployment server (Render / Railway / VPS) doesn't run any LLM weights;
all generation happens on Hugging Face's infrastructure. The server just
needs an HF user-access token in the HF_TOKEN env var.

Behaviour
---------
* Sends the same chat-style prompt the original notebook used.
* Falls back to the deterministic rule-based explanation if:
    - DISABLE_LLM_EXPLAINER=1, OR
    - HF_TOKEN is missing, OR
    - the API call errors / times out.
* Caches the failure flag so we don't hammer the API on every request when
  it's clearly down.
"""
import logging
import os
import re
import threading

logger = logging.getLogger("explainer")


# ── Config ─────────────────────────────────────────────────────────────────
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
DISABLE_LLM = os.getenv("DISABLE_LLM_EXPLAINER", "0") == "1"
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "120"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
HF_TOKEN = os.getenv("HF_TOKEN")
# Optional: pin a specific HF Inference Provider (e.g. "together", "fireworks-ai",
# "hf-inference"). Leave unset to let HF auto-route.
HF_PROVIDER = os.getenv("HF_PROVIDER")

_failed_lock = threading.Lock()
_llm_failed = False  # latched on first hard failure to avoid retry storms
_inference_client = None
_client_lock = threading.Lock()


def _get_client():
    """Lazy-init the huggingface_hub InferenceClient."""
    global _inference_client
    if _inference_client is not None:
        return _inference_client
    with _client_lock:
        if _inference_client is not None:
            return _inference_client
        from huggingface_hub import InferenceClient
        kwargs = {"token": HF_TOKEN, "timeout": LLM_TIMEOUT}
        if HF_PROVIDER:
            kwargs["provider"] = HF_PROVIDER
        _inference_client = InferenceClient(**kwargs)
        logger.info(
            "InferenceClient ready (model=%s, provider=%s)",
            LLM_MODEL_NAME, HF_PROVIDER or "auto",
        )
        return _inference_client


# ────────────────────────────────────────────────────────────────────────────
# 1. Post-processing — same rules as the notebook
# ────────────────────────────────────────────────────────────────────────────
def _clean_explanation(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    for pat in (
        r"^No other text[.\s]*",
        r"^Explanation:\s*",
        r"^Output:\s*",
        r"^Feedback:\s*",
        r"^Answer:\s*",
    ):
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    # Cap at 2 sentences.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    if len(parts) > 2:
        text = " ".join(parts[:2])
        if text and text[-1] not in ".!?":
            text += "."
    return text.strip()


# ────────────────────────────────────────────────────────────────────────────
# 2. Prompt construction — identical to the original
# ────────────────────────────────────────────────────────────────────────────
def _build_messages(criterion_name, score, max_score,
                    question, answer, signals, criterion_desc=None):
    top_tokens = signals.get("top_answer_tokens", [])
    unattended = signals.get("unattended_concepts", [])
    confidence = signals.get("confidence", 0.0)
    source = signals.get("source", "cross_attention")

    tok_lines = [
        f'"{t.get("token", "")}" (importance: {float(t.get("importance", 0)):.4f})'
        for t in top_tokens
    ]
    tok_str = "; ".join(tok_lines) if tok_lines else "(none)"
    miss_str = ", ".join([f'"{x}"' for x in unattended]) if unattended else "(none)"

    rubric_str = ""
    if criterion_desc and criterion_desc != criterion_name:
        rubric_str = f"Rubric: {criterion_desc}\n"

    percent = (score / max_score * 100) if max_score else 0
    evidence_label = (
        "Tokens the grading model attended to most:"
        if source == "cross_attention"
        else "Tokens that most influenced this score:"
    )

    system_msg = (
        "You are a concise grading assistant. "
        "You write EXACTLY 2 sentences of feedback for a student. "
        "Sentence 1: why this score was given. "
        "Sentence 2: what is missing or how to improve. "
        "Rules: no bullet points, no paragraphs, no lists, no extra sentences. "
        "Output only those 2 sentences and nothing else."
    )
    user_msg = (
        f"CRITERION: {criterion_name}\n"
        f"SCORE: {score:.2f} / {max_score} ({percent:.0f}%)\n"
        f"CONFIDENCE: {confidence * 100:.0f}%\n"
        f"{rubric_str}"
        f"\nQUESTION:\n{question}\n"
        f"\nSTUDENT ANSWER:\n{answer}\n"
        f"\n{evidence_label}\n{tok_str}\n"
        f"\nConcepts with low coverage: {miss_str}\n"
        f"\nWrite exactly 2 sentences of feedback."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# ────────────────────────────────────────────────────────────────────────────
# 3. HF Inference API caller (OpenAI-compatible chat-completions endpoint)
# ────────────────────────────────────────────────────────────────────────────
def _call_hf_chat(messages):
    """Returns the raw assistant text, or raises on failure."""
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN env var is not set")

    client = _get_client()
    resp = client.chat_completion(
        model=LLM_MODEL_NAME,
        messages=messages,
        max_tokens=LLM_MAX_NEW_TOKENS,
        temperature=LLM_TEMPERATURE,
    )
    try:
        return resp.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        raise RuntimeError(f"Unexpected HF chat response shape: {resp!r}")


# ────────────────────────────────────────────────────────────────────────────
# 4. Rule-based fallback (no LLM needed)
# ────────────────────────────────────────────────────────────────────────────
def _band(pred_norm):
    if pred_norm >= 0.85: return "excellent"
    if pred_norm >= 0.65: return "good"
    if pred_norm >= 0.40: return "partial"
    if pred_norm >= 0.15: return "weak"
    return "very poor"


def _rule_based_explanation(criterion_name, pred_norm, score, max_score, signals):
    band = _band(pred_norm)
    top_tokens = [t["token"] for t in signals.get("top_answer_tokens", [])[:3]]
    unattended = signals.get("unattended_concepts", [])[:3]

    s1 = (
        f"The answer earned {score:.2f} / {max_score:.0f} on '{criterion_name}' "
        f"({band}, {pred_norm*100:.0f}%)"
    )
    if top_tokens:
        s1 += f"; the model focused on: {', '.join(top_tokens)}."
    else:
        s1 += "."
    if unattended and pred_norm < 0.85:
        s2 = f"To improve, address the under-covered concepts: {', '.join(unattended)}."
    else:
        s2 = "The response covers the main rubric points."
    return f"{s1} {s2}"


# ────────────────────────────────────────────────────────────────────────────
# 5. Public API — same signature as before
# ────────────────────────────────────────────────────────────────────────────
def build_explanation(criterion_name, pred_norm, score, max_score, signals,
                      question=None, answer=None, criterion_desc=None):
    global _llm_failed

    if DISABLE_LLM or _llm_failed or not HF_TOKEN:
        return _rule_based_explanation(criterion_name, pred_norm, score, max_score, signals)
    if question is None or answer is None:
        return _rule_based_explanation(criterion_name, pred_norm, score, max_score, signals)

    messages = _build_messages(
        criterion_name=criterion_name,
        score=score,
        max_score=max_score,
        question=question,
        answer=answer,
        signals=signals,
        criterion_desc=criterion_desc,
    )

    try:
        raw = _call_hf_chat(messages)
        return _clean_explanation(raw)
    except Exception as e:
        # Latch the failure flag on auth-style errors so we stop retrying.
        msg = str(e)
        if "401" in msg or "403" in msg or "HF_TOKEN" in msg:
            with _failed_lock:
                _llm_failed = True
            logger.warning(
                "Disabling LLM explainer for the rest of this process: %s", e
            )
        else:
            logger.warning("LLM call failed (%s) — using rule-based fallback this request.", e)
        return _rule_based_explanation(criterion_name, pred_norm, score, max_score, signals)


# ────────────────────────────────────────────────────────────────────────────
# 6. Signal normalisation helper (unchanged)
# ────────────────────────────────────────────────────────────────────────────
def normalize_signals(signals):
    if signals is None:
        signals = {}
    missed = signals.get("missed_answer_tokens", [])
    active = signals.get("active_rubric_concepts", [])
    if "unattended_concepts" not in signals:
        signals["unattended_concepts"] = missed or active or []
    signals.setdefault("top_answer_tokens", [])
    signals.setdefault("confidence", 0.0)
    signals.setdefault("source", "cross_attention")
    return signals
