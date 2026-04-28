"""
Attention-signal extraction — derived from the notebook's
`extract_signals_from_attn` + `decode_clean_tokens` helpers.

Given the three cross-attention tensors (qc, qa, ac) produced by the
scoring model, this module surfaces four human-interpretable signals:

    top_answer_tokens      — answer tokens most aligned with the rubric
    missed_answer_tokens   — tokens that were "important" but under-attended
    active_rubric_concepts — rubric terms the answer engaged with selectively
    confidence             — aggregate confidence score in [0, 1]
"""
import torch


SPECIAL_TOKENS = {"", "<s>", "</s>", "<pad>", "<criterion>", "<score>"}


def decode_clean_tokens(text, tokenizer, max_length=512):
    """Tokenise text and return a list of stripped, special-free subword tokens."""
    ids = tokenizer(text, max_length=max_length, truncation=True)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    special = set(tokenizer.all_special_tokens)

    clean = []
    for t in tokens:
        if t in special or t.strip() in ["", "\u2581"]:
            continue
        cleaned = t.replace("\u2581", "").replace("\u0120", "").strip()
        if cleaned:
            clean.append(cleaned)
    return clean


def extract_signals_from_attn(attn_dict, answer_tokens, criterion_tokens,
                              question_tokens, top_k=8):
    """
    attn_dict shapes (matching CriterionWiseScoringSystem):
      "qc" : (1, n_heads, q_len,  c_len)  — Q attending to criterion
      "qa" : (1, n_heads, q_len,  a_len)  — Q_cond attending to answer
      "ac" : (1, n_heads, a_len,  c_len)  — answer attending to criterion
    """

    def avg_heads(attn):
        # Average across attention heads, batch index = 0 (single sample).
        return attn[0].mean(dim=0).cpu()

    ac = avg_heads(attn_dict["ac"])  # (a_len, c_len) — answer is the query
    qa = avg_heads(attn_dict["qa"])  # (q_len, a_len) — Q_cond is the query

    # ── 1. Top answer tokens ────────────────────────────────────────────────
    # Weight each criterion column by how selective it is (variance across
    # answer rows), then sum to get per-answer-token importance.
    c_col_variance = ac.var(dim=0, keepdim=True).clamp(min=1e-9)
    c_weights = c_col_variance / c_col_variance.sum()
    a_importance = (ac * c_weights).sum(dim=1)
    a_importance = a_importance / (a_importance.sum() + 1e-9)

    n_ans = min(len(answer_tokens), a_importance.shape[0])
    top_ans_idx = a_importance[:n_ans].topk(min(top_k, n_ans)).indices.tolist()
    top_answer_tokens = [
        {"token": answer_tokens[i], "importance": round(float(a_importance[i]), 4)}
        for i in sorted(top_ans_idx, key=lambda i: -float(a_importance[i]))
        if answer_tokens[i].strip() not in SPECIAL_TOKENS
    ]

    # ── 2. Missed answer tokens ─────────────────────────────────────────────
    # Tokens that scored high in `a_importance` but low in Q_cond coverage
    # → things the rubric wanted but the question-pathway under-attended to.
    a_coverage_by_q = qa.sum(dim=0)
    a_coverage_by_q = a_coverage_by_q / (a_coverage_by_q.sum() + 1e-9)

    n = min(n_ans, a_coverage_by_q.shape[0])
    gap_score = (a_importance[:n] - a_coverage_by_q[:n]).clamp(min=0)
    gap_idx = gap_score.topk(min(top_k, n)).indices.tolist()

    missed_answer_tokens = [
        answer_tokens[i]
        for i in sorted(gap_idx, key=lambda i: -float(gap_score[i]))
        if answer_tokens[i].strip() not in SPECIAL_TOKENS
        and float(gap_score[i]) > 0.01
    ][:5]

    # ── 3. Active rubric concepts ───────────────────────────────────────────
    # Criterion columns with high variance across answer rows → rubric terms
    # that the answer attended to selectively (not uniform filler).
    n_crit = min(len(criterion_tokens), ac.shape[1])
    crit_selectivity = ac[:, :n_crit].var(dim=0)
    top_crit_idx = crit_selectivity.topk(min(5, n_crit)).indices.tolist()

    active_rubric_concepts = [
        criterion_tokens[i]
        for i in sorted(top_crit_idx, key=lambda i: -float(crit_selectivity[i]))
        if criterion_tokens[i].strip() not in SPECIAL_TOKENS
    ][:4]

    # ── 4. Confidence ───────────────────────────────────────────────────────
    # Blend of importance peakiness and importance/coverage agreement.
    peak_ratio = (
        float(a_importance[:n_ans].max()) /
        (float(a_importance[:n_ans].mean()) + 1e-9)
    )
    peak_conf = min(peak_ratio / 10.0, 1.0)
    agreement = float((a_importance[:n] * a_coverage_by_q[:n]).sum())
    confidence = round(0.6 * peak_conf + 0.4 * agreement, 4)

    return {
        "top_answer_tokens": top_answer_tokens,
        "missed_answer_tokens": missed_answer_tokens,
        "active_rubric_concepts": active_rubric_concepts,
        "confidence": confidence,
        "source": "cross_attention",
    }
