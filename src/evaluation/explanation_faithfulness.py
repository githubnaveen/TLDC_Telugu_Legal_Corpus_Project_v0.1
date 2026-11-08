def _score(predict_proba, text, target_index):
    prediction = predict_proba([text])[0]
    return float(prediction[target_index])


def _replace_spans(text, spans, keep_selected=False):
    chars = list(text)
    selected = set()
    for span in spans:
        start, end = span.get("start"), span.get("end")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
            selected.update(range(start, end))
    for index, char in enumerate(chars):
        if keep_selected != (index in selected) and char not in "\r\n":
            chars[index] = " "
    return "".join(chars)


def evaluate_removal_faithfulness(predict_proba, text, target_index, attributions, top_k=5):
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    valid = [
        item for item in attributions
        if isinstance(item.get("start"), int)
        and isinstance(item.get("end"), int)
        and 0 <= item["start"] < item["end"] <= len(text)
    ]
    ranked = sorted(valid, key=lambda item: abs(float(item.get("score", 0.0))), reverse=True)
    high = ranked[:top_k]
    remaining = [item for item in ranked if item not in high]
    low = remaining[-top_k:] if remaining else []
    original_score = _score(predict_proba, text, target_index)
    high_removed_score = _score(predict_proba, _replace_spans(text, high), target_index)
    low_removed_score = _score(predict_proba, _replace_spans(text, low), target_index)
    evidence_only_score = _score(predict_proba, _replace_spans(text, high, keep_selected=True), target_index)
    high_drop = original_score - high_removed_score
    low_drop = original_score - low_removed_score
    return {
        "target_score_original": original_score,
        "target_score_after_high_attribution_removal": high_removed_score,
        "target_score_after_low_attribution_removal": low_removed_score,
        "target_score_with_high_attribution_only": evidence_only_score,
        "comprehensiveness": high_drop,
        "low_attribution_drop": low_drop,
        "high_vs_low_removal_gap": high_drop - low_drop,
        "sufficiency_score_drop": original_score - evidence_only_score,
        "perturbation": "replace selected character spans with spaces; may be linguistically unnatural",
        "selected_high_span_count": len(high),
        "selected_low_span_count": len(low),
    }
