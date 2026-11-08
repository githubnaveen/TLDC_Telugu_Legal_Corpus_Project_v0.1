def assess_evidence_sufficiency(evidence, citation_check, minimum_evidence=1, minimum_relevance=0.0, conflicts_present=False):
    reasons = []
    if conflicts_present:
        reasons.append("conflicting_sources_require_review")
    if len(evidence or []) < minimum_evidence:
        reasons.append("insufficient_retrieved_evidence")
    for item in evidence or []:
        try:
            page = int(item.get("page_number"))
        except (TypeError, ValueError):
            page = 0
        if not item.get("document_id") or page < 1 or not item.get("section_type") or not item.get("chunk_text"):
            reasons.append("evidence_missing_source_provenance")
            break
    if float(citation_check.get("citation_precision", 0.0)) < 1.0 or citation_check.get("total_citations", 0) < 1:
        reasons.append("citations_not_fully_validated")
    if evidence and max(float(item.get("rrf_score", item.get("retrieval_score", 0.0)) or 0.0) for item in evidence) < minimum_relevance:
        reasons.append("relevance_below_configured_threshold")

    if reasons:
        return {
            "decision": "ABSTAIN",
            "reasons": sorted(set(reasons)),
            "human_review_required": True,
            "confidence": "NOT_CALIBRATED",
        }
    return {
        "decision": "PRESENT_DRAFT_FOR_HUMAN_REVIEW",
        "reasons": [],
        "human_review_required": True,
        "confidence": "NOT_CALIBRATED",
    }
