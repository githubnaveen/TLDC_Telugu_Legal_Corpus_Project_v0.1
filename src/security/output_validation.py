from src.rag.evidence_verification import verify_citations

REQUIRED_FIELDS = {"answer_id", "query_id", "question", "answer", "claims", "limitations", "confidence", "status", "human_review_required"}
ALLOWED_STATUSES = {"ABSTAINED", "ABSTAINED_INVALID_INPUT", "ABSTAINED_UNSUPPORTED_CITATIONS", "ABSTAINED_SUSPICIOUS_RETRIEVED_CONTENT", "UNVERIFIED", "CITATIONS_VALIDATED_REQUIRES_REVIEW"}


def validate_answer_output(record, evidence):
    errors = []
    missing = sorted(REQUIRED_FIELDS - set(record))
    if missing:
        errors.append("missing_required_fields:" + ",".join(missing))
    if record.get("status") not in ALLOWED_STATUSES:
        errors.append("invalid_status")
    if record.get("human_review_required") is not True:
        errors.append("human_review_must_remain_required")
    if record.get("confidence") != "NOT_CALIBRATED":
        errors.append("confidence_must_not_claim_calibration")
    if not isinstance(record.get("claims"), list) or not isinstance(record.get("limitations"), list):
        errors.append("claims_and_limitations_must_be_lists")
    citation_check = verify_citations(record, evidence)
    if citation_check["invalid_citations"]:
        errors.append("citation_identity_or_quote_mismatch")
    if record.get("status") == "CITATIONS_VALIDATED_REQUIRES_REVIEW" and (not record.get("claims") or citation_check["total_citations"] == 0):
        errors.append("citation_validated_status_requires_cited_claims")
    if str(record.get("status", "")).startswith("ABSTAINED") and record.get("claims"):
        errors.append("abstained_record_must_not_present_claims")
    return {
        "valid": not errors,
        "errors": errors,
        "invalid_citation_count": len(citation_check["invalid_citations"]),
        "citation_verification": citation_check,
    }
