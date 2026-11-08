from copy import deepcopy
from datetime import datetime, timezone

REVIEW_DECISIONS = {
    "ACCEPT": "ACCEPTED",
    "REJECT": "REJECTED",
    "FLAG": "FLAGGED",
}


def record_human_review(record, decision, reviewer_id, comment=None):
    normalized_decision = str(decision or "").strip().upper()
    if normalized_decision not in REVIEW_DECISIONS:
        raise ValueError("decision must be ACCEPT, REJECT, or FLAG")
    if not reviewer_id or not str(reviewer_id).strip():
        raise ValueError("reviewer_id is required")

    reviewed = deepcopy(record)
    review = reviewed.setdefault("human_review", {})
    review.update({
        "status": REVIEW_DECISIONS[normalized_decision],
        "reviewer_id": str(reviewer_id).strip(),
        "reviewer_comment": comment,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    })
    history = reviewed.setdefault("review_history", [])
    history.append(deepcopy(review))
    return reviewed
