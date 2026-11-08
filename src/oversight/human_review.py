from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from itertools import combinations

from src.evaluation.annotation_agreement import cohens_kappa

REVIEW_LABELS = {
    "ACCEPTED",
    "ACCEPTED_WITH_CORRECTION",
    "REJECTED",
    "INSUFFICIENT_EVIDENCE",
    "INCORRECT_CITATION",
    "UNSUPPORTED_CLAIM",
    "LEGAL_EXPERT_REVIEW_REQUIRED",
}


def validate_review(record):
    answer_id = str(record.get("answer_id", "")).strip()
    reviewer_id = str(record.get("reviewer_id", "")).strip()
    label = str(record.get("review_label", "")).strip().upper()
    if not answer_id or not reviewer_id:
        raise ValueError("answer_id and reviewer_id are required")
    if label not in REVIEW_LABELS:
        raise ValueError("review_label must be one of the controlled review labels")
    if label == "ACCEPTED_WITH_CORRECTION" and not str(record.get("reviewer_comment", "")).strip():
        raise ValueError("corrected answers require a reviewer_comment")
    validated = dict(record)
    validated["review_label"] = label
    validated.setdefault("reviewed_at", datetime.now(timezone.utc).isoformat())
    validated.setdefault("review_status", "COMPLETED")
    return validated


def build_review_summary(reviews):
    counts = Counter(str(review.get("review_label", "UNKNOWN")).upper() for review in reviews)
    return {
        "review_count": len(reviews),
        "label_counts": dict(sorted(counts.items())),
        "distinct_answers": len({review.get("answer_id") for review in reviews}),
        "reviewers": sorted({str(review.get("reviewer_id", "")) for review in reviews if review.get("reviewer_id")}),
    }


def pairwise_review_agreement(reviews):
    by_answer = defaultdict(dict)
    for review in reviews:
        answer_id = str(review.get("answer_id", ""))
        reviewer_id = str(review.get("reviewer_id", ""))
        label = str(review.get("review_label", "")).upper()
        if answer_id and reviewer_id and label in REVIEW_LABELS:
            by_answer[answer_id][reviewer_id] = label
    reviewer_pairs = defaultdict(lambda: {"left": [], "right": []})
    for reviewers in by_answer.values():
        reviewer_ids = sorted(reviewers)
        for left, right in combinations(reviewer_ids, 2):
            pair = reviewer_pairs[(left, right)]
            pair["left"].append(reviewers[left])
            pair["right"].append(reviewers[right])
    if not reviewer_pairs:
        return {"paired_answer_count": 0, "cohens_kappa": None, "agreement": None}
    kappas = []
    agreements = []
    paired_answers = 0
    for values in reviewer_pairs.values():
        labels_a = values["left"]
        labels_b = values["right"]
        paired_answers += len(labels_a)
        agreements.append(sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a))
        kappas.append(cohens_kappa(labels_a, labels_b))
    return {
        "paired_answer_count": paired_answers,
        "reviewer_pair_count": len(kappas),
        "cohens_kappa": sum(kappas) / len(kappas),
        "agreement": sum(agreements) / len(agreements),
    }


def append_review(review, output_path):
    validated = validate_review(review)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(validated, ensure_ascii=False) + "\n")
    return validated
