import argparse
import csv
import json
from pathlib import Path

from src.evaluation.rag_metrics import abstention_metrics, citation_precision, claim_support_rate, unsupported_claim_rate
from src.rag.evidence_verification import verify_citations


def evaluate_trust(answers, query_records, human_reviews):
    queries = {str(row.get("query_id", "")): row for row in query_records}
    reviews = {str(row.get("answer_id", "")): row for row in human_reviews}
    rows = []
    for answer in answers:
        answer_id = str(answer.get("answer_id") or answer.get("question_id") or "")
        query_id = str(answer.get("query_id", answer_id))
        query = queries.get(query_id, {})
        review = reviews.get(answer_id, {})
        citations = answer.get("citation_verification", {})
        if not citations:
            evidence = answer.get("retrieved_evidence", answer.get("evidence", []))
            citations = verify_citations(answer, evidence)
        claims = answer.get("claims", [])
        supported_claims = sum(1 for claim in claims if str(claim.get("human_support_status", "")).upper() == "SUPPORTED")
        unsupported_claims = sum(1 for claim in claims if str(claim.get("human_support_status", "")).upper() in {"UNSUPPORTED", "CONTRADICTED"})
        reviewed_claims = supported_claims + unsupported_claims
        expected_abstain = not bool(query.get("answerable", False))
        did_abstain = str(answer.get("status", "")).upper().startswith("ABSTAINED")
        review_abstention = str(review.get("abstention_appropriate", "")).strip().lower() in {"true", "1", "yes"}
        rows.append({
            "answer_id": answer_id,
            "query_id": query_id,
            "language": query.get("language", "UNKNOWN"),
            "answerable": bool(query.get("answerable", False)),
            "citation_precision": citation_precision(citations.get("correct_citations", 0), citations.get("total_citations", 0)),
            "claim_support_rate": claim_support_rate(supported_claims, reviewed_claims),
            "unsupported_claim_rate": unsupported_claim_rate(unsupported_claims, reviewed_claims),
            "claim_review_count": reviewed_claims,
            "abstained": did_abstain,
            "abstention_expected": expected_abstain,
            "abstention_human_review": review_abstention,
            "review_status": review.get("review_status") or review.get("review_label") or "PENDING",
        })
    abstention = abstention_metrics(
        [row["abstained"] for row in rows],
        [row["answerable"] for row in rows],
    ) if rows else {"abstention_precision": None, "abstention_recall": None}
    return rows, abstention


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Summarize trust metrics using explicitly human-reviewed answers.")
    parser.add_argument("--answers", default="results/grounded_answers.jsonl")
    parser.add_argument("--queries", default="rag_question_evidence_dataset.jsonl")
    parser.add_argument("--reviews", default="evaluation/human_review.csv")
    parser.add_argument("--output", default="results/rag_trust_evaluation.csv")
    args = parser.parse_args()
    answers = _read_jsonl(args.answers)
    queries = _read_jsonl(args.queries)
    reviews_path = Path(args.reviews)
    if reviews_path.exists():
        with reviews_path.open(encoding="utf-8-sig", newline="") as source:
            reviews = list(csv.DictReader(source))
    else:
        reviews = []
    rows, abstention = evaluate_trust(answers, queries, reviews)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["answer_id", "query_id", "language", "answerable", "citation_precision", "claim_support_rate", "unsupported_claim_rate", "claim_review_count", "abstained", "abstention_expected", "abstention_human_review", "review_status"]
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"answer_count": len(answers), "review_count": len(reviews), "evaluated_rows": len(rows), "abstention_metrics": abstention}, indent=2))


if __name__ == "__main__":
    main()
