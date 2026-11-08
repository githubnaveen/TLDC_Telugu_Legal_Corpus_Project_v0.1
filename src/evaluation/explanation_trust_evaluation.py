from src.evaluation.evidence_coverage import calculate_evidence_coverage
from src.evaluation.explanation_stability import top_feature_jaccard


def evaluate_explanation_records(records):
    results = []
    for record in records:
        explanation = record.get("explanation", {})
        segments = explanation.get("important_segments", [])
        review = record.get("human_review", {})
        faithfulness = record.get("faithfulness_evaluation", {})
        alternate = record.get("counterfactual_explanation")
        stability = None
        if alternate is not None:
            first = [(segment.get("text", ""), segment.get("attribution_score", 0.0)) for segment in segments]
            second = [(segment.get("text", ""), segment.get("attribution_score", 0.0)) for segment in alternate.get("important_segments", [])]
            stability = top_feature_jaccard(first, second, k=min(10, max(1, len(first), len(second))))
        results.append({
            "answer_id": record.get("answer_id", record.get("document_id", "")),
            "explanation_method": explanation.get("method", "UNKNOWN"),
            "highlighted_segment_count": len(segments),
            "evidence_coverage": calculate_evidence_coverage(record),
            "stability_jaccard": stability,
            "comprehensiveness": faithfulness.get("comprehensiveness"),
            "sufficiency_score_drop": faithfulness.get("sufficiency_score_drop"),
            "high_vs_low_removal_gap": faithfulness.get("high_vs_low_removal_gap"),
            "human_review_status": review.get("status", "PENDING"),
            "human_interpretability": review.get("interpretability", "NOT_REVIEWED"),
            "claim_support_review": review.get("claim_support", "NOT_REVIEWED"),
        })
    return results


def main():
    import argparse
    import csv
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Summarize evidence coverage and human review for explanation records.")
    parser.add_argument("--input", default="xai/evidence_attribution/evidence_attribution.jsonl")
    parser.add_argument("--output", default="results/explanation_evaluation.csv")
    args = parser.parse_args()
    input_path = Path(args.input)
    records = []
    if input_path.exists():
        with input_path.open(encoding="utf-8") as source:
            records = [json.loads(line) for line in source if line.strip()]
    rows = evaluate_explanation_records(records)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["answer_id", "explanation_method", "highlighted_segment_count", "evidence_coverage", "stability_jaccard", "comprehensiveness", "sufficiency_score_drop", "high_vs_low_removal_gap", "human_review_status", "human_interpretability", "claim_support_review"]
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print("Explanation records evaluated:", len(rows))


if __name__ == "__main__":
    main()
