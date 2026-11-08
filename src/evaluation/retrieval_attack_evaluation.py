import argparse
import csv
import json
from pathlib import Path

from src.security.retrieval_attack import compare_clean_and_candidate_rankings


def main():
    parser = argparse.ArgumentParser(description="Compare clean and controlled candidate retrieval rankings in isolated fixtures.")
    parser.add_argument("--input", default="adversarial_tests/retrieval/retrieval_attack_pairs.jsonl")
    parser.add_argument("--output", default="results/retrieval_attack_evaluation.csv")
    args = parser.parse_args()
    path = Path(args.input)
    pairs = []
    if path.exists():
        with path.open(encoding="utf-8") as source:
            pairs = [json.loads(line) for line in source if line.strip()]
    columns = ["test_id", "clean_relevant_recall_at_k", "candidate_relevant_recall_at_k", "recall_degradation", "top_result_changed", "duplicate_count", "single_document_dominance"]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair in pairs:
        result = compare_clean_and_candidate_rankings(
            pair.get("clean_results", []),
            pair.get("candidate_results", []),
            pair.get("relevant_chunk_ids", []),
        )
        rows.append({
            "test_id": pair.get("test_id", ""),
            "clean_relevant_recall_at_k": result["clean_relevant_recall_at_k"],
            "candidate_relevant_recall_at_k": result["candidate_relevant_recall_at_k"],
            "recall_degradation": result["recall_degradation"],
            "top_result_changed": result["top_result_changed"],
            "duplicate_count": result["candidate_retrieval_audit"]["duplicate_chunk_count"],
            "single_document_dominance": result["candidate_retrieval_audit"]["single_document_dominance"],
        })
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print("Controlled retrieval pairs evaluated:", len(rows))


if __name__ == "__main__":
    main()
