import argparse
import csv
import json
from pathlib import Path

from src.evaluation.counterfactual_fairness import evaluate_counterfactual_pairs


def main():
    parser = argparse.ArgumentParser(description="Evaluate expert-adjudicated counterfactual legal-RAG pairs.")
    parser.add_argument("--input", default="counterfactual_fairness_dataset.jsonl")
    parser.add_argument("--output", default="results/counterfactual_evaluation.csv")
    parser.add_argument("--summary", default="reports/counterfactual_summary.json")
    args = parser.parse_args()
    input_path = Path(args.input)
    pairs = []
    if input_path.exists():
        with input_path.open(encoding="utf-8") as source:
            pairs = [json.loads(line) for line in source if line.strip()]
    result = evaluate_counterfactual_pairs(pairs)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["pair_id", "conclusion_consistent", "citations_consistent", "claims_consistent", "pair_consistent", "changed_attribute", "review_status"]
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result["pairs"])
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({key: value for key, value in result.items() if key != "pairs"}, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
