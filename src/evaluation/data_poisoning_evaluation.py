import argparse
import csv
import json
from pathlib import Path

from src.security.corpus_integrity import evaluate_poisoning_records


def _read_jsonl(path):
    file_path = Path(path)
    if not file_path.exists():
        return []
    with file_path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Compare a candidate manifest with a trusted baseline in an isolated test set.")
    parser.add_argument("--trusted", default="adversarial_tests/corpus_poisoning/trusted_manifest.jsonl")
    parser.add_argument("--candidate", default="adversarial_tests/corpus_poisoning/candidate_records.jsonl")
    parser.add_argument("--output", default="results/data_poisoning_evaluation.csv")
    parser.add_argument("--summary", default="results/data_poisoning_metrics.json")
    args = parser.parse_args()
    result = evaluate_poisoning_records(_read_jsonl(args.trusted), _read_jsonl(args.candidate))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["document_id", "registered", "label_changed", "hash_mismatch", "status"]
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result["details"])
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({key: value for key, value in result.items() if key != "details"}, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
