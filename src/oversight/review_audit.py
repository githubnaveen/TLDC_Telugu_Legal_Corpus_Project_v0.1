import argparse
import csv
import json
from pathlib import Path

from src.oversight.human_review import REVIEW_LABELS, pairwise_review_agreement


def main():
    parser = argparse.ArgumentParser(description="Summarize controlled human-review labels and independent reviewer agreement.")
    parser.add_argument("--input", default="evaluation/human_review.csv")
    parser.add_argument("--output", default="results/inter_annotator_agreement.json")
    args = parser.parse_args()
    input_path = Path(args.input)
    records = []
    if input_path.exists():
        with input_path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                label = str(row.get("review_label", "")).upper().strip()
                if label in REVIEW_LABELS and row.get("answer_id") and row.get("reviewer_id"):
                    records.append({**row, "review_label": label})
    result = pairwise_review_agreement(records)
    result["review_count"] = len(records)
    result["status"] = "COMPUTED" if records else "NO_INDEPENDENT_REVIEWS"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
