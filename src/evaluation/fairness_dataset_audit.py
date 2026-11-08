import argparse
import csv
import json
from pathlib import Path

from src.evaluation.fairness_audit import audit_group_distributions, evaluate_group_performance


def main():
    parser = argparse.ArgumentParser(description="Audit representation by state, language, court, document type, district, and case category.")
    parser.add_argument("--input", default="data/test/tldc_test.jsonl")
    parser.add_argument("--output", default="reports/fairness_dataset_audit.csv")
    parser.add_argument("--predictions", default="data/test/group_predictions.jsonl")
    parser.add_argument("--performance-output", default="results/fairness_performance.csv")
    parser.add_argument("--minimum-group-size", type=int, default=10)
    args = parser.parse_args()
    records = []
    input_path = Path(args.input)
    if input_path.exists():
        with input_path.open(encoding="utf-8") as source:
            records = [json.loads(line) for line in source if line.strip()]
    rows = audit_group_distributions(records, min_group_size=args.minimum_group_size)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["dimension", "group", "count", "class_distribution", "sample_adequacy", "interpretation"]
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            row["class_distribution"] = json.dumps(row["class_distribution"], ensure_ascii=False)
            writer.writerow(row)
    prediction_path = Path(args.predictions)
    prediction_records = []
    if prediction_path.exists():
        with prediction_path.open(encoding="utf-8") as source:
            prediction_records = [json.loads(line) for line in source if line.strip()]
    performance_rows = []
    if prediction_records and all("label" in row and "prediction" in row for row in prediction_records):
        performance_rows = evaluate_group_performance(
            prediction_records,
            [row["prediction"] for row in prediction_records],
            min_group_size=args.minimum_group_size,
        )
    performance_path = Path(args.performance_output)
    performance_path.parent.mkdir(parents=True, exist_ok=True)
    performance_columns = ["dimension", "group", "count", "accuracy", "macro_f1", "weighted_f1", "macro_recall", "macro_precision", "sample_adequacy", "interpretation"]
    with performance_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=performance_columns)
        writer.writeheader()
        writer.writerows(performance_rows)
    print(json.dumps({"input_records": len(records), "audit_rows": len(rows), "prediction_records": len(prediction_records), "performance_rows": len(performance_rows)}, indent=2))


if __name__ == "__main__":
    main()
