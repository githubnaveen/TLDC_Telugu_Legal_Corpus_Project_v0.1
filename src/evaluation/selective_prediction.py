def risk_coverage_curve(correctness, confidence, thresholds=None):
    if len(correctness) != len(confidence):
        raise ValueError("correctness and confidence must have equal lengths")
    if thresholds is None:
        thresholds = [index / 20.0 for index in range(21)]
    if any(threshold < 0 or threshold > 1 for threshold in thresholds):
        raise ValueError("thresholds must be in [0, 1]")
    if any(score < 0 or score > 1 for score in confidence):
        raise ValueError("confidence scores must be in [0, 1]")
    total = len(correctness)
    results = []
    for threshold in sorted(set(float(value) for value in thresholds)):
        accepted = [bool(correct) for correct, score in zip(correctness, confidence) if score >= threshold]
        accepted_count = len(accepted)
        errors = sum(not correct for correct in accepted)
        results.append({
            "threshold": threshold,
            "accepted_count": accepted_count,
            "coverage": accepted_count / total if total else None,
            "selective_risk": errors / accepted_count if accepted_count else None,
            "error_count": errors,
            "total_count": total,
        })
    return results


def main():
    import argparse
    import csv
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Compute selective risk and coverage from labeled validation predictions.")
    parser.add_argument("--input", default="data/selective_predictions.jsonl")
    parser.add_argument("--output", default="results/risk_coverage_report.csv")
    parser.add_argument("--thresholds", default="0,0.25,0.5,0.75,0.9")
    args = parser.parse_args()
    input_path = Path(args.input)
    records = []
    if input_path.exists():
        with input_path.open(encoding="utf-8") as source:
            records = [json.loads(line) for line in source if line.strip()]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["threshold", "accepted_count", "coverage", "selective_risk", "error_count", "total_count"]
    rows = []
    if records:
        thresholds = [float(value) for value in args.thresholds.split(",") if value.strip()]
        rows = risk_coverage_curve(
            [bool(row["correct"]) for row in records],
            [float(row["confidence"]) for row in records],
            thresholds,
        )
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print("No labeled validation predictions; risk-coverage values unavailable." if not records else "Risk-coverage rows: {}".format(len(rows)))


if __name__ == "__main__":
    main()
