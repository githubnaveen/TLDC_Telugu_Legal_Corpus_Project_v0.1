def evaluate_detection(records, detector):
    true_positive = false_positive = true_negative = false_negative = 0
    details = []
    for record in records:
        expected = bool(record.get("expected_suspicious"))
        result = detector(record.get("text", ""))
        predicted = bool(result.get("suspicious"))
        true_positive += int(expected and predicted)
        false_positive += int(not expected and predicted)
        true_negative += int(not expected and not predicted)
        false_negative += int(expected and not predicted)
        details.append({
            "test_id": record.get("test_id", ""),
            "expected_suspicious": expected,
            "detected": predicted,
            "categories": ";".join(result.get("categories", [])),
            "correct": expected == predicted,
        })
    total_attacks = true_positive + false_negative
    clean_count = true_negative + false_positive
    return {
        "test_count": len(records),
        "attack_count": total_attacks,
        "clean_count": clean_count,
        "attack_detection_rate": true_positive / total_attacks if total_attacks else None,
        "attack_detection_miss_rate": false_negative / total_attacks if total_attacks else None,
        "attack_success_rate": None,
        "false_positive_rate": false_positive / clean_count if clean_count else None,
        "details": details,
        "limitations": "Static heuristic evaluation on controlled test strings only. Detector misses are not end-to-end attack success; attack success requires observing a defined policy violation in a configured generation system.",
    }


def main():
    import argparse
    import csv
    import json
    from pathlib import Path

    from src.security.prompt_injection_detector import scan_untrusted_content

    parser = argparse.ArgumentParser(description="Evaluate static prompt-injection flags on isolated test strings.")
    parser.add_argument("--input", default="adversarial_tests/prompt_injection/prompt_injection_test_dataset.jsonl")
    parser.add_argument("--output", default="results/prompt_injection_evaluation.csv")
    parser.add_argument("--summary", default="results/prompt_injection_metrics.json")
    args = parser.parse_args()
    input_path = Path(args.input)
    records = []
    if input_path.exists():
        with input_path.open(encoding="utf-8") as source:
            records = [json.loads(line) for line in source if line.strip()]
    result = evaluate_detection(records, scan_untrusted_content)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["test_id", "expected_suspicious", "detected", "categories", "correct"]
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result["details"])
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {key: value for key, value in result.items() if key != "details"}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
