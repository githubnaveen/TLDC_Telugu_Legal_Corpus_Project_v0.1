import argparse
import csv
import json
from pathlib import Path


def build_summary_rows(prompt_injection_metrics):
    if int(prompt_injection_metrics.get("test_count", 0)) == 0:
        return []
    return [{
        "attack_category": "PROMPT_INJECTION_STATIC_HEURISTIC_SYNTHETIC",
        "attempt_count": prompt_injection_metrics.get("attack_count", 0),
        "successful_attacks": None,
        "attack_success_rate": None,
        "detection_rate": prompt_injection_metrics.get("attack_detection_rate"),
        "false_positive_rate": prompt_injection_metrics.get("false_positive_rate"),
        "robust_accuracy": None,
        "notes": "Synthetic detector-only fixture. End-to-end attack success and robust accuracy not measured; see limitations in prompt_injection_metrics.json.",
    }]


def main():
    parser = argparse.ArgumentParser(description="Aggregate measured security checks without treating detector misses as end-to-end attack success.")
    parser.add_argument("--prompt-metrics", default="results/prompt_injection_metrics.json")
    parser.add_argument("--output", default="results/adversarial_robustness_results.csv")
    args = parser.parse_args()
    metric_path = Path(args.prompt_metrics)
    metrics = json.loads(metric_path.read_text(encoding="utf-8")) if metric_path.exists() else {}
    rows = build_summary_rows(metrics)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["attack_category", "attempt_count", "successful_attacks", "attack_success_rate", "detection_rate", "false_positive_rate", "robust_accuracy", "notes"]
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print("Security summary rows:", len(rows))


if __name__ == "__main__":
    main()
