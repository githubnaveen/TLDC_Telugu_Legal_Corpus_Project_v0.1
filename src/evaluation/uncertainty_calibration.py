import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar


def _validated_probabilities(y_true, probabilities, class_labels=None):
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim == 1:
        matrix = np.column_stack((1.0 - matrix, matrix))
    raw_labels = list(y_true)
    if class_labels is not None:
        if len(class_labels) != matrix.shape[1]:
            raise ValueError("class_labels length must match probability columns")
        label_to_index = {str(label): index for index, label in enumerate(class_labels)}
        try:
            labels = np.asarray([label_to_index[str(label)] for label in raw_labels], dtype=int)
        except KeyError as error:
            raise ValueError("y_true contains a label missing from class_labels") from error
    elif all(isinstance(label, (int, np.integer)) and not isinstance(label, bool) for label in raw_labels):
        labels = np.asarray(raw_labels, dtype=int)
    else:
        raise ValueError("class_labels in probability-column order are required for non-integer labels")
    if matrix.ndim != 2 or matrix.shape[0] != len(labels) or matrix.shape[1] < 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes)")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0) or np.any(matrix > 1):
        raise ValueError("probabilities must be finite values in [0, 1]")
    row_sums = matrix.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("each probability row must have positive mass")
    matrix = matrix / row_sums[:, None]
    if np.any(labels < 0) or np.any(labels >= matrix.shape[1]):
        raise ValueError("true labels must be integer class indices matching probability columns")
    return labels, matrix


def brier_score(y_true, probabilities, class_labels=None):
    labels, matrix = _validated_probabilities(y_true, probabilities, class_labels)
    targets = np.eye(matrix.shape[1], dtype=float)[labels]
    if matrix.shape[1] == 2:
        return float(np.mean((matrix[:, 1] - targets[:, 1]) ** 2))
    return float(np.mean(np.sum((matrix - targets) ** 2, axis=1)))


def expected_calibration_error(y_true, probabilities, bins=10, class_labels=None):
    if bins < 1:
        raise ValueError("bins must be positive")
    labels, matrix = _validated_probabilities(y_true, probabilities, class_labels)
    confidence = matrix.max(axis=1)
    predicted = matrix.argmax(axis=1)
    correct = (predicted == labels).astype(float)
    ece = 0.0
    bin_rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            selected = (confidence >= lower) & (confidence <= upper)
        else:
            selected = (confidence >= lower) & (confidence < upper)
        count = int(selected.sum())
        if count:
            mean_confidence = float(confidence[selected].mean())
            accuracy = float(correct[selected].mean())
            ece += count / len(labels) * abs(accuracy - mean_confidence)
        else:
            mean_confidence = None
            accuracy = None
        bin_rows.append({"bin": index, "lower": float(lower), "upper": float(upper), "count": count, "mean_confidence": mean_confidence, "accuracy": accuracy})
    return float(ece), bin_rows


def fit_temperature(y_true, probabilities, class_labels=None):
    labels, matrix = _validated_probabilities(y_true, probabilities, class_labels)
    if len(labels) == 0:
        raise ValueError("temperature scaling requires validation examples")
    logits = np.log(np.clip(matrix, 1e-12, 1.0))

    def objective(log_temperature):
        temperature = math.exp(float(log_temperature))
        scaled = logits / temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        exp_values = np.exp(scaled)
        calibrated = exp_values / exp_values.sum(axis=1, keepdims=True)
        return float(-np.log(np.clip(calibrated[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())

    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    if not result.success:
        raise RuntimeError("temperature optimization did not converge")
    return float(math.exp(result.x))


def apply_temperature(probabilities, temperature):
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim == 1:
        matrix = np.column_stack((1.0 - matrix, matrix))
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    logits = np.log(np.clip(matrix, 1e-12, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    exp_values = np.exp(logits)
    return (exp_values / exp_values.sum(axis=1, keepdims=True)).tolist()


def evaluate_calibration(y_true, probabilities, bins=10, temperature=None, class_labels=None):
    labels, matrix = _validated_probabilities(y_true, probabilities, class_labels)
    if len(labels) == 0:
        raise ValueError("calibration metrics require held-out labeled examples")
    raw_ece, raw_bins = expected_calibration_error(labels, matrix, bins)
    report = {
        "sample_count": len(labels),
        "raw_expected_calibration_error": raw_ece,
        "raw_brier_score": brier_score(labels, matrix),
        "temperature": temperature,
        "raw_reliability_bins": raw_bins,
        "confidence_source": "provided classifier probabilities; not applicable to uncalibrated LLM verbal confidence",
    }
    if temperature is not None:
        scaled = apply_temperature(matrix, temperature)
        calibrated_ece, calibrated_bins = expected_calibration_error(labels, scaled, bins)
        report["calibrated_expected_calibration_error"] = calibrated_ece
        report["calibrated_brier_score"] = brier_score(labels, scaled)
        report["calibrated_reliability_bins"] = calibrated_bins
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate probability calibration on held-out, labeled predictions.")
    parser.add_argument("--calibration-predictions", default="data/calibration_predictions.jsonl")
    parser.add_argument("--test-predictions", default="data/test_calibration_predictions.jsonl")
    parser.add_argument("--report", default="reports/calibration_report.md")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    def read_predictions(path):
        prediction_path = Path(path)
        if not prediction_path.exists():
            return []
        with prediction_path.open(encoding="utf-8") as source:
            return [json.loads(line) for line in source if line.strip()]

    calibration_records = read_predictions(args.calibration_predictions)
    test_records = read_predictions(args.test_predictions)
    if not calibration_records or not test_records:
        print("Separate labeled calibration and held-out test predictions are required; calibration metrics unavailable.")
        return
    temperature = fit_temperature(
        [row["true_label"] for row in calibration_records],
        [row["probabilities"] for row in calibration_records],
        class_labels=calibration_records[0].get("class_labels"),
    )
    report = evaluate_calibration(
        [row["true_label"] for row in test_records],
        [row["probabilities"] for row in test_records],
        args.bins,
        temperature=temperature,
        class_labels=test_records[0].get("class_labels"),
    )
    report["calibration_fit_count"] = len(calibration_records)
    output_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("# Calibration report\n\n" + json.dumps(report, indent=2), encoding="utf-8")
    bins_path = output_path.parent / "calibration_bins.csv"
    bin_rows = []
    for phase in ("raw", "calibrated"):
        for row in report.get(phase + "_reliability_bins", []):
            bin_rows.append({"phase": phase, **row})
    with bins_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["phase", "bin", "lower", "upper", "count", "mean_confidence", "accuracy"])
        writer.writeheader()
        writer.writerows(bin_rows)
    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        for phase in ("raw", "calibrated"):
            rows = report.get(phase + "_reliability_bins", [])
            populated = [row for row in rows if row["count"]]
            if populated:
                axis.plot([row["mean_confidence"] for row in populated], [row["accuracy"] for row in populated], marker="o", label=phase)
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
        axis.set(xlabel="Mean confidence", ylabel="Accuracy", title="Reliability diagram")
        axis.legend()
        figure.savefig(output_path.parent / "calibration_reliability.png", dpi=150, bbox_inches="tight")
        plt.close(figure)
    except ImportError:
        pass
    print(json.dumps({key: report.get(key) for key in ("sample_count", "calibration_fit_count", "raw_expected_calibration_error", "calibrated_expected_calibration_error", "raw_brier_score", "calibrated_brier_score", "temperature")}, indent=2))


if __name__ == "__main__":
    main()
