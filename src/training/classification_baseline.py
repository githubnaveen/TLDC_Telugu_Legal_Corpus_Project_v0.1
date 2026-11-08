import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Union

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from src.evaluation.regional_evaluation import evaluate_by_group


def load_task_rows(path: Union[str, Path], task: str) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                record = json.loads(line)
                if record.get("task") == task and record.get("text") and record.get("label"):
                    rows.append(record)
    return rows


def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, max_features=100000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])


def train_evaluate(train_rows: list[dict], test_rows: list[dict]) -> dict:
    if not train_rows or not test_rows:
        raise ValueError("Training and test data must both contain reviewed examples")
    labels = {row["label"] for row in train_rows}
    if len(labels) < 2:
        raise ValueError("At least two training labels are required")
    unknown = {row["label"] for row in test_rows} - labels
    if unknown:
        raise ValueError(f"Test set contains labels absent from training: {sorted(unknown)}")

    model = build_model()
    model.fit([row["text"] for row in train_rows], [row["label"] for row in train_rows])
    expected = [row["label"] for row in test_rows]
    predicted = model.predict([row["text"] for row in test_rows])
    label_order = sorted(set(expected) | set(predicted))
    state_metrics = evaluate_by_group(test_rows, predicted, "state")
    language_metrics = evaluate_by_group(test_rows, predicted, "language")
    return {
        "count": len(test_rows),
        "accuracy": accuracy_score(expected, predicted),
        "macro_f1": f1_score(expected, predicted, average="macro", zero_division=0),
        "weighted_f1": f1_score(expected, predicted, average="weighted", zero_division=0),
        "labels": label_order,
        "confusion_matrix": confusion_matrix(expected, predicted, labels=label_order).tolist(),
        "per_class": classification_report(expected, predicted, output_dict=True, zero_division=0),
        "class_distribution": dict(Counter(expected)),
        "by_state": state_metrics,
        "by_language": language_metrics,
        "model": "TF-IDF character ngrams + LogisticRegression(class_weight=balanced, max_iter=1000)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a classical TLDC text classifier.")
    parser.add_argument("--task", choices=["document_classification", "section_classification"], required=True)
    parser.add_argument("--train", default="data/train/tldc_train.jsonl")
    parser.add_argument("--test", default="data/test/tldc_test.jsonl")
    parser.add_argument("--output", default="results/classification_results.json")
    parser.add_argument("--model-output", default=None, help="Optional path to save the fitted sklearn pipeline with joblib")
    args = parser.parse_args()
    train_rows = load_task_rows(args.train, args.task)
    test_rows = load_task_rows(args.test, args.task)
    result = train_evaluate(train_rows, test_rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    import csv

    summary_path = output.parent / ("classification_results.csv" if args.task == "document_classification" else "section_classification_results.csv")
    with summary_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=["task", "model", "count", "accuracy", "macro_f1", "weighted_f1"])
        writer.writeheader()
        writer.writerow({"task": args.task, "model": result["model"], **{key: result[key] for key in ("count", "accuracy", "macro_f1", "weighted_f1")}})
    if args.model_output:
        import joblib

        model = build_model()
        model.fit([row["text"] for row in train_rows], [row["label"] for row in train_rows])
        model_path = Path(args.model_output)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
    print(json.dumps({key: result[key] for key in ("count", "accuracy", "macro_f1", "weighted_f1")}, indent=2))


if __name__ == "__main__":
    main()
