import re
import argparse
import csv
import json
from pathlib import Path

from src.explainability.evidence_linker import link_attributions


def predict_and_explain(model, text, document_id, sections, model_version, max_features=30, legal_references=None):
    if not text:
        raise ValueError("text must not be empty")
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise ValueError("Prototype currently requires a classifier with predict_proba")
    if not hasattr(model, "named_steps") or "tfidf" not in model.named_steps:
        raise ValueError("Prototype currently expects a sklearn pipeline with a 'tfidf' step")

    predicted_label = str(model.predict([text])[0])
    class_names = list(model.classes_)
    class_index = class_names.index(predicted_label)
    score = float(model.predict_proba([text])[0][class_index])
    vectorizer = model.named_steps["tfidf"]
    estimator = model.steps[-1][1]
    if not hasattr(estimator, "coef_"):
        raise ValueError("Prototype local feature attribution currently requires a linear classifier")

    coefficients = estimator.coef_
    if coefficients.shape[0] == 1 and len(class_names) == 2:
        class_weights = coefficients[0] if class_index == 1 else -coefficients[0]
    else:
        class_weights = coefficients[class_index]
    vector = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    contributions = []
    for feature_index, value in zip(vector.indices, vector.data):
        feature = str(feature_names[feature_index])
        contribution = float(value * class_weights[feature_index])
        matches = list(re.finditer(re.escape(feature), text))
        if not matches:
            continue
        contribution_per_match = contribution / len(matches)
        for match in matches:
            contributions.append({
                "start": match.start(),
                "end": match.end(),
                "score": contribution_per_match,
                "feature": feature,
            })
    contributions.sort(key=lambda item: abs(item["score"]), reverse=True)

    return link_attributions(
        document_id=document_id,
        model_text=text,
        sections=sections,
        attributions=contributions[:max_features],
        predicted_label=predicted_label,
        model_version=model_version,
        method="LINEAR_TFIDF_FEATURE_CONTRIBUTIONS",
        score=score,
        legal_references=legal_references,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate page-linked local explanations for prepared held-out TLDC documents.")
    parser.add_argument("--model", default="models/classical_ml/document_classifier.joblib")
    parser.add_argument("--input", default="data/test/tldc_test.jsonl")
    parser.add_argument("--segments", default="annotations/document_segmentation/segmented_documents.jsonl")
    parser.add_argument("--statutes", default="annotations/statutory_references/statutory_references.csv")
    parser.add_argument("--output", default="xai/evidence_attribution/evidence_attribution.jsonl")
    parser.add_argument("--model-version", required=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model artifact not found: {model_path}")
    with Path(args.input).open(encoding="utf-8") as source:
        test_rows = [
            json.loads(line) for line in source
            if line.strip() and json.loads(line).get("task") == "document_classification"
        ]
    if not test_rows:
        print("No eligible held-out document examples; no explanation records written.")
        return

    documents = {}
    with Path(args.segments).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                document = json.loads(line)
                documents[document.get("document_id")] = document
    references_by_document = {}
    statute_path = Path(args.statutes)
    if statute_path.exists():
        with statute_path.open(encoding="utf-8-sig", newline="") as source:
            for reference in csv.DictReader(source):
                references_by_document.setdefault(reference.get("document_id", ""), []).append(reference)

    import joblib

    model = joblib.load(model_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for row in test_rows:
            document = documents.get(row.get("document_id"), {})
            record = predict_and_explain(
                model,
                row["text"],
                row["document_id"],
                document.get("sections", []),
                args.model_version,
                legal_references=references_by_document.get(row.get("document_id"), []),
            )
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(test_rows)} explanation records to {output_path}")


if __name__ == "__main__":
    main()
