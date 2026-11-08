from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def evaluate_predictions(expected, predicted):
    if len(expected) != len(predicted):
        raise ValueError("Expected and predicted labels must have equal lengths")
    labels = sorted(set(expected) | set(predicted))
    if not labels:
        raise ValueError("At least one evaluated example is required")
    return {
        "accuracy": accuracy_score(expected, predicted),
        "macro_f1": f1_score(expected, predicted, average="macro", zero_division=0),
        "weighted_f1": f1_score(expected, predicted, average="weighted", zero_division=0),
        "labels": labels,
        "confusion_matrix": confusion_matrix(expected, predicted, labels=labels).tolist(),
        "per_class": classification_report(expected, predicted, output_dict=True, zero_division=0),
    }
