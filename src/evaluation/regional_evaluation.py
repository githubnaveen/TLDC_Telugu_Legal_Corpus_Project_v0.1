from collections import defaultdict

from sklearn.metrics import accuracy_score, f1_score


def evaluate_by_group(rows, predictions, group_field):
    if len(rows) != len(predictions):
        raise ValueError("Rows and predictions must have equal lengths")
    grouped = defaultdict(lambda: {"expected": [], "predicted": []})
    for row, prediction in zip(rows, predictions):
        group = row.get(group_field) or "UNKNOWN"
        grouped[group]["expected"].append(row["label"])
        grouped[group]["predicted"].append(prediction)
    return {
        group: {
            "count": len(values["expected"]),
            "accuracy": accuracy_score(values["expected"], values["predicted"]),
            "macro_f1": f1_score(values["expected"], values["predicted"], average="macro", zero_division=0),
        }
        for group, values in sorted(grouped.items())
    }
