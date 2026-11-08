from collections import Counter, defaultdict

from sklearn.metrics import f1_score, precision_score, recall_score


def _group_value(row, field):
    value = row.get(field)
    return str(value).strip() if value is not None and str(value).strip() else "UNKNOWN"


def audit_group_distributions(rows, dimensions=None, min_group_size=10):
    if min_group_size < 1:
        raise ValueError("min_group_size must be positive")
    dimensions = dimensions or ["state", "language", "court_level", "document_type", "district", "case_category"]
    results = []
    for dimension in dimensions:
        grouped = defaultdict(list)
        for row in rows:
            grouped[_group_value(row, dimension)].append(row)
        for group, records in sorted(grouped.items()):
            labels = Counter(str(row.get("label", row.get("document_type", "UNKNOWN"))) for row in records)
            results.append({
                "dimension": dimension,
                "group": group,
                "count": len(records),
                "class_distribution": dict(sorted(labels.items())),
                "sample_adequacy": "ADEQUATE_FOR_SCREENING" if len(records) >= min_group_size else "INSUFFICIENT",
                "interpretation": "Descriptive dataset/performance difference only; not evidence of demographic bias or unjust treatment.",
            })
    return results


def evaluate_group_performance(rows, predictions, dimensions=None, min_group_size=10):
    if len(rows) != len(predictions):
        raise ValueError("Rows and predictions must have equal lengths")
    dimensions = dimensions or ["state", "language", "court_level", "document_type", "district", "case_category"]
    results = []
    for dimension in dimensions:
        grouped = defaultdict(lambda: {"rows": [], "predictions": []})
        for row, prediction in zip(rows, predictions):
            grouped[_group_value(row, dimension)]["rows"].append(row)
            grouped[_group_value(row, dimension)]["predictions"].append(prediction)
        for group, values in sorted(grouped.items()):
            expected = [row["label"] for row in values["rows"]]
            results.append({
                "dimension": dimension,
                "group": group,
                "count": len(expected),
                "accuracy": sum(truth == guess for truth, guess in zip(expected, values["predictions"])) / len(expected) if expected else None,
                "macro_f1": f1_score(expected, values["predictions"], average="macro", zero_division=0) if expected else None,
                "weighted_f1": f1_score(expected, values["predictions"], average="weighted", zero_division=0) if expected else None,
                "macro_recall": recall_score(expected, values["predictions"], average="macro", zero_division=0) if expected else None,
                "macro_precision": precision_score(expected, values["predictions"], average="macro", zero_division=0) if expected else None,
                "sample_adequacy": "ADEQUATE_FOR_SCREENING" if len(expected) >= min_group_size else "INSUFFICIENT",
                "interpretation": "Observed disparity requires case-mix, collection, annotation, and legal-context review; it is not itself proof of bias.",
            })
    return results
