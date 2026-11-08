def cohens_kappa(labels_a, labels_b):
    if len(labels_a) != len(labels_b):
        raise ValueError("Annotators must provide the same number of labels")
    if not labels_a:
        return 0.0

    total = len(labels_a)
    observed_agreement = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / total
    categories = sorted(set(labels_a) | set(labels_b))
    counts_a = {cat: labels_a.count(cat) / total for cat in categories}
    counts_b = {cat: labels_b.count(cat) / total for cat in categories}
    expected_agreement = sum(counts_a[cat] * counts_b[cat] for cat in categories)

    if 1.0 - expected_agreement == 0:
        return 1.0 if observed_agreement == 1.0 else 0.0
    return (observed_agreement - expected_agreement) / (1.0 - expected_agreement)


def compute_precision_recall_f1(true_labels, predicted_labels):
    true_set = set(true_labels)
    predicted_set = set(predicted_labels)
    tp = len(true_set & predicted_set)
    fp = len(predicted_set - true_set)
    fn = len(true_set - predicted_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
