def top_feature_jaccard(first_explanation, second_explanation, k=10):
    if k <= 0:
        raise ValueError("k must be positive")
    first_features = {feature for feature, _ in sorted(first_explanation, key=lambda item: abs(item[1]), reverse=True)[:k]}
    second_features = {feature for feature, _ in sorted(second_explanation, key=lambda item: abs(item[1]), reverse=True)[:k]}
    union = first_features | second_features
    if not union:
        return 1.0
    return len(first_features & second_features) / len(union)
