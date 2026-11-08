def explain_text(model, text, background_texts, model_version, max_features=200):
    try:
        import numpy as np
        import shap
    except ImportError as error:
        raise ImportError("Install SHAP with `pip install shap` to use this explainer") from error
    if not hasattr(model, "predict_proba") or not hasattr(model, "named_steps") or "tfidf" not in model.named_steps:
        raise ValueError("SHAP adapter requires a predict_proba sklearn pipeline with a 'tfidf' step")
    if not background_texts:
        raise ValueError("A reviewed background dataset is required for SHAP")

    vectorizer = model.named_steps["tfidf"]
    estimator = model.steps[-1][1]
    background = vectorizer.transform(background_texts)
    input_vector = vectorizer.transform([text])
    explainer = shap.LinearExplainer(estimator, background)
    explanation = explainer(input_vector)
    probabilities = model.predict_proba([text])[0]
    target_index = int(np.argmax(probabilities))
    values = np.asarray(explanation.values)
    negate_binary_score = len(model.classes_) == 2 and target_index == 0
    if values.ndim == 1:
        target_values = -values if negate_binary_score else values
    elif values.ndim == 2:
        if values.shape[0] == 1:
            target_values = -values[0] if negate_binary_score else values[0]
        else:
            target_values = values[:, target_index]
    elif values.ndim == 3:
        target_values = values[0, :, target_index]
    else:
        raise ValueError("Unexpected SHAP value shape for the selected classifier")

    feature_names = vectorizer.get_feature_names_out()
    sparse_row = input_vector.getrow(0)

    attributions = []
    for feature_index, _ in zip(sparse_row.indices, sparse_row.data):
        feature = str(feature_names[feature_index])
        score = float(target_values[feature_index])
        starts = []
        cursor = 0
        while True:
            start = text.find(feature, cursor)
            if start < 0:
                break
            starts.append(start)
            cursor = start + 1
        if not starts:
            continue
        for start in starts:
            attributions.append({
                "start": start,
                "end": start + len(feature),
                "score": score / len(starts),
                "feature": feature,
            })
    attributions.sort(key=lambda item: abs(item["score"]), reverse=True)
    attributions = attributions[:max_features]

    return {
        "method": "SHAP_LINEAR_TFIDF",
        "predicted_label": str(model.classes_[target_index]),
        "score": float(probabilities[target_index]),
        "model_version": model_version,
        "attributions": attributions,
        "background_count": len(background_texts),
        "background": "reviewed training text transformed by the fitted TF-IDF vectorizer",
        "score_is_calibrated_probability": False,
    }
