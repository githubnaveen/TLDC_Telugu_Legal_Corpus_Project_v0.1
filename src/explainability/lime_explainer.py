def explain_text(model, text, class_names=None, model_version="unknown", num_features=20):
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError as error:
        raise ImportError("Install LIME with `pip install lime` to use this explainer") from error
    if not hasattr(model, "predict_proba"):
        raise ValueError("LIME adapter requires a classifier with predict_proba")

    explainer = LimeTextExplainer(class_names=class_names or list(model.classes_))
    probabilities = model.predict_proba([text])[0]
    target_index = int(probabilities.argmax())
    explanation = explainer.explain_instance(
        text,
        model.predict_proba,
        labels=[target_index],
        num_features=num_features,
    )
    feature_scores = explanation.as_list(label=target_index)
    attributions = []
    for feature, score in feature_scores:
        start = text.find(feature)
        if start < 0:
            continue
        attributions.append({
            "feature": feature,
            "start": start,
            "end": start + len(feature),
            "score": float(score),
        })
    return {
        "method": "LIME_TEXT",
        "predicted_label": str(model.classes_[target_index]),
        "score": float(probabilities[target_index]),
        "model_version": model_version,
        "attributions": attributions,
        "score_is_calibrated_probability": False,
    }
