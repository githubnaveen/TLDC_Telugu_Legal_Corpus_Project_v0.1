ALLOWED_LABELS = {
    "CASE_FACTS", "PETITIONER_ARGUMENTS", "RESPONDENT_ARGUMENTS",
    "LEGAL_PROVISIONS", "EVIDENCE", "JUDICIAL_REASONING", "FINAL_ORDER", "OUTCOME"
}

def validate_annotation(annotation: dict, page_text: str) -> list[str]:
    errors = []
    required = ["document_id", "page_number", "start_character", "end_character",
                "source_text", "annotation_label", "annotator_id"]
    for key in required:
        if key not in annotation or annotation[key] in (None, ""):
            errors.append(f"Missing required field: {key}")
    label = annotation.get("annotation_label")
    if label and label not in ALLOWED_LABELS:
        errors.append(f"Unknown annotation label: {label}")
    try:
        start, end = int(annotation["start_character"]), int(annotation["end_character"])
        if start < 0 or end <= start or end > len(page_text):
            errors.append("Invalid annotation character offsets")
        elif page_text[start:end] != annotation.get("source_text"):
            errors.append("source_text does not match the referenced page span")
    except (KeyError, TypeError, ValueError):
        errors.append("Character offsets must be valid integers")
    return errors
