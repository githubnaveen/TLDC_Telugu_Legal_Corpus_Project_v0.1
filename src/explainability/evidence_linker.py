from typing import Any, Optional

from src.preprocessing.text_cleaner import normalise_telugu_text


def link_attributions(
    document_id: str,
    model_text: str,
    sections: list[dict[str, Any]],
    attributions: list[dict[str, Any]],
    predicted_label: str,
    model_version: str,
    method: str,
    task: str = "DOCUMENT_CLASSIFICATION",
    score: Optional[float] = None,
    legal_references: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    normalized_sections = []
    for section in sections:
        section_text = normalise_telugu_text(str(section.get("text", "")))
        if section_text:
            normalized_sections.append((section, section_text))

    expected_text = "\n".join(text for _, text in normalized_sections)
    section_ranges = []
    cursor = 0
    for section, section_text in normalized_sections:
        section_ranges.append((cursor, cursor + len(section_text), section, section_text))
        cursor += len(section_text) + 1

    important_segments = []
    evidence = []
    warnings = []
    if expected_text != model_text:
        warnings.append("normalized_section_text_does_not_match_model_input")
    else:
        for attribution in attributions:
            start, end = attribution.get("start"), attribution.get("end")
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start or end > len(model_text):
                warnings.append("invalid_attribution_offsets")
                continue
            source_text = model_text[start:end]
            match = next((
                (section, section_text)
                for section_start, section_end, section, section_text in section_ranges
                if start >= section_start and end <= section_end
            ), None)
            if match is None:
                warnings.append("attribution_not_contained_in_one_source_section")
                continue

            section, _ = match
            linked = {
                "segment_id": section.get("section_id"),
                "page_number": section.get("page_number"),
                "section_type": section.get("section_type"),
                "text": source_text,
                "char_start": start,
                "char_end": end,
                "attribution_score": float(attribution.get("score", 0.0)),
            }
            important_segments.append(linked)
            evidence_record = {
                "source_document_id": document_id,
                "page_number": section.get("page_number"),
                "section_id": section.get("section_id"),
                "section_type": section.get("section_type"),
            }
            if evidence_record not in evidence:
                evidence.append(evidence_record)

    return {
        "schema_version": "1.0",
        "document_id": document_id,
        "prediction": {
            "task": task,
            "predicted_label": predicted_label,
            "model_version": model_version,
            "score": score,
            "score_is_calibrated_probability": False,
        },
        "explanation": {
            "method": method,
            "important_segments": important_segments,
            "warnings": warnings,
        },
        "evidence": evidence,
        "legal_references": legal_references or [],
        "human_review": {
            "status": "PENDING",
            "reviewer_id": None,
            "reviewer_comment": None,
            "reviewed_at": None,
        },
    }
