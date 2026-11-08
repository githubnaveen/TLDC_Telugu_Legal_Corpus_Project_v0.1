def calculate_evidence_coverage(explanation_record):
    segments = explanation_record.get("explanation", {}).get("important_segments", [])
    if not segments:
        return 0.0
    linked_count = sum(
        1 for segment in segments
        if segment.get("page_number") is not None
        and segment.get("segment_id")
        and segment.get("text")
    )
    return linked_count / len(segments)
