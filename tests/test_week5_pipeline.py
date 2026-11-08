import json

import pytest

from src.training.dataset_builder import build_week5_examples, split_by_case
from src.training.legal_ner_baseline import entity_metrics, spans_to_bio
from src.training.classification_baseline import train_evaluate


def test_split_by_case_keeps_groups_together():
    examples = [
        {"document_id": f"DOC-{group}-{item}", "case_group_id": f"CASE-{group}", "label": "Bail"}
        for group in range(12)
        for item in range(2)
    ]
    splits = split_by_case(examples, random_state=17)
    case_partitions = {}
    for partition, rows in splits.items():
        for row in rows:
            case_partitions.setdefault(row["case_group_id"], set()).add(partition)
    assert all(len(partitions) == 1 for partitions in case_partitions.values())
    assert set(splits) == {"train", "validation", "test"}


def test_build_examples_excludes_pending_and_missing_provenance(tmp_path):
    segments_path = tmp_path / "segments.jsonl"
    segments_path.write_text(
        json.dumps({
            "document_id": "DOC-1",
            "state": "Telangana",
            "document_language": "Telugu",
            "document_type": "Bail Order",
            "annotation_status": "PENDING_REVIEW",
            "sections": [{"section_id": "S001", "section_type": "FACTS", "text": "facts", "page_number": 1}],
        }) + "\n",
        encoding="utf-8",
    )
    register_path = tmp_path / "register.csv"
    register_path.write_text(
        "document_id,case_group_id,privacy_status,annotation_status\n"
        "DOC-1,CASE-1,approved,approved\n",
        encoding="utf-8",
    )

    assert build_week5_examples(segments_path, register_path) == []


def test_split_by_case_requires_three_groups():
    with pytest.raises(ValueError, match="at least three case groups"):
        split_by_case([{"document_id": "DOC-1", "case_group_id": "CASE-1"}])


def test_split_by_case_supports_minimum_three_groups():
    examples = [
        {"document_id": f"DOC-{group}", "case_group_id": f"CASE-{group}"}
        for group in range(3)
    ]
    splits = split_by_case(examples)
    assert all(len(splits[name]) == 1 for name in ("train", "validation", "test"))


def test_ner_span_offsets_are_exact_and_metrics_are_perfect():
    text = "IPC Section 302"
    start = text.index("IPC")
    entities = [{"start": start, "end": start + 3, "label": "LEGAL_ACT"}]
    tokens, tags = spans_to_bio(text, entities)
    assert tags == ["B-LEGAL_ACT", "O", "O"]
    assert entity_metrics([[(0, 3, "LEGAL_ACT")]], [[(0, 3, "LEGAL_ACT")]])["f1"] == 1.0


def test_ner_rejects_spans_that_do_not_match_token_boundaries():
    with pytest.raises(ValueError, match="token boundaries"):
        spans_to_bio("IPC Section", [{"start": 1, "end": 3, "label": "LEGAL_ACT"}])


def test_classification_baseline_reports_state_and_language_metrics():
    train_rows = [
        {"text": "తెలుగు bail order", "label": "Bail", "state": "Telangana", "language": "Telugu"},
        {"text": "తెలుగు civil judgment", "label": "Civil", "state": "Andhra Pradesh", "language": "Mixed"},
        {"text": "english bail decision", "label": "Bail", "state": "Telangana", "language": "English"},
        {"text": "english civil order", "label": "Civil", "state": "Andhra Pradesh", "language": "English"},
    ]
    test_rows = [
        {"text": "bail order", "label": "Bail", "state": "Telangana", "language": "Telugu"},
        {"text": "civil judgment", "label": "Civil", "state": "Andhra Pradesh", "language": "Mixed"},
    ]
    result = train_evaluate(train_rows, test_rows)
    assert set(result["by_state"]) == {"Telangana", "Andhra Pradesh"}
    assert set(result["by_language"]) == {"Telugu", "Mixed"}
