import json
import random
from pathlib import Path
from typing import Any, Optional, Union

from src.preprocessing.text_cleaner import normalise_telugu_text

APPROVED_STATUSES = {"APPROVED", "ADJUDICATED"}
PRIVACY_CLEARED_STATUSES = {"APPROVED", "ANONYMISED", "ANONYMIZED", "REVIEWED"}


def _read_jsonl(path: Optional[Union[str, Path]]) -> list[dict[str, Any]]:
    if path is None or not Path(path).exists():
        return []
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _status(value: Any) -> str:
    return str(value or "").strip().upper()


def _read_register(path: Union[str, Path]) -> dict[str, dict[str, str]]:
    import csv

    register = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            doc_id = (row.get("document_id") or "").strip()
            if doc_id:
                register[doc_id] = row
    return register


def _metadata_is_eligible(record: dict[str, Any], register_row: Optional[dict[str, str]]) -> bool:
    if _status(record.get("annotation_status")) not in APPROVED_STATUSES:
        return False
    if register_row is None:
        return False
    if _status(register_row.get("annotation_status")) not in APPROVED_STATUSES:
        return False
    if _status(register_row.get("privacy_status")) not in PRIVACY_CLEARED_STATUSES:
        return False
    return bool((register_row.get("case_group_id") or "").strip())


def build_week5_examples(segmentation_path: Union[str, Path], register_path: Union[str, Path], entities_path: Optional[Union[str, Path]] = None) -> list[dict[str, Any]]:
    register = _read_register(register_path)
    segmentations = _read_jsonl(segmentation_path)
    entity_rows = _read_jsonl(entities_path)
    entities_by_document: dict[str, list[dict[str, Any]]] = {}
    for entity in entity_rows:
        entities_by_document.setdefault(str(entity.get("document_id", "")), []).append(entity)

    examples = []
    for document in segmentations:
        document_id = str(document.get("document_id", ""))
        metadata = register.get(document_id)
        if not document_id or not _metadata_is_eligible(document, metadata):
            continue

        group_id = (metadata.get("case_group_id") or "").strip()
        state = (metadata.get("state") or document.get("state") or "").strip()
        district = (metadata.get("district") or document.get("district") or "").strip()
        language = (metadata.get("document_language") or document.get("document_language") or "").strip()
        sections = document.get("sections") or []
        document_text = normalise_telugu_text("\n".join(
            str(section.get("text", "")) for section in sections if section.get("text")
        ))
        if document_text and document.get("document_type"):
            examples.append({
                "task": "document_classification",
                "document_id": document_id,
                "case_group_id": group_id,
                "state": state,
                "district": district,
                "language": language,
                "text": document_text,
                "label": str(document["document_type"]),
            })

        for section in sections:
            section_text = normalise_telugu_text(str(section.get("text", "")))
            section_type = str(section.get("section_type", "")).strip()
            if section_text and section_type:
                examples.append({
                    "task": "section_classification",
                    "document_id": document_id,
                    "case_group_id": group_id,
                    "state": state,
                    "district": district,
                    "language": language,
                    "section_id": section.get("section_id"),
                    "page_number": section.get("page_number"),
                    "text": section_text,
                    "label": section_type,
                })

        valid_entities = []
        for entity in entities_by_document.get(document_id, []):
            entity_text = str(entity.get("text", entity.get("entity", "")))
            start, end = entity.get("start"), entity.get("end")
            if not isinstance(start, int) or not isinstance(end, int) or end <= start:
                continue
            if document_text[start:end] != entity_text or not entity.get("label"):
                continue
            valid_entities.append({"start": start, "end": end, "label": str(entity["label"])})
        if document_text and valid_entities:
            examples.append({
                "task": "legal_ner",
                "document_id": document_id,
                "case_group_id": group_id,
                "state": state,
                "district": district,
                "language": language,
                "text": document_text,
                "entities": valid_entities,
            })

    return examples


def split_by_case(
    examples: list[dict[str, Any]],
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    if abs(train_size + validation_size + test_size - 1.0) > 1e-8:
        raise ValueError("Split proportions must sum to 1.0")
    if not examples:
        return {"train": [], "validation": [], "test": []}
    groups = [str(row.get("case_group_id", "")).strip() for row in examples]
    if any(not group for group in groups):
        raise ValueError("Every example must have a case_group_id before splitting")
    unique_groups = set(groups)
    if len(unique_groups) < 3:
        raise ValueError("at least three case groups are required for train/validation/test splits")

    split_names = ["train", "validation", "test"]
    proportions = [train_size, validation_size, test_size]
    group_count = len(unique_groups)
    exact_counts = [group_count * proportion for proportion in proportions]
    group_counts = [int(count) for count in exact_counts]
    while sum(group_counts) < group_count:
        candidate = max(range(3), key=lambda index: exact_counts[index] - group_counts[index])
        group_counts[candidate] += 1
    while any(count == 0 for count in group_counts):
        empty_split = group_counts.index(0)
        donor = max(range(3), key=lambda index: group_counts[index])
        group_counts[donor] -= 1
        group_counts[empty_split] += 1

    group_order = list(unique_groups)
    random.Random(random_state).shuffle(group_order)
    boundaries = [group_counts[0], group_counts[0] + group_counts[1]]
    groups_by_split = {
        split_names[0]: set(group_order[:boundaries[0]]),
        split_names[1]: set(group_order[boundaries[0]:boundaries[1]]),
        split_names[2]: set(group_order[boundaries[1]:]),
    }
    return {
        split_name: [example for example in examples if example["case_group_id"] in split_groups]
        for split_name, split_groups in groups_by_split.items()
    }


def write_jsonl_splits(splits: dict[str, list[dict[str, Any]]], output_root: Union[str, Path]) -> None:
    root = Path(output_root)
    for split_name, examples in splits.items():
        path = root / split_name / f"tldc_{split_name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as output:
            for example in examples:
                output.write(json.dumps(example, ensure_ascii=False) + "\n")
