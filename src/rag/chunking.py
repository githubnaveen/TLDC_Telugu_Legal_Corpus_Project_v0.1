import argparse
import hashlib
import json
import re
from pathlib import Path

from src.preprocessing.text_cleaner import normalise_telugu_text
from src.security.prompt_injection_detector import scan_untrusted_content
from src.security.unicode_validation import analyze_unicode_text
from src.training.dataset_builder import _metadata_is_eligible, _read_jsonl, _read_register


def build_chunks(segmentation_path, register_path, max_words=180, overlap_words=30):
    if max_words < 1:
        raise ValueError("max_words must be positive")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must be non-negative and smaller than max_words")

    register = _read_register(register_path)
    segmentations = _read_jsonl(segmentation_path)
    chunks = []
    stats = {"documents_added": 0, "documents_skipped": 0, "chunks_added": 0}

    for document in segmentations:
        document_id = str(document.get("document_id", "")).strip()
        metadata = register.get(document_id)
        if not document_id or not _metadata_is_eligible(document, metadata):
            stats["documents_skipped"] += 1
            continue

        raw_sections = document.get("sections") or []
        source_text = normalise_telugu_text("\n".join(
            str(section.get("text", "")) for section in raw_sections if section.get("text")
        ))
        checksum = str(metadata.get("file_checksum", "")).strip().lower()
        source_hash = checksum if re.fullmatch(r"[0-9a-f]{64}", checksum) else hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        created_for_document = 0
        for section in raw_sections:
            text = normalise_telugu_text(str(section.get("text", "")))
            if not text:
                continue
            try:
                page_number = int(section.get("page_number"))
            except (TypeError, ValueError):
                continue
            if page_number < 1:
                continue
            words = list(re.finditer(r"\S+", text, flags=re.UNICODE))
            if not words:
                continue
            section_id = str(section.get("section_id", "")).strip()
            section_type = str(section.get("section_type", "")).strip()
            if not section_id or not section_type:
                continue
            prompt_scan = scan_untrusted_content(text)
            unicode_scan = analyze_unicode_text(text)
            security_categories = list(prompt_scan["categories"])
            if unicode_scan["requires_review"]:
                security_categories.append("SUSPICIOUS_UNICODE_CONTROLS")
            security_categories = sorted(set(security_categories))

            step = max_words - overlap_words
            for start_word in range(0, len(words), step):
                end_word = min(start_word + max_words, len(words))
                start_char = words[start_word].start()
                end_char = words[end_word - 1].end()
                chunk_text = text[start_char:end_char]
                identity = "\0".join((document_id, section_id, str(page_number), str(start_char), chunk_text))
                chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
                chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "case_group_id": str(metadata.get("case_group_id", "")).strip(),
                    "state": str(metadata.get("state") or document.get("state") or "").strip(),
                    "district": str(metadata.get("district") or document.get("district") or "").strip(),
                    "court_name": str(metadata.get("court_name", "")).strip(),
                    "document_type": str(metadata.get("document_type") or document.get("document_type") or "").strip(),
                    "language": str(metadata.get("document_language") or document.get("document_language") or "").strip(),
                    "section_type": section_type,
                    "section_id": section_id,
                    "page_number": page_number,
                    "char_start": start_char,
                    "char_end": end_char,
                    "chunk_text": chunk_text,
                    "source_hash": source_hash,
                    "embedding_model": "PENDING",
                    "embedding_status": "NOT_COMPUTED",
                    "decision_outcome": str((document.get("decision") or {}).get("outcome", "")).strip().upper() if section_type.upper() == "DECISION" else "",
                    "security_review_required": bool(security_categories),
                    "security_flags": security_categories,
                })
                created_for_document += 1
                if end_word == len(words):
                    break
        if created_for_document:
            stats["documents_added"] += 1
            stats["chunks_added"] += created_for_document
        else:
            stats["documents_skipped"] += 1
    stats["security_review_chunks"] = sum(chunk["security_review_required"] for chunk in chunks)
    stats["prompt_injection_flagged_chunks"] = sum("DIRECT_INSTRUCTION_OVERRIDE" in chunk["security_flags"] or "CITATION_OR_EVIDENCE_SUPPRESSION" in chunk["security_flags"] for chunk in chunks)
    stats["unicode_flagged_chunks"] = sum("SUSPICIOUS_UNICODE_CONTROLS" in chunk["security_flags"] for chunk in chunks)
    return chunks, stats


def write_jsonl(path, records):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare provenance-preserving chunks from approved TLDC annotations.")
    parser.add_argument("--segments", default="annotations/document_segmentation/segmented_documents.jsonl")
    parser.add_argument("--register", default="metadata/document_register.csv")
    parser.add_argument("--output", default="data/embedded_chunks/tldc_embedded_chunks.jsonl")
    parser.add_argument("--max-words", type=int, default=180)
    parser.add_argument("--overlap-words", type=int, default=30)
    args = parser.parse_args()
    chunks, stats = build_chunks(args.segments, args.register, args.max_words, args.overlap_words)
    write_jsonl(args.output, chunks)
    print(json.dumps({**stats, "embedding_status": "NOT_COMPUTED; configure a local encoder to generate vectors"}, indent=2))


if __name__ == "__main__":
    main()
