import hashlib
import unicodedata
from collections import defaultdict


def _normalize(text):
    return " ".join(unicodedata.normalize("NFC", str(text or "")).casefold().split())


def find_exact_duplicate_chunks(chunks):
    groups = defaultdict(list)
    for chunk in chunks:
        text = _normalize(chunk.get("chunk_text", ""))
        if not text:
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        groups[digest].append(str(chunk.get("chunk_id", "")))
    return [sorted(ids) for ids in groups.values() if len(set(ids)) > 1]


def audit_retrieval_result(results, maximum_document_share=0.8):
    if not 0.0 <= maximum_document_share <= 1.0:
        raise ValueError("maximum_document_share must be between zero and one")
    documents = [str(row.get("document_id", "")) for row in results if row.get("document_id")]
    unique_documents = set(documents)
    max_share = max((documents.count(document) / len(documents) for document in unique_documents), default=0.0)
    duplicate_groups = find_exact_duplicate_chunks(results)
    return {
        "result_count": len(results),
        "distinct_document_count": len(unique_documents),
        "maximum_document_share": max_share,
        "single_document_dominance": bool(results) and max_share > maximum_document_share,
        "duplicate_chunk_groups": duplicate_groups,
        "duplicate_chunk_count": sum(len(group) for group in duplicate_groups),
        "flags_are_review_signals_only": True,
    }
