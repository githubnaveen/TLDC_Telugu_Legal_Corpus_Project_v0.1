import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path


def calculate_sha256(file_path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as source:
        for block in iter(lambda: source.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file_against_checksum(file_path, expected_checksum):
    path = Path(file_path)
    expected = str(expected_checksum or "").strip().lower()
    if not path.is_file():
        return {"status": "MISSING_FILE", "expected_checksum": expected, "actual_checksum": None}
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        return {"status": "UNVERIFIABLE_CHECKSUM", "expected_checksum": expected or None, "actual_checksum": calculate_sha256(path)}
    actual = calculate_sha256(path)
    return {"status": "MATCH" if actual == expected else "MISMATCH", "expected_checksum": expected, "actual_checksum": actual}


def verify_registered_files(register_path, raw_root):
    rows = []
    root = Path(raw_root).resolve()
    state_directories = {"Telangana": "Telangana", "Andhra Pradesh": "Andhra_Pradesh"}
    with Path(register_path).open(encoding="utf-8-sig", newline="") as source:
        register = csv.DictReader(source)
        for row in register:
            document_id = str(row.get("document_id", "")).strip()
            state = str(row.get("state", "")).strip()
            filename = str(row.get("original_filename", "")).strip()
            if not document_id or not state or not filename:
                rows.append({"document_id": document_id, "status": "INCOMPLETE_REGISTER_ROW"})
                continue
            state_directory = state_directories.get(state)
            if state_directory is None:
                rows.append({"document_id": document_id, "status": "UNKNOWN_STATE_DIRECTORY"})
                continue
            candidate = (root / state_directory / filename).resolve()
            if root not in candidate.parents:
                rows.append({"document_id": document_id, "status": "INVALID_PATH"})
                continue
            result = verify_file_against_checksum(candidate, row.get("file_checksum"))
            rows.append({"document_id": document_id, "source_url": row.get("source_url", ""), **result,
                         "interpretation": "Digest match checks consistency with the stored checksum; it does not prove source authenticity."})
    return rows


def evaluate_poisoning_records(trusted_records, candidate_records):
    trusted = {str(row.get("document_id", "")): row for row in trusted_records if row.get("document_id")}
    candidates = [row for row in candidate_records if row.get("document_id")]
    by_hash = defaultdict(list)
    details = []
    label_changes = 0
    hash_mismatches = 0
    unregistered = 0
    for row in candidates:
        document_id = str(row["document_id"])
        baseline = trusted.get(document_id)
        source_hash = str(row.get("source_hash", row.get("file_checksum", ""))).lower()
        if source_hash:
            by_hash[source_hash].append(document_id)
        changed_label = bool(baseline and baseline.get("label") and row.get("label") and str(baseline["label"]) != str(row["label"]))
        changed_hash = bool(baseline and baseline.get("source_hash", baseline.get("file_checksum")) and source_hash and str(baseline.get("source_hash", baseline.get("file_checksum"))).lower() != source_hash)
        label_changes += int(changed_label)
        hash_mismatches += int(changed_hash)
        unregistered += int(baseline is None)
        details.append({
            "document_id": document_id,
            "registered": baseline is not None,
            "label_changed": changed_label,
            "hash_mismatch": changed_hash,
            "status": "FLAGGED" if baseline is None or changed_label or changed_hash else "CONSISTENT_WITH_MANIFEST",
        })
    duplicate_groups = [sorted(document_ids) for document_ids in by_hash.values() if len(set(document_ids)) > 1]
    return {
        "candidate_count": len(candidates),
        "changed_label_count": label_changes,
        "hash_mismatch_count": hash_mismatches,
        "unregistered_document_count": unregistered,
        "duplicate_hash_group_count": len(duplicate_groups),
        "duplicate_hash_groups": duplicate_groups,
        "details": details,
        "interpretation": "Controlled integrity comparison only. Checksums do not prove source authenticity; review all flagged records.",
    }


def main():
    parser = argparse.ArgumentParser(description="Read-only verification of registered corpus file checksums.")
    parser.add_argument("--register", default="metadata/document_register.csv")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--output", default="results/corpus_integrity_evaluation.csv")
    args = parser.parse_args()
    records = verify_registered_files(args.register, args.raw_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["document_id", "source_url", "status", "expected_checksum", "actual_checksum", "interpretation"]
    with output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    print("Registered file rows checked:", len(records))


if __name__ == "__main__":
    main()
