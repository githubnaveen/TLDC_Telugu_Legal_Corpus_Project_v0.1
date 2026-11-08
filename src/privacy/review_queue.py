from pathlib import Path
import csv

def create_privacy_review_queue(register_csv: Path, output_csv: Path) -> int:
    # Administrative queue only: this does not detect, redact or certify PII.
    with Path(register_csv).open("r", encoding="utf-8-sig", newline="") as src:
        rows = list(csv.DictReader(src))
    fields = ["document_id", "state", "district", "document_type", "privacy_status", "review_action"]
    queue = []
    for row in rows:
        status = (row.get("privacy_status") or "").strip().upper()
        if status != "REVIEWED":
            queue.append({
                "document_id": row.get("document_id", ""),
                "state": row.get("state", ""),
                "district": row.get("district", ""),
                "document_type": row.get("document_type", ""),
                "privacy_status": status or "NOT_REVIEWED",
                "review_action": "Human privacy review required"
            })
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        writer.writerows(queue)
    return len(queue)
