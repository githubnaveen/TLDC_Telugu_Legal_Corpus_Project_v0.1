import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

FIELDNAMES = ["event_id", "timestamp", "event_type", "record_id", "flag_categories", "action"]


def append_security_event(event_type, record_id, flag_categories, action="ABSTAINED_REQUIRES_REVIEW", output_path="audit/security_events.csv"):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "event_id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": str(event_type),
        "record_id": str(record_id),
        "flag_categories": json.dumps(sorted(set(flag_categories)), ensure_ascii=False),
        "action": str(action),
    }
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return {key: value for key, value in row.items() if key != "timestamp"}
