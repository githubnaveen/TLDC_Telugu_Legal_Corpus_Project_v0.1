import pandas as pd

from src.collection.source_inventory import validate_source_inventory


def test_validate_source_inventory_accepts_valid_data(tmp_path):
    csv_path = tmp_path / "source_inventory.csv"
    df = pd.DataFrame(
        [
            {
                "source_id": "SRC-TS-001",
                "state": "Telangana",
                "court_or_portal": "Telangana High Court",
                "document_category": "Bail order",
                "language_availability": "Telugu",
                "access_url": "https://example.com/ts-001",
                "terms_reviewed": "pending",
                "permission_status": "pending_review",
                "notes": "Sample record",
            },
            {
                "source_id": "SRC-AP-001",
                "state": "Andhra Pradesh",
                "court_or_portal": "Andhra Pradesh High Court",
                "document_category": "Judgment",
                "language_availability": "English",
                "access_url": "https://example.com/ap-001",
                "terms_reviewed": "pending",
                "permission_status": "pending_review",
                "notes": "Sample record",
            },
        ]
    )
    df.to_csv(csv_path, index=False)

    result = validate_source_inventory(csv_path)

    assert len(result) == 2
    assert result["state"].tolist() == ["Telangana", "Andhra Pradesh"]


def test_validate_source_inventory_rejects_invalid_state(tmp_path):
    csv_path = tmp_path / "source_inventory.csv"
    pd.DataFrame(
        [
            {
                "source_id": "SRC-XX-001",
                "state": "Tamil Nadu",
                "court_or_portal": "Example",
                "document_category": "Judgment",
                "language_availability": "English",
                "access_url": "https://example.com",
                "terms_reviewed": "pending",
                "permission_status": "pending_review",
                "notes": "bad state",
            }
        ]
    ).to_csv(csv_path, index=False)

    try:
        validate_source_inventory(csv_path)
        assert False, "Expected ValueError for unsupported state"
    except ValueError:
        pass
