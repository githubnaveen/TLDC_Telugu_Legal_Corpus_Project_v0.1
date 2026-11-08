import pandas as pd

from src.collection.source_inventory import select_approved_sources
from src.preprocessing.language_classifier import detect_language


def test_select_approved_sources_filters_approved_only(tmp_path):
    csv_path = tmp_path / "source_inventory.csv"
    df = pd.DataFrame(
        [
            {
                "source_id": "SRC-TS-001",
                "state": "Telangana",
                "court_or_portal": "Telangana High Court",
                "document_category": "Judgment",
                "language_availability": "Telugu",
                "access_url": "https://example.com/1",
                "terms_reviewed": "yes",
                "permission_status": "approved",
                "notes": "ok",
            },
            {
                "source_id": "SRC-TS-002",
                "state": "Telangana",
                "court_or_portal": "District Court",
                "document_category": "Order",
                "language_availability": "English",
                "access_url": "https://example.com/2",
                "terms_reviewed": "pending",
                "permission_status": "restricted",
                "notes": "not approved",
            },
        ]
    )
    df.to_csv(csv_path, index=False)

    approved = select_approved_sources(csv_path)

    assert list(approved["source_id"]) == ["SRC-TS-001"]


def test_detect_language_handles_telugu_english_and_mixed():
    assert detect_language("విశ్లేషణ మరియు తీర్పు") == "Telugu"
    assert detect_language("This is an English judgment.") == "English"
    assert detect_language("ఈ నిర్ణయం is a mixed judgment") == "Mixed"
