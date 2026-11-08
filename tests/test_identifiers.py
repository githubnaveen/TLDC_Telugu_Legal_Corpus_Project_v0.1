from src.collection.identifiers import make_document_id

def test_telangana_identifier():
    assert make_document_id("Telangana", "HYD", 2025, 1) == "TLDC-TS-HYD-2025-000001"

def test_andhra_identifier():
    assert make_document_id("Andhra Pradesh", "GNT", 2025, 12) == "TLDC-AP-GNT-2025-000012"
