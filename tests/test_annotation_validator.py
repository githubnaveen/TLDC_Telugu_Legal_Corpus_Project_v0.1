from src.annotation.validator import validate_annotation

def test_valid_annotation():
    page = "ఈ కేసులో బెయిల్ మంజూరు చేయబడింది."
    word = "బెయిల్"
    start = page.index(word)
    annotation = {
        "document_id": "TLDC-TS-HYD-2025-000001", "page_number": 1,
        "start_character": start, "end_character": start + len(word),
        "source_text": word, "annotation_label": "CASE_FACTS", "annotator_id": "A1"
    }
    assert validate_annotation(annotation, page) == []
