from src.annotation.segmenter import segment_document
from src.annotation.entity_extractor import extract_legal_entities
from src.evaluation.annotation_agreement import cohens_kappa


def test_segment_document_creates_expected_sections():
    text = (
        "హైకోర్ట్, హైదరాబాదు\n"
        "కేసు సంఖ్య: CRLP 1234/2024\n"
        "వాస్తవాలు: ఆరోపణారులు.....\n"
        "పెటిషనర్ వాదనలు: ఎలక్షన్ కాదు.\n"
        "స్పష్ట చట్టం: IPC సెక్షన్ 302.\n"
        "పరీక్షలు మరియు సాక్షులు: పత్రాలు.\n"
        "న్యాయమూర్తి పరిశీలన: లేబుల్.\n"
        "ఆర్డర్: బెయిల్ మంజూరు."
    )
    sections = segment_document(text)
    labels = {item["section_type"] for item in sections}
    assert {"HEADER", "FACTS", "PETITIONER_ARGUMENT", "LEGAL_PROVISIONS", "EVIDENCE", "JUDICIAL_OBSERVATION", "DECISION"}.issubset(labels)
    assert all("section_id" in item for item in sections)


def test_extract_legal_entities_returns_expected_types():
    text = "హైదరాబాద్ హైకోర్ట్ కేసు సంఖ్య CRLP 1234/2024, IPC సెక్షన్ 302, సోమవారం 2024-01-15 న బెయిల్ మంజూరు"
    entities = extract_legal_entities(text)
    labels = {item["label"] for item in entities}
    assert {"COURT", "CASE_NUMBER", "LEGAL_ACT", "SECTION", "DATE"}.issubset(labels)


def test_cohens_kappa_handles_agreement():
    annotator_a = ["A", "B", "A", "B"]
    annotator_b = ["A", "B", "A", "B"]
    assert cohens_kappa(annotator_a, annotator_b) == 1.0
