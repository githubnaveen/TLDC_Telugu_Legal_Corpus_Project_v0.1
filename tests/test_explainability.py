import pytest

from src.evaluation.evidence_coverage import calculate_evidence_coverage
from src.evaluation.explanation_faithfulness import evaluate_removal_faithfulness
from src.evaluation.explanation_stability import top_feature_jaccard
from src.explainability.evidence_linker import link_attributions
from src.explainability.prototype import predict_and_explain
from src.explainability.review import record_human_review
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def test_link_attribution_to_page_and_section_provenance():
    sections = [
        {"section_id": "S001", "section_type": "FACTS", "text": "case facts", "page_number": 2},
        {"section_id": "S002", "section_type": "DECISION", "text": "bail granted", "page_number": 4},
    ]
    explanation = link_attributions(
        document_id="DOC-1",
        model_text="case facts\nbail granted",
        sections=sections,
        attributions=[{"start": 11, "end": 23, "score": 0.8}],
        predicted_label="Bail Order",
        model_version="test-v1",
        method="TEST",
    )
    segment = explanation["explanation"]["important_segments"][0]
    assert segment["page_number"] == 4
    assert segment["section_type"] == "DECISION"
    assert segment["text"] == "bail granted"
    assert calculate_evidence_coverage(explanation) == 1.0


def test_unmatched_attribution_is_not_linked_as_evidence():
    explanation = link_attributions(
        document_id="DOC-2",
        model_text="original text",
        sections=[{"section_id": "S1", "section_type": "FACTS", "text": "different text", "page_number": 1}],
        attributions=[{"start": 0, "end": 8, "score": 0.7}],
        predicted_label="Other",
        model_version="v1",
        method="TEST",
    )
    assert explanation["explanation"]["important_segments"] == []
    assert explanation["evidence"] == []
    assert calculate_evidence_coverage(explanation) == 0.0


def test_removal_faithfulness_reports_high_vs_low_attribution():
    def predictor(texts):
        return [[0.9 if "grant" in text else 0.2, 0.1 if "grant" in text else 0.8] for text in texts]

    result = evaluate_removal_faithfulness(
        predictor,
        "grant the application",
        target_index=0,
        attributions=[{"start": 0, "end": 5, "score": 1.0}, {"start": 6, "end": 9, "score": 0.1}],
        top_k=1,
    )
    assert result["comprehensiveness"] > result["low_attribution_drop"]


def test_stability_jaccard_and_review_audit_fields():
    assert top_feature_jaccard([("court", 0.8), ("bail", 0.5)], [("court", 0.7), ("order", 0.4)], k=1) == 1.0
    record = {"document_id": "DOC-3", "human_review": {"status": "PENDING"}}
    reviewed = record_human_review(record, "ACCEPT", "reviewer-1", "Source passage verified")
    assert reviewed["human_review"]["status"] == "ACCEPTED"
    assert reviewed["human_review"]["reviewer_id"] == "reviewer-1"
    with pytest.raises(ValueError):
        record_human_review(record, "MAYBE", "reviewer-1", "unclear")


def test_linear_prototype_generates_source_linked_attributions():
    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4))),
        ("classifier", LogisticRegression(random_state=7)),
    ])
    model.fit(["bail order granted", "bail application", "civil dispute", "civil order"], ["Bail", "Bail", "Civil", "Civil"])
    text = "bail order granted"
    record = predict_and_explain(
        model,
        text,
        "DOC-4",
        [{"section_id": "S1", "section_type": "DECISION", "text": text, "page_number": 3}],
        "test-model-v1",
    )
    assert record["prediction"]["predicted_label"] == "Bail"
    assert record["explanation"]["important_segments"]
    assert all(segment["page_number"] == 3 for segment in record["explanation"]["important_segments"])
