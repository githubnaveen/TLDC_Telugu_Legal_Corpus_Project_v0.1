import pytest

from src.evaluation.counterfactual_fairness import evaluate_counterfactual_pairs
from src.evaluation.fairness_audit import audit_group_distributions, evaluate_group_performance
from src.evaluation.uncertainty_calibration import brier_score, expected_calibration_error, fit_temperature, apply_temperature, evaluate_calibration
from src.evaluation.selective_prediction import risk_coverage_curve
from src.rag.abstention_policy import assess_evidence_sufficiency
from src.oversight.human_review import validate_review, build_review_summary, pairwise_review_agreement


def test_fairness_audit_reports_small_group_counts_without_outcome_parity_claims():
    rows = [
        {"state": "Telangana", "language": "Telugu", "label": "Bail"},
        {"state": "Telangana", "language": "Telugu", "label": "Civil"},
        {"state": "Andhra Pradesh", "language": "Mixed", "label": "Bail"},
    ]
    audited = audit_group_distributions(rows, ["state", "language"], min_group_size=2)
    andhra = next(row for row in audited if row["dimension"] == "state" and row["group"] == "Andhra Pradesh")
    assert andhra["count"] == 1
    assert andhra["sample_adequacy"] == "INSUFFICIENT"
    assert andhra["class_distribution"] == {"Bail": 1}


def test_group_performance_reports_metrics_separately():
    rows = [
        {"state": "Telangana", "label": "A"},
        {"state": "Telangana", "label": "B"},
        {"state": "Andhra Pradesh", "label": "A"},
        {"state": "Andhra Pradesh", "label": "B"},
    ]
    results = evaluate_group_performance(rows, ["A", "B", "A", "A"], ["state"])
    telangana = next(row for row in results if row["group"] == "Telangana")
    andhra = next(row for row in results if row["group"] == "Andhra Pradesh")
    assert telangana["macro_f1"] == 1.0
    assert andhra["macro_f1"] < telangana["macro_f1"]


def test_counterfactual_evaluator_uses_only_adjudicated_valid_pairs():
    pairs = [{
        "pair_id": "P1", "review_status": "ADJUDICATED",
        "changed_attribute_irrelevant": True, "legally_relevant_facts_preserved": True,
        "original": {"conclusion": "ABSTAINED", "citations": ["C1"], "claims": ["x"]},
        "counterfactual": {"conclusion": "ABSTAINED", "citations": ["C1"], "claims": ["x"]},
    }, {
        "pair_id": "P2", "review_status": "PENDING",
        "changed_attribute_irrelevant": True, "legally_relevant_facts_preserved": True,
        "original": {"conclusion": "GRANTED"}, "counterfactual": {"conclusion": "REJECTED"},
    }]
    result = evaluate_counterfactual_pairs(pairs)
    assert result["valid_pair_count"] == 1
    assert result["counterfactual_consistency"] == 1.0
    assert result["excluded_pair_count"] == 1


def test_calibration_metrics_and_temperature_scaling():
    labels = [0, 1, 1, 0]
    probabilities = [[0.8, 0.2], [0.2, 0.8], [0.4, 0.6], [0.4, 0.6]]
    assert brier_score(labels, probabilities) == pytest.approx(0.15)
    ece, reliability_bins = expected_calibration_error(labels, probabilities, bins=2)
    assert ece > 0
    assert len(reliability_bins) == 2
    temperature = fit_temperature(labels, probabilities)
    assert temperature > 0
    calibrated = apply_temperature(probabilities, temperature)
    assert all(abs(sum(row) - 1.0) < 1e-8 for row in calibrated)
    raw_only = evaluate_calibration(labels, probabilities, bins=2)
    assert raw_only["temperature"] is None
    assert "calibrated_brier_score" not in raw_only
    with_temperature = evaluate_calibration(labels, probabilities, bins=2, temperature=temperature)
    assert with_temperature["calibrated_brier_score"] is not None
    named_result = evaluate_calibration(
        ["Bail", "Civil"], [[0.8, 0.2], [0.2, 0.8]], bins=2,
        class_labels=["Bail", "Civil"],
    )
    assert named_result["sample_count"] == 2
    with pytest.raises(ValueError, match="class_labels"):
        evaluate_calibration(["Bail"], [[0.8, 0.2]])


def test_risk_coverage_and_abstention_policy():
    curve = risk_coverage_curve([True, False, True], [0.9, 0.8, 0.2], thresholds=[0.0, 0.5, 0.85])
    assert curve[0]["coverage"] == 1.0
    assert curve[-1]["coverage"] < curve[0]["coverage"]
    assert assess_evidence_sufficiency([], {"citation_precision": 0.0})["decision"] == "ABSTAIN"
    adequate = [{"document_id": "D1", "page_number": 1, "section_type": "FACTS", "chunk_text": "source"}]
    decision = assess_evidence_sufficiency(adequate, {"citation_precision": 1.0, "total_citations": 1})
    assert decision["decision"] == "PRESENT_DRAFT_FOR_HUMAN_REVIEW"
    assert decision["human_review_required"] is True


def test_human_review_requires_valid_label_and_summary_tracks_reviews():
    review = validate_review({
        "answer_id": "A1", "reviewer_id": "R1", "review_label": "ACCEPTED_WITH_CORRECTION",
        "reviewer_comment": "Fixed section citation.",
    })
    assert review["review_label"] == "ACCEPTED_WITH_CORRECTION"
    summary = build_review_summary([review])
    assert summary["review_count"] == 1
    agreement = pairwise_review_agreement([
        {"answer_id": "A1", "reviewer_id": "R1", "review_label": "ACCEPTED"},
        {"answer_id": "A1", "reviewer_id": "R2", "review_label": "ACCEPTED"},
    ])
    assert agreement["paired_answer_count"] == 1
    assert agreement["agreement"] == 1.0
    with pytest.raises(ValueError):
        validate_review({"answer_id": "A2", "reviewer_id": "R2", "review_label": "MAYBE"})
