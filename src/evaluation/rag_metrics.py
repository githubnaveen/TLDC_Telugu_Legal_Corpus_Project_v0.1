import math


def precision_at_k(ranked_ids, relevant_ids, k):
    if k < 1:
        raise ValueError("k must be at least one")
    relevant = set(relevant_ids)
    return sum(1 for chunk_id in ranked_ids[:k] if chunk_id in relevant) / k


def recall_at_k(ranked_ids, relevant_ids, k):
    if k < 1:
        raise ValueError("k must be at least one")
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return sum(1 for chunk_id in ranked_ids[:k] if chunk_id in relevant) / len(relevant)


def reciprocal_rank(ranked_ids, relevant_ids):
    relevant = set(relevant_ids)
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids, relevant_ids, k):
    if k < 1:
        raise ValueError("k must be at least one")
    relevant = set(relevant_ids)
    dcg = sum(1.0 / math.log2(rank + 1) for rank, chunk_id in enumerate(ranked_ids[:k], start=1) if chunk_id in relevant)
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def citation_precision(correct_citations, total_citations):
    return correct_citations / total_citations if total_citations else 0.0


def claim_support_rate(supported_claims, total_claims):
    return supported_claims / total_claims if total_claims else 0.0


def unsupported_claim_rate(unsupported_claims, total_claims):
    return unsupported_claims / total_claims if total_claims else 0.0


def abstention_metrics(predicted_abstain, gold_answerable):
    if len(predicted_abstain) != len(gold_answerable):
        raise ValueError("Abstention predictions and gold labels must have equal lengths")
    abstain_tp = sum(bool(predicted) and not bool(answerable) for predicted, answerable in zip(predicted_abstain, gold_answerable))
    abstain_fp = sum(bool(predicted) and bool(answerable) for predicted, answerable in zip(predicted_abstain, gold_answerable))
    abstain_fn = sum(not bool(predicted) and not bool(answerable) for predicted, answerable in zip(predicted_abstain, gold_answerable))
    precision = abstain_tp / (abstain_tp + abstain_fp) if abstain_tp + abstain_fp else 0.0
    recall = abstain_tp / (abstain_tp + abstain_fn) if abstain_tp + abstain_fn else 0.0
    return {"abstention_precision": precision, "abstention_recall": recall}
