from src.security.retrieval_integrity import audit_retrieval_result


def compare_clean_and_candidate_rankings(clean_results, candidate_results, relevant_chunk_ids=None):
    relevant = set(relevant_chunk_ids or [])
    clean_ids = [str(row.get("chunk_id", "")) for row in clean_results]
    candidate_ids = [str(row.get("chunk_id", "")) for row in candidate_results]
    clean_recall = sum(item in relevant for item in clean_ids) / len(relevant) if relevant else None
    candidate_recall = sum(item in relevant for item in candidate_ids) / len(relevant) if relevant else None
    return {
        "clean_retrieval_audit": audit_retrieval_result(clean_results),
        "candidate_retrieval_audit": audit_retrieval_result(candidate_results),
        "clean_relevant_recall_at_k": clean_recall,
        "candidate_relevant_recall_at_k": candidate_recall,
        "recall_degradation": clean_recall - candidate_recall if clean_recall is not None and candidate_recall is not None else None,
        "top_result_changed": bool(clean_ids and candidate_ids and clean_ids[0] != candidate_ids[0]),
        "interpretation": "Controlled isolated comparison; candidate data is not automatically admitted to the corpus.",
    }
