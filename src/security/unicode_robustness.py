from src.security.unicode_validation import analyze_unicode_text


def compare_retrieval_consistency(retriever, clean_query, perturbed_query, top_k=5, relevant_chunk_ids=None):
    clean_unicode = analyze_unicode_text(clean_query)
    perturbed_unicode = analyze_unicode_text(perturbed_query)
    clean_results = retriever.retrieve(clean_query, top_k=top_k, mode="hybrid")
    perturbed_results = retriever.retrieve(perturbed_query, top_k=top_k, mode="hybrid")
    clean_ids = [row["chunk_id"] for row in clean_results]
    perturbed_ids = [row["chunk_id"] for row in perturbed_results]
    clean_set = set(clean_ids)
    perturbed_set = set(perturbed_ids)
    union = clean_set | perturbed_set
    relevant = set(relevant_chunk_ids or [])
    clean_recall = sum(item in relevant for item in clean_ids) / len(relevant) if relevant else None
    perturbed_recall = sum(item in relevant for item in perturbed_ids) / len(relevant) if relevant else None
    return {
        "clean_query": clean_query,
        "perturbed_query": perturbed_query,
        "clean_flags": {key: value for key, value in clean_unicode.items() if key not in {"source_text", "normalized_text", "invisible_positions", "bidirectional_positions", "control_positions"}},
        "perturbed_flags": {key: value for key, value in perturbed_unicode.items() if key not in {"source_text", "normalized_text", "invisible_positions", "bidirectional_positions", "control_positions"}},
        "clean_chunk_ids": clean_ids,
        "perturbed_chunk_ids": perturbed_ids,
        "top_k_jaccard": len(clean_set & perturbed_set) / len(union) if union else None,
        "clean_recall_at_k": clean_recall,
        "perturbed_recall_at_k": perturbed_recall,
        "recall_degradation": clean_recall - perturbed_recall if clean_recall is not None and perturbed_recall is not None else None,
        "interpretation": "Controlled query perturbation only; inspect normalization and source wording before treating variants as equivalent.",
    }
