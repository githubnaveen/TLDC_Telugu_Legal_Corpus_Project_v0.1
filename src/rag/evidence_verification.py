def verify_citations(answer_record, evidence_rows):
    by_id = {str(row.get("chunk_id", "")): row for row in evidence_rows if row.get("chunk_id")}
    correct = 0
    total = 0
    invalid = []
    claims = answer_record.get("claims", [])
    if not isinstance(claims, list):
        return {
            "correct_citations": 0,
            "total_citations": 1,
            "citation_precision": 0.0,
            "invalid_citations": [{"claim_index": None, "citation": None, "reason": "claims_not_list"}],
            "verification_scope": "citation identity and exact quoted-span check only; not semantic entailment",
        }
    for claim_index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            total += 1
            invalid.append({"claim_index": claim_index, "citation": None, "reason": "claim_not_object"})
            continue
        citations = claim.get("citations") or []
        if not isinstance(citations, list):
            total += 1
            invalid.append({"claim_index": claim_index, "citation": None, "reason": "citations_not_list"})
            continue
        for citation in citations:
            total += 1
            if not isinstance(citation, dict):
                invalid.append({"claim_index": claim_index, "citation": None, "reason": "citation_not_object"})
                continue
            source = by_id.get(str(citation.get("chunk_id", "")))
            evidence_text = str(citation.get("evidence_text", ""))
            valid = source is not None
            if valid:
                valid = (
                        str(citation.get("document_id", "")) == str(source.get("document_id", ""))
                        and str(citation.get("page_number", "")) == str(source.get("page_number", ""))
                        and str(citation.get("section_type", "")) == str(source.get("section_type", ""))
                    and bool(evidence_text)
                    and evidence_text in str(source.get("chunk_text", source.get("evidence_text", "")))
                )
            if valid:
                correct += 1
            else:
                invalid.append({"claim_index": claim_index, "citation": citation})

    return {
        "correct_citations": correct,
        "total_citations": total,
        "citation_precision": correct / total if total else 0.0,
        "invalid_citations": invalid,
        "verification_scope": "citation identity and exact quoted-span check only; not semantic entailment",
    }
