import unicodedata


def _normal(value):
    return " ".join(unicodedata.normalize("NFC", str(value or "")).casefold().split())


def _stable_set(values):
    return {_normal(value) for value in (values or [])}


def evaluate_counterfactual_pairs(pairs):
    rows = []
    excluded = 0
    for pair in pairs:
        valid = (
            str(pair.get("review_status", "")).upper() == "ADJUDICATED"
            and pair.get("changed_attribute_irrelevant") is True
            and pair.get("legally_relevant_facts_preserved") is True
            and isinstance(pair.get("original"), dict)
            and isinstance(pair.get("counterfactual"), dict)
        )
        if not valid:
            excluded += 1
            continue
        original = pair["original"]
        counterfactual = pair["counterfactual"]
        conclusion_consistent = _normal(original.get("conclusion")) == _normal(counterfactual.get("conclusion"))
        citations_consistent = _stable_set(original.get("citations")) == _stable_set(counterfactual.get("citations"))
        claims_consistent = _stable_set(original.get("claims")) == _stable_set(counterfactual.get("claims"))
        rows.append({
            "pair_id": pair.get("pair_id", ""),
            "conclusion_consistent": conclusion_consistent,
            "citations_consistent": citations_consistent,
            "claims_consistent": claims_consistent,
            "pair_consistent": conclusion_consistent and citations_consistent and claims_consistent,
            "changed_attribute": pair.get("changed_attribute", ""),
            "review_status": "ADJUDICATED",
        })
    count = len(rows)
    return {
        "valid_pair_count": count,
        "excluded_pair_count": excluded,
        "counterfactual_consistency": sum(row["pair_consistent"] for row in rows) / count if count else None,
        "conclusion_consistency": sum(row["conclusion_consistent"] for row in rows) / count if count else None,
        "citation_consistency": sum(row["citations_consistent"] for row in rows) / count if count else None,
        "claim_consistency": sum(row["claims_consistent"] for row in rows) / count if count else None,
        "pairs": rows,
        "interpretation": "Descriptive consistency on expert-adjudicated pairs only; not proof of fairness.",
    }
