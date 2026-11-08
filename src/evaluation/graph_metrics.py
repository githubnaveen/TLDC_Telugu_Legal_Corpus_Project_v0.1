from rdflib import RDF


def relation_metrics(predicted, gold):
    predicted_set = set(predicted)
    gold_set = set(gold)
    true_positive = len(predicted_set & gold_set)
    false_positive = len(predicted_set - gold_set)
    false_negative = len(gold_set - predicted_set)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def provenance_coverage(graph, assertion_class, source_document_predicate, source_page_predicate):
    assertions = set(graph.subjects(RDF.type, assertion_class))
    if not assertions:
        return {"assertion_count": 0, "covered_count": 0, "coverage": 0.0}
    covered = sum(
        1 for assertion in assertions
        if list(graph.objects(assertion, source_document_predicate))
        and list(graph.objects(assertion, source_page_predicate))
    )
    return {
        "assertion_count": len(assertions),
        "covered_count": covered,
        "coverage": covered / len(assertions),
    }
