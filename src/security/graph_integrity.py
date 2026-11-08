from rdflib import RDF

from src.graph.entity_linker import TLDC

SOURCED_RELATIONS = {
    TLDC.belongsToCase,
    TLDC.hasDocument,
    TLDC.heardByCourt,
    TLDC.referencesAct,
    TLDC.referencesSection,
    TLDC.containsEvidence,
    TLDC.containsArgument,
    TLDC.containsObservation,
    TLDC.resultsInDecision,
}


def audit_graph_assertion_provenance(graph):
    assertions = set(graph.subjects(RDF.type, TLDC.ProvenanceAssertion))
    covered = set()
    for assertion in assertions:
        subject = next(graph.objects(assertion, TLDC.assertionSubject), None)
        predicate = next(graph.objects(assertion, TLDC.assertionPredicate), None)
        obj = next(graph.objects(assertion, TLDC.assertionObject), None)
        source = next(graph.objects(assertion, TLDC.sourceDocument), None)
        page = next(graph.objects(assertion, TLDC.sourcePage), None)
        if subject is not None and predicate is not None and obj is not None and source is not None and page is not None:
            covered.add((subject, predicate, obj))

    required = [(subject, predicate, obj) for predicate in SOURCED_RELATIONS for subject, obj in graph.subject_objects(predicate)]
    missing = [triple for triple in required if triple not in covered]
    return {
        "sourced_relationship_count": len(required),
        "relationships_with_document_page_assertion": len(required) - len(missing),
        "unprovenanced_assertion_count": len(missing),
        "unprovenanced_relationships": [{"subject": str(s), "predicate": str(p), "object": str(o)} for s, p, o in missing],
        "interpretation": "Structural provenance audit only; does not validate factual truth or legal correctness.",
    }
