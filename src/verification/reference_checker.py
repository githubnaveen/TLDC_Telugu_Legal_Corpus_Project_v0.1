import unicodedata

from rdflib import Graph, RDF

from src.graph.entity_linker import TLDC


def _normalize(value):
    return " ".join(unicodedata.normalize("NFC", str(value or "")).casefold().split())


def check_references(data_graph, authority_graph=None):
    references = []
    if authority_graph is None:
        for section in data_graph.subjects(RDF.type, TLDC.LegalSection):
            references.append({
                "section_uri": str(section),
                "status": "UNVERIFIED_NO_AUTHORITY_GRAPH",
                "reason": "A selected authoritative source graph was not supplied.",
            })
        return references

    authority_pairs = set()
    for authority_section in authority_graph.subjects(RDF.type, TLDC.LegalSection):
        identifier = next(authority_graph.objects(authority_section, TLDC.sectionIdentifier), None)
        parent_act = next(authority_graph.objects(authority_section, TLDC.parentAct), None)
        act_keys = _act_keys(authority_graph, parent_act)
        if identifier is not None:
            authority_pairs.update((act_key, _normalize(identifier)) for act_key in act_keys)

    for section in data_graph.subjects(RDF.type, TLDC.LegalSection):
        identifier = next(data_graph.objects(section, TLDC.sectionIdentifier), None)
        parent_act = next(data_graph.objects(section, TLDC.parentAct), None)
        act_keys = _act_keys(data_graph, parent_act)
        if identifier is None or not act_keys:
            status = "INCOMPLETE_REFERENCE"
        elif any((act_key, _normalize(identifier)) in authority_pairs for act_key in act_keys):
            status = "MATCH_IN_SUPPLIED_AUTHORITY_GRAPH"
        else:
            status = "NOT_FOUND_IN_SUPPLIED_AUTHORITY_GRAPH"
        references.append({
            "section_uri": str(section),
            "section_identifier": str(identifier or ""),
            "status": status,
            "reason": "Exact identifier comparison only; this does not validate legal applicability or interpretation.",
        })
    return references


def _act_keys(graph, act_node):
    if act_node is None:
        return set()
    canonical_id = next(graph.objects(act_node, TLDC.canonicalActId), None)
    act_name = next(graph.objects(act_node, TLDC.actName), None)
    return {_normalize(value) for value in (canonical_id, act_name) if value is not None and _normalize(value)}
