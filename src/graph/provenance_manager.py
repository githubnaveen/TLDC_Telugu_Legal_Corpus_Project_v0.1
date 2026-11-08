import hashlib

from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import XSD

from src.graph.entity_linker import RESOURCE, TLDC, assertion_uri


def add_provenance_assertion(
    graph: Graph,
    subject: URIRef,
    predicate: URIRef,
    obj,
    source_document: URIRef,
    source_page=None,
    source_section=None,
    source_text=None,
    source_url=None,
    assertion_status="EXTRACTED_FROM_ANNOTATION",
):
    identity = "|".join((str(subject), str(predicate), str(obj), str(source_document), str(source_page), str(source_section)))
    assertion = assertion_uri(hashlib.sha256(identity.encode("utf-8")).hexdigest())
    graph.add((assertion, RDF.type, TLDC.ProvenanceAssertion))
    graph.add((assertion, TLDC.assertionSubject, subject))
    graph.add((assertion, TLDC.assertionPredicate, predicate))
    graph.add((assertion, TLDC.assertionObject, obj))
    graph.add((assertion, TLDC.sourceDocument, source_document))
    graph.add((assertion, TLDC.assertionStatus, Literal(assertion_status)))
    if source_page is not None:
        graph.add((assertion, TLDC.sourcePage, Literal(int(source_page), datatype=XSD.integer)))
    if source_section is not None:
        graph.add((assertion, TLDC.sourceSection, source_section))
    if source_text:
        graph.add((assertion, TLDC.sourceText, Literal(source_text)))
    if source_url:
        graph.add((assertion, TLDC.sourceUrl, Literal(source_url)))
    return assertion
