import csv
import json
import re
from pathlib import Path

from rdflib import Graph, Literal, RDF, URIRef

from src.graph.entity_linker import TLDC, document_uri, legal_section_uri
from src.graph.graph_builder import build_graph, export_registries
from src.evaluation.graph_metrics import provenance_coverage, relation_metrics
from src.verification.reference_checker import check_references
from src.verification.shacl_validator import validate_graph

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHAPES_PATH = PROJECT_ROOT / "validation" / "tldc_validation_shapes.ttl"


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_register(path, rows):
    headers = ["document_id", "state", "district", "court_name", "court_level", "case_group_id", "case_category", "document_type", "document_language", "source_url", "privacy_status", "annotation_status"]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def test_pending_week4_records_are_not_added_to_graph(tmp_path):
    graph_path = tmp_path / "empty_graph.ttl"
    graph, stats = build_graph(
        PROJECT_ROOT / "annotations/document_segmentation/segmented_documents.jsonl",
        PROJECT_ROOT / "metadata/document_register.csv",
        PROJECT_ROOT / "annotations/statutory_references/statutory_references.csv",
        graph_path,
    )
    assert len(graph) == 0
    assert "@prefix tldc:" in graph_path.read_text(encoding="utf-8")
    parsed_graph = Graph()
    parsed_graph.parse(graph_path, format="turtle")
    assert len(parsed_graph) == 0
    assert stats["documents_added"] == 0
    assert stats["documents_skipped"] >= 2


def test_approved_annotations_create_traceable_rdf_and_validate(tmp_path):
    document_id = "TLDC-TS-TEST-0001"
    segmentation = tmp_path / "segments.jsonl"
    register = tmp_path / "register.csv"
    statutes = tmp_path / "statutes.csv"
    aliases = tmp_path / "aliases.csv"
    graph_path = tmp_path / "graph.ttl"

    write_jsonl(segmentation, [{
        "document_id": document_id,
        "state": "Telangana",
        "document_language": "Mixed",
        "document_type": "Bail Order",
        "annotation_status": "APPROVED",
        "decision": {"outcome": "GRANTED"},
        "sections": [
            {"section_id": "S001", "section_type": "LEGAL_PROVISIONS", "text": "Section 302 IPC", "page_number": 2},
            {"section_id": "S002", "section_type": "EVIDENCE", "text": "Documentary exhibit", "page_number": 3},
            {"section_id": "S003", "section_type": "JUDICIAL_OBSERVATION", "text": "The court considered the record.", "page_number": 4},
            {"section_id": "S004", "section_type": "DECISION", "text": "Bail granted.", "page_number": 5},
        ],
    }])
    write_register(register, [{
        "document_id": document_id,
        "state": "Telangana",
        "court_name": "Test High Court",
        "court_level": "High Court",
        "case_group_id": "CASE-TEST-1",
        "case_category": "Criminal",
        "document_type": "Bail Order",
        "document_language": "Mixed",
        "source_url": "https://example.test/order",
        "privacy_status": "APPROVED",
        "annotation_status": "APPROVED",
    }])
    with statutes.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["document_id", "act_name", "section_number", "source_text", "page_number", "language"])
        writer.writeheader()
        writer.writerow({"document_id": document_id, "act_name": "Example Act", "section_number": "302", "source_text": "Section 302 IPC", "page_number": "2", "language": "English"})
    with aliases.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["alias_label", "canonical_act_id", "canonical_label", "state_scope", "source_url", "verification_status"])
        writer.writeheader()
        writer.writerow({"alias_label": "Example Act", "canonical_act_id": "ACT-EXAMPLE", "canonical_label": "Example Act", "source_url": "https://authority.example/act", "verification_status": "VERIFIED"})

    graph, stats = build_graph(segmentation, register, statutes, graph_path, aliases)
    document = document_uri(document_id)
    provision = legal_section_uri("Example Act", "302", "ACT-EXAMPLE")
    assert stats["documents_added"] == 1
    assert (document, RDF.type, TLDC.LegalDocument) in graph
    assert (document, TLDC.referencesSection, provision) in graph
    assert graph_path.exists()
    assert validate_graph(graph, SHAPES_PATH)["conforms"] is True
    entities_path = tmp_path / "entity_registry.csv"
    relationships_path = tmp_path / "relationship_registry.csv"
    export_registries(graph, entities_path, relationships_path)
    with entities_path.open(encoding="utf-8", newline="") as source:
        entity_rows = list(csv.DictReader(source))
    with relationships_path.open(encoding="utf-8", newline="") as source:
        relationship_rows = list(csv.DictReader(source))
    assert len(entity_rows) == len({row["entity_uri"] for row in entity_rows})
    assert relationship_rows


def test_reference_check_requires_matching_act_and_section():
    data_graph = Graph()
    section = legal_section_uri("Example Act", "302")
    data_act = URIRef("https://tldc.example.org/resource/act/example")
    data_graph.add((section, RDF.type, TLDC.LegalSection))
    data_graph.add((section, TLDC.sectionIdentifier, Literal("302")))
    data_graph.add((section, TLDC.parentAct, data_act))
    data_graph.add((data_act, TLDC.actName, Literal("Example Act")))

    authority_graph = Graph()
    authority_section = URIRef("https://authority.example/section/302")
    authority_act = URIRef("https://authority.example/act/other-act")
    authority_graph.add((authority_section, RDF.type, TLDC.LegalSection))
    authority_graph.add((authority_section, TLDC.sectionIdentifier, Literal("302")))
    authority_graph.add((authority_section, TLDC.parentAct, authority_act))
    authority_graph.add((authority_act, TLDC.actName, Literal("Different Act")))
    assert check_references(data_graph, authority_graph)[0]["status"] == "NOT_FOUND_IN_SUPPLIED_AUTHORITY_GRAPH"

    authority_graph.remove((authority_act, TLDC.actName, Literal("Different Act")))
    authority_graph.add((authority_act, TLDC.actName, Literal("Example Act")))
    assert check_references(data_graph, authority_graph)[0]["status"] == "MATCH_IN_SUPPLIED_AUTHORITY_GRAPH"


def test_shacl_rejects_evidence_without_page_provenance():
    graph = Graph()
    evidence = URIRef("https://tldc.example.org/resource/evidence/invalid")
    graph.add((evidence, RDF.type, TLDC.Evidence))
    graph.add((evidence, TLDC.sourceDocument, document_uri("DOC-X")))
    report = validate_graph(graph, SHAPES_PATH)
    assert report["conforms"] is False
    assert "sourcePage" in report["report_text"]


def test_graph_evaluation_metrics_report_relation_and_provenance_coverage():
    assert relation_metrics({"a", "b"}, {"a", "c"})["f1"] == 0.5
    graph = Graph()
    assertion_with_page = URIRef("https://tldc.example.org/resource/assertion/with-page")
    assertion_without_page = URIRef("https://tldc.example.org/resource/assertion/no-page")
    graph.add((assertion_with_page, RDF.type, TLDC.ProvenanceAssertion))
    graph.add((assertion_with_page, TLDC.sourceDocument, document_uri("DOC-1")))
    graph.add((assertion_with_page, TLDC.sourcePage, Literal(1)))
    graph.add((assertion_without_page, RDF.type, TLDC.ProvenanceAssertion))
    graph.add((assertion_without_page, TLDC.sourceDocument, document_uri("DOC-2")))
    coverage = provenance_coverage(graph, TLDC.ProvenanceAssertion, TLDC.sourceDocument, TLDC.sourcePage)
    assert coverage == {"assertion_count": 2, "covered_count": 1, "coverage": 0.5}


def test_ontology_shapes_and_sparql_queries_parse():
    ontology = Graph()
    ontology.parse(PROJECT_ROOT / "ontology/tldc_ontology_v1.owl", format="turtle")
    shapes = Graph()
    shapes.parse(SHAPES_PATH, format="turtle")
    assert len(ontology) > 0
    assert len(shapes) > 0

    for query_path in (
        PROJECT_ROOT / "queries/legal_graph_queries.sparql",
        PROJECT_ROOT / "queries/missing_evidence_queries.sparql",
    ):
        query_text = query_path.read_text(encoding="utf-8")
        prefixes = "\n".join(line for line in query_text.splitlines() if line.startswith("PREFIX ")) + "\n"
        body = "\n".join(line for line in query_text.splitlines() if not line.startswith("PREFIX "))
        for query_body in re.split(r"(?m)^SELECT\b", body)[1:]:
            Graph().query(prefixes + "SELECT" + query_body)
