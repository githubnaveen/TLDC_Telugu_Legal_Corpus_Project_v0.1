import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Union

from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import XSD

from src.graph.entity_linker import (
    RESOURCE,
    TLDC,
    case_uri,
    court_uri,
    decision_uri,
    document_uri,
    legal_act_uri,
    legal_section_uri,
    section_uri,
)
from src.graph.provenance_manager import add_provenance_assertion

APPROVED_STATUSES = {"APPROVED", "ADJUDICATED"}
PRIVACY_CLEARED_STATUSES = {"APPROVED", "ANONYMISED", "ANONYMIZED", "REVIEWED"}
DECISION_OUTCOMES = {"GRANTED", "REJECTED", "PARTLY_GRANTED", "UNKNOWN", "AMBIGUOUS"}
SECTION_TYPES = {
    "FACTS": TLDC.Fact,
    "PETITIONER_ARGUMENT": TLDC.LegalArgument,
    "RESPONDENT_ARGUMENT": TLDC.LegalArgument,
    "LEGAL_PROVISIONS": TLDC.LegalProvisionPassage,
    "EVIDENCE": TLDC.Evidence,
    "JUDICIAL_OBSERVATION": TLDC.JudicialObservation,
    "DECISION": TLDC.DecisionPassage,
}


def _status(value):
    return str(value or "").strip().upper()


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
    return rows


def _read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def _normalize_alias(value):
    return " ".join(str(value or "").strip().casefold().split())


def _read_verified_aliases(path):
    aliases = {}
    for row in _read_csv(path) if path else []:
        if _status(row.get("verification_status")) != "VERIFIED":
            continue
        alias = _normalize_alias(row.get("alias_label"))
        canonical_id = str(row.get("canonical_act_id", "")).strip()
        canonical_label = str(row.get("canonical_label", "")).strip()
        authority_url = str(row.get("source_url", "")).strip()
        if alias and canonical_id and canonical_label and authority_url:
            aliases[alias] = canonical_id
    return aliases


def _approved(document, register_row):
    if register_row is None:
        return False, "missing_register_record"
    if _status(document.get("annotation_status")) not in APPROVED_STATUSES:
        return False, "annotation_not_approved"
    if _status(register_row.get("annotation_status")) not in APPROVED_STATUSES:
        return False, "register_annotation_not_approved"
    if _status(register_row.get("privacy_status")) not in PRIVACY_CLEARED_STATUSES:
        return False, "privacy_not_cleared"
    return True, ""


def _literal(graph, subject, predicate, value, datatype=None):
    if value is not None and str(value).strip():
        graph.add((subject, predicate, Literal(str(value).strip(), datatype=datatype)))


def _add_section(graph, document_id, document_node, section):
    section_id = str(section.get("section_id", "")).strip()
    section_type = str(section.get("section_type", "")).strip().upper()
    text = str(section.get("text", "")).strip()
    page = section.get("page_number")
    if not section_id or not section_type or not text:
        return None
    try:
        page = int(page)
    except (TypeError, ValueError):
        return None
    if page < 1:
        return None

    node = section_uri(document_id, section_id)
    graph.add((node, RDF.type, TLDC.DocumentSection))
    graph.add((node, RDF.type, SECTION_TYPES.get(section_type, TLDC.DocumentSection)))
    graph.add((node, TLDC.sectionType, Literal(section_type)))
    graph.add((node, TLDC.sectionText, Literal(text)))
    graph.add((node, TLDC.sourceDocument, document_node))
    graph.add((node, TLDC.sourcePage, Literal(page, datatype=XSD.integer)))
    graph.add((document_node, TLDC.hasSection, node))
    add_provenance_assertion(
        graph, document_node, TLDC.hasSection, node, document_node,
        source_page=page, source_section=node, source_text=text,
    )
    return {
        "node": node,
        "section_id": section_id,
        "section_type": section_type,
        "text": text,
        "page_number": page,
    }


def _register_map(rows):
    return {
        str(row.get("document_id", "")).strip(): row
        for row in rows
        if str(row.get("document_id", "")).strip()
    }


def build_graph(
    segmentation_path: Union[str, Path],
    register_path: Union[str, Path],
    statutory_path: Optional[Union[str, Path]] = None,
    graph_output: Optional[Union[str, Path]] = None,
    aliases_path: Optional[Union[str, Path]] = None,
):
    segmentations = _read_jsonl(segmentation_path)
    register = _register_map(_read_csv(register_path))
    statutory_rows = _read_csv(statutory_path) if statutory_path else []
    verified_aliases = _read_verified_aliases(aliases_path)
    statutory_by_document = {}
    for row in statutory_rows:
        statutory_by_document.setdefault(str(row.get("document_id", "")).strip(), []).append(row)

    graph = Graph()
    graph.bind("tldc", TLDC)
    graph.bind("tldcr", RESOURCE)
    stats = Counter({
        "documents_added": 0,
        "documents_skipped": 0,
        "sections_skipped_invalid_provenance": 0,
        "statutory_references_added": 0,
        "decisions_added": 0,
    })
    eligible_documents = {}
    document_sections = {}

    for document in segmentations:
        doc_id = str(document.get("document_id", "")).strip()
        if not doc_id:
            stats["documents_skipped"] += 1
            stats["skipped_missing_document_id"] += 1
            continue
        register_row = register.get(doc_id)
        eligible, reason = _approved(document, register_row)
        if not eligible:
            stats["documents_skipped"] += 1
            stats["skipped_" + reason] += 1
            continue

        doc_node = document_uri(doc_id)
        graph.add((doc_node, RDF.type, TLDC.LegalDocument))
        graph.add((doc_node, TLDC.documentId, Literal(doc_id)))
        document_type = document.get("document_type") or register_row.get("document_type")
        language = document.get("document_language") or register_row.get("document_language")
        _literal(graph, doc_node, TLDC.documentType, document_type)
        _literal(graph, doc_node, TLDC.documentLanguage, language)
        _literal(graph, doc_node, TLDC.state, register_row.get("state") or document.get("state"))
        _literal(graph, doc_node, TLDC.district, register_row.get("district") or document.get("district"))
        _literal(graph, doc_node, TLDC.sourceUrl, register_row.get("source_url"), XSD.anyURI)

        case_group_id = str(register_row.get("case_group_id", "")).strip()
        source_url = str(register_row.get("source_url", "")).strip() or None
        if case_group_id:
            case_node = case_uri(case_group_id)
            graph.add((case_node, RDF.type, TLDC.Case))
            graph.add((case_node, TLDC.caseId, Literal(case_group_id)))
            _literal(graph, case_node, TLDC.caseCategory, register_row.get("case_category"))
            graph.add((doc_node, TLDC.belongsToCase, case_node))
            graph.add((case_node, TLDC.hasDocument, doc_node))
            add_provenance_assertion(
                graph, doc_node, TLDC.belongsToCase, case_node, doc_node,
                source_url=source_url, assertion_status="FROM_APPROVED_REGISTER",
            )
            add_provenance_assertion(
                graph, case_node, TLDC.hasDocument, doc_node, doc_node,
                source_url=source_url, assertion_status="FROM_APPROVED_REGISTER",
            )

            court_name = str(register_row.get("court_name", "")).strip()
            if court_name:
                court_node = court_uri(court_name)
                graph.add((court_node, RDF.type, TLDC.Court))
                graph.add((court_node, TLDC.courtName, Literal(court_name)))
                _literal(graph, court_node, TLDC.courtLevel, register_row.get("court_level"))
                _literal(graph, court_node, TLDC.jurisdiction, register_row.get("state") or document.get("state"))
                graph.add((case_node, TLDC.heardByCourt, court_node))
                add_provenance_assertion(
                    graph, case_node, TLDC.heardByCourt, court_node, doc_node,
                    source_url=source_url, assertion_status="FROM_APPROVED_REGISTER",
                )

        sections = []
        for section in document.get("sections", []):
            section_record = _add_section(graph, doc_id, doc_node, section)
            if section_record is None:
                stats["sections_skipped_invalid_provenance"] += 1
                continue
            sections.append(section_record)
            section_type = section_record["section_type"]
            relation = {
                "EVIDENCE": TLDC.containsEvidence,
                "PETITIONER_ARGUMENT": TLDC.containsArgument,
                "RESPONDENT_ARGUMENT": TLDC.containsArgument,
                "JUDICIAL_OBSERVATION": TLDC.containsObservation,
            }.get(section_type)
            if relation:
                graph.add((doc_node, relation, section_record["node"]))
                add_provenance_assertion(
                    graph, doc_node, relation, section_record["node"], doc_node,
                    source_page=section_record["page_number"],
                    source_section=section_record["node"],
                    source_text=section_record["text"],
                )

        eligible_documents[doc_id] = (document, register_row, doc_node)
        document_sections[doc_id] = sections
        stats["documents_added"] += 1

    for row in statutory_rows:
        doc_id = str(row.get("document_id", "")).strip()
        eligible = eligible_documents.get(doc_id)
        if eligible is None:
            stats["statutory_references_skipped_unapproved_document"] += 1
            continue
        act_name = str(row.get("act_name", "")).strip()
        section_number = str(row.get("section_number", "")).strip()
        source_text = str(row.get("source_text", "")).strip()
        if not act_name or not section_number or not source_text:
            stats["statutory_references_skipped_incomplete"] += 1
            continue
        try:
            source_page = int(row.get("page_number"))
        except (TypeError, ValueError):
            stats["statutory_references_skipped_missing_page"] += 1
            continue
        source_section = next((
            section for section in document_sections[doc_id]
            if section["page_number"] == source_page
            and section["section_type"] == "LEGAL_PROVISIONS"
            and source_text in section["text"]
        ), None)
        if source_section is None:
            stats["statutory_references_skipped_unmatched_source_text"] += 1
            continue

        _, _, doc_node = eligible
        canonical_act_id = verified_aliases.get(_normalize_alias(act_name))
        act_node = legal_act_uri(act_name, canonical_act_id)
        section_node = legal_section_uri(act_name, section_number, canonical_act_id)
        graph.add((act_node, RDF.type, TLDC.LegalAct))
        graph.add((act_node, TLDC.actName, Literal(act_name)))
        if canonical_act_id:
            graph.add((act_node, TLDC.canonicalActId, Literal(canonical_act_id)))
        graph.add((section_node, RDF.type, TLDC.LegalSection))
        graph.add((section_node, TLDC.sectionIdentifier, Literal(section_number)))
        graph.add((section_node, TLDC.parentAct, act_node))
        graph.add((section_node, TLDC.referenceVerificationStatus, Literal("MENTIONED_UNVERIFIED")))
        graph.add((section_node, TLDC.sourceDocument, doc_node))
        graph.add((section_node, TLDC.sourcePage, Literal(source_page, datatype=XSD.integer)))
        graph.add((section_node, TLDC.sourceText, Literal(source_text)))
        graph.add((doc_node, TLDC.referencesAct, act_node))
        graph.add((doc_node, TLDC.referencesSection, section_node))
        add_provenance_assertion(
            graph, doc_node, TLDC.referencesAct, act_node, doc_node,
            source_page=source_page, source_section=source_section["node"], source_text=source_text,
        )
        add_provenance_assertion(
            graph, doc_node, TLDC.referencesSection, section_node, doc_node,
            source_page=source_page, source_section=source_section["node"], source_text=source_text,
        )
        stats["statutory_references_added"] += 1

    for doc_id, (document, _, doc_node) in eligible_documents.items():
        decision = document.get("decision") or {}
        outcome = str(decision.get("outcome", "")).strip().upper()
        if outcome not in DECISION_OUTCOMES:
            if outcome:
                stats["decisions_skipped_invalid_outcome"] += 1
            continue
        source_section = next((section for section in document_sections[doc_id] if section["section_type"] == "DECISION"), None)
        if source_section is None:
            stats["decisions_skipped_missing_source_passage"] += 1
            continue
        decision_node = decision_uri(doc_id)
        graph.add((decision_node, RDF.type, TLDC.Decision))
        graph.add((decision_node, TLDC.outcomeStatus, Literal(outcome)))
        graph.add((decision_node, TLDC.sourceDocument, doc_node))
        graph.add((decision_node, TLDC.sourcePage, Literal(source_section["page_number"], datatype=XSD.integer)))
        graph.add((decision_node, TLDC.sourceSection, source_section["node"]))
        graph.add((decision_node, TLDC.sourceText, Literal(source_section["text"])))
        graph.add((doc_node, TLDC.resultsInDecision, decision_node))
        add_provenance_assertion(
            graph, doc_node, TLDC.resultsInDecision, decision_node, doc_node,
            source_page=source_section["page_number"],
            source_section=source_section["node"],
            source_text=source_section["text"],
        )
        stats["decisions_added"] += 1

    if graph_output:
        output_path = Path(graph_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if len(graph) == 0:
            output_path.write_text(
                "@prefix tldc: <https://tldc.example.org/ontology#> .\n"
                "@prefix tldcr: <https://tldc.example.org/resource/> .\n"
                "@prefix prov: <http://www.w3.org/ns/prov#> .\n",
                encoding="utf-8",
            )
        else:
            graph.serialize(destination=str(output_path), format="turtle", encoding="utf-8")

    return graph, dict(stats)


def export_registries(graph: Graph, entity_path: Union[str, Path], relationship_path: Union[str, Path]):
    entity_classes = [
        TLDC.LegalDocument, TLDC.Case, TLDC.Court, TLDC.LegalAct, TLDC.LegalSection,
        TLDC.DocumentSection, TLDC.Evidence, TLDC.LegalArgument, TLDC.JudicialObservation, TLDC.Decision,
    ]
    entity_types = {}
    for entity_type in entity_classes:
        for entity in graph.subjects(RDF.type, entity_type):
            entity_types.setdefault(entity, set()).add(str(entity_type).rsplit("#", 1)[-1])
    entity_rows = []
    label_predicates = (
        TLDC.sectionText, TLDC.documentId, TLDC.caseId, TLDC.courtName,
        TLDC.actName, TLDC.sectionIdentifier, TLDC.outcomeStatus,
    )
    for entity in sorted(entity_types, key=str):
        types = entity_types[entity]
        label = ""
        for predicate in label_predicates:
            label_value = next(graph.objects(entity, predicate), None)
            if label_value is not None:
                label = str(label_value)
                break
        entity_rows.append({
            "entity_uri": str(entity),
            "entity_type": ";".join(sorted(types)),
            "label": label,
            "source_document": str(next(graph.objects(entity, TLDC.sourceDocument), "")),
            "source_page": str(next(graph.objects(entity, TLDC.sourcePage), "")),
        })
    relation_rows = []
    for assertion in sorted(graph.subjects(RDF.type, TLDC.ProvenanceAssertion), key=str):
        relation_rows.append({
            "subject_uri": str(next(graph.objects(assertion, TLDC.assertionSubject), "")),
            "predicate_uri": str(next(graph.objects(assertion, TLDC.assertionPredicate), "")),
            "object": str(next(graph.objects(assertion, TLDC.assertionObject), "")),
            "source_document": str(next(graph.objects(assertion, TLDC.sourceDocument), "")),
            "source_page": str(next(graph.objects(assertion, TLDC.sourcePage), "")),
            "source_section": str(next(graph.objects(assertion, TLDC.sourceSection), "")),
            "source_text": str(next(graph.objects(assertion, TLDC.sourceText), "")),
            "review_status": str(next(graph.objects(assertion, TLDC.assertionStatus), "")),
        })
    _write_csv(entity_path, ["entity_uri", "entity_type", "label", "source_document", "source_page"], entity_rows)
    _write_csv(relationship_path, ["subject_uri", "predicate_uri", "object", "source_document", "source_page", "source_section", "source_text", "review_status"], relation_rows)


def _write_csv(path, columns, rows):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Build a provenance-aware TLDC RDF graph from approved annotations.")
    parser.add_argument("--segments", default="annotations/document_segmentation/segmented_documents.jsonl")
    parser.add_argument("--register", default="metadata/document_register.csv")
    parser.add_argument("--statutes", default="annotations/statutory_references/statutory_references.csv")
    parser.add_argument("--aliases", default="metadata/legal_reference_aliases.csv")
    parser.add_argument("--output", default="knowledge_graph/tldc_knowledge_graph.ttl")
    parser.add_argument("--entities", default="knowledge_graph/entity_registry.csv")
    parser.add_argument("--relationships", default="knowledge_graph/relationship_registry.csv")
    parser.add_argument("--stats", default="reports/graph_build_stats.json")
    args = parser.parse_args()

    graph, stats = build_graph(args.segments, args.register, args.statutes, args.output, args.aliases)
    export_registries(graph, args.entities, args.relationships)
    stats_path = Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({"triples": len(graph), **stats}, indent=2))


if __name__ == "__main__":
    main()
