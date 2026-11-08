import argparse
import json
from pathlib import Path

from rdflib import Graph

from src.security.graph_integrity import audit_graph_assertion_provenance
from src.verification.shacl_validator import validate_graph


def validate_graph_security(graph_path, shapes_path, ontology_path=None):
    graph = Graph()
    graph.parse(str(graph_path), format="turtle")
    shacl = validate_graph(graph, shapes_path, ontology_path)
    provenance = audit_graph_assertion_provenance(graph)
    return {
        "triple_count": len(graph),
        "shacl_conforms": shacl["conforms"],
        "shacl_report": shacl["report_text"],
        "provenance_audit": provenance,
        "factual_truth_verified": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit TLDC knowledge graph structure and assertion provenance.")
    parser.add_argument("--graph", default="knowledge_graph/tldc_knowledge_graph.ttl")
    parser.add_argument("--shapes", default="validation/tldc_validation_shapes.ttl")
    parser.add_argument("--ontology", default="ontology/tldc_ontology_v1.owl")
    parser.add_argument("--output", default="results/graph_integrity_report.json")
    args = parser.parse_args()
    result = validate_graph_security(args.graph, args.shapes, args.ontology)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
