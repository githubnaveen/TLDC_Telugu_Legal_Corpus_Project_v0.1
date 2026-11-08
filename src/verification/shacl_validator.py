import argparse
import json
from pathlib import Path
from typing import Union

from pyshacl import validate
from rdflib import Graph


def validate_graph(data_graph, shapes_graph, ontology_graph=None):
    if not isinstance(data_graph, Graph):
        loaded_data = Graph()
        loaded_data.parse(str(data_graph), format="turtle")
        data_graph = loaded_data
    if not isinstance(shapes_graph, Graph):
        loaded_shapes = Graph()
        loaded_shapes.parse(str(shapes_graph), format="turtle")
        shapes_graph = loaded_shapes
    if ontology_graph is not None and not isinstance(ontology_graph, Graph):
        loaded_ontology = Graph()
        loaded_ontology.parse(str(ontology_graph), format="turtle")
        ontology_graph = loaded_ontology

    conforms, report_graph, report_text = validate(
        data_graph,
        shacl_graph=shapes_graph,
        ont_graph=ontology_graph,
        inference="none",
        abort_on_first=False,
        allow_infos=False,
        allow_warnings=False,
    )
    return {
        "conforms": bool(conforms),
        "report_text": str(report_text),
        "report_graph": report_graph,
        "data_triples": len(data_graph),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate a TLDC Turtle graph using SHACL shapes.")
    parser.add_argument("--graph", default="knowledge_graph/tldc_knowledge_graph.ttl")
    parser.add_argument("--shapes", default="validation/tldc_validation_shapes.ttl")
    parser.add_argument("--ontology", default="ontology/tldc_ontology_v1.owl")
    parser.add_argument("--report", default="validation/validation_reports/shacl_report.ttl")
    args = parser.parse_args()

    result = validate_graph(args.graph, args.shapes, args.ontology)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result["report_graph"].serialize(destination=str(report_path), format="turtle", encoding="utf-8")
    print(json.dumps({"conforms": result["conforms"], "data_triples": result["data_triples"], "report": result["report_text"]}, indent=2))
    if not result["conforms"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
