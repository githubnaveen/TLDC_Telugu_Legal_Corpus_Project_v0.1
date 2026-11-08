from rdflib import RDF

from src.graph.entity_linker import TLDC


def retrieve_graph_context(graph, document_ids=None, query=None, top_k=10):
    if top_k < 1:
        raise ValueError("top_k must be at least one")
    requested_ids = set(document_ids or [])
    query_terms = {term.casefold() for term in str(query or "").split() if term}
    documents = {}
    for document in graph.subjects(RDF.type, TLDC.LegalDocument):
        document_id = str(next(graph.objects(document, TLDC.documentId), ""))
        if requested_ids and document_id not in requested_ids:
            continue
        documents[document] = document_id

    facts = []
    for document, document_id in documents.items():
        for section in graph.objects(document, TLDC.hasSection):
            text = str(next(graph.objects(section, TLDC.sectionText), ""))
            section_type = str(next(graph.objects(section, TLDC.sectionType), ""))
            page = next(graph.objects(section, TLDC.sourcePage), None)
            section_id = str(section).rsplit("/section/", 1)[-1]
            score = 1.0 if not query_terms else sum(term in text.casefold() for term in query_terms) / len(query_terms)
            if query_terms and score == 0:
                continue
            facts.append({
                "chunk_id": "graph:" + str(section),
                "document_id": document_id,
                "page_number": int(page) if page is not None else None,
                "section_id": section_id,
                "section_type": section_type,
                "chunk_text": text,
                "evidence_text": text,
                "retrieval_source": "knowledge_graph_section",
                "retrieval_score": score,
            })
        for section in graph.objects(document, TLDC.referencesSection):
            source_text = str(next(graph.objects(section, TLDC.sourceText), ""))
            page = next(graph.objects(section, TLDC.sourcePage), None)
            section_id = str(next(graph.objects(section, TLDC.sourceSection), ""))
            identifier = str(next(graph.objects(section, TLDC.sectionIdentifier), ""))
            act = next(graph.objects(section, TLDC.parentAct), None)
            act_name = str(next(graph.objects(act, TLDC.actName), "")) if act else ""
            fact_text = "Statutory reference mentioned: {} Section {}".format(act_name, identifier)
            if query_terms and not any(term in (fact_text + " " + source_text).casefold() for term in query_terms):
                continue
            facts.append({
                "chunk_id": "graph:" + str(section),
                "document_id": document_id,
                "page_number": int(page) if page is not None else None,
                "section_id": section_id.rsplit("/section/", 1)[-1] if section_id else "",
                "section_type": "LEGAL_PROVISIONS",
                "chunk_text": source_text,
                "evidence_text": source_text,
                "graph_fact": fact_text,
                "reference_verification_status": str(next(graph.objects(section, TLDC.referenceVerificationStatus), "UNVERIFIED")),
                "retrieval_source": "knowledge_graph_statute",
                "retrieval_score": 1.0,
            })
    facts.sort(key=lambda fact: (-fact["retrieval_score"], fact["document_id"], fact["page_number"] or 0, fact["chunk_id"]))
    return facts[:top_k]


def combine_text_and_graph(text_results, graph_results, top_k=5, graph_weight=1.0):
    combined = {}
    for source, weight in ((text_results, 1.0), (graph_results, graph_weight)):
        for rank, item in enumerate(source, start=1):
            chunk_id = item["chunk_id"]
            entry = combined.setdefault(chunk_id, dict(item, graph_enhanced_score=0.0, retrieval_sources=[]))
            entry["graph_enhanced_score"] += weight / (60 + rank)
            if item.get("retrieval_source") not in entry["retrieval_sources"]:
                entry["retrieval_sources"].append(item.get("retrieval_source", "text"))
    return sorted(combined.values(), key=lambda item: (-item["graph_enhanced_score"], item["chunk_id"]))[:top_k]
