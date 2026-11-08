import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from rdflib import Graph

from src.evaluation.rag_metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from src.rag.graph_enhanced_retrieval import combine_text_and_graph, retrieve_graph_context
from src.rag.hybrid_retrieval import HybridRetriever


def evaluate_retrieval(queries, chunks, graph=None, top_k=5):
    retriever = HybridRetriever(chunks)
    results = []
    approaches = ("bm25", "vector", "hybrid", "graph_enhanced_hybrid")
    for query_record in queries:
        relevant_ids = set(query_record.get("relevant_chunk_ids", []))
        query = str(query_record.get("question", ""))
        filters = query_record.get("filters")
        keyword = retriever.retrieve(query, top_k=top_k, filters=filters, mode="bm25")
        vector = retriever.retrieve(query, top_k=top_k, filters=filters, mode="vector")
        hybrid = retriever.retrieve(query, top_k=top_k, filters=filters, mode="hybrid")
        graph_results = retrieve_graph_context(
            graph,
            document_ids=[row["document_id"] for row in hybrid] if hybrid else None,
            query=query,
            top_k=top_k,
        ) if graph is not None else []
        graph_hybrid = combine_text_and_graph(hybrid, graph_results, top_k=top_k)
        candidates = {
            "bm25": keyword,
            "vector": vector,
            "hybrid": hybrid,
            "graph_enhanced_hybrid": graph_hybrid,
        }
        for approach, rows in candidates.items():
            ranked_ids = [str(row.get("chunk_id", "")) for row in rows]
            results.append({
                "query_id": query_record.get("query_id", ""),
                "language": query_record.get("language", "UNKNOWN"),
                "approach": approach,
                "answerable": bool(query_record.get("answerable", bool(relevant_ids))),
                "precision_at_k": precision_at_k(ranked_ids, relevant_ids, top_k),
                "recall_at_k": recall_at_k(ranked_ids, relevant_ids, top_k),
                "mrr": reciprocal_rank(ranked_ids, relevant_ids),
                "ndcg_at_k": ndcg_at_k(ranked_ids, relevant_ids, top_k),
                "retrieved_count": len(rows),
                "relevant_chunk_ids": json.dumps(sorted(relevant_ids)),
                "retrieved_chunk_ids": json.dumps(ranked_ids),
            })
    return results


def summarize_results(results):
    grouped = defaultdict(list)
    for row in results:
        grouped[row["approach"]].append(row)
    summary = []
    for approach, rows in sorted(grouped.items()):
        denominator = len(rows)
        summary.append({
            "approach": approach,
            "query_count": denominator,
            "mean_precision_at_k": sum(row["precision_at_k"] for row in rows) / denominator if denominator else 0.0,
            "mean_recall_at_k": sum(row["recall_at_k"] for row in rows) / denominator if denominator else 0.0,
            "mean_mrr": sum(row["mrr"] for row in rows) / denominator if denominator else 0.0,
            "mean_ndcg_at_k": sum(row["ndcg_at_k"] for row in rows) / denominator if denominator else 0.0,
        })
    return summary


def write_csv(path, rows, columns):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate BM25, vector, hybrid, and graph-enhanced retrieval on reviewed gold queries.")
    parser.add_argument("--queries", default="rag_question_evidence_dataset.jsonl")
    parser.add_argument("--chunks", default="data/embedded_chunks/tldc_embedded_chunks.jsonl")
    parser.add_argument("--graph", default="knowledge_graph/tldc_knowledge_graph.ttl")
    parser.add_argument("--comparison", default="results/retrieval_comparison.csv")
    parser.add_argument("--details", default="results/rag_retrieval_details.csv")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    def read_jsonl(path):
        path = Path(path)
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as source:
            return [json.loads(line) for line in source if line.strip()]

    queries = read_jsonl(args.queries)
    chunks = read_jsonl(args.chunks)
    graph = Graph()
    graph_path = Path(args.graph)
    if graph_path.exists():
        graph.parse(graph_path, format="turtle")
    results = evaluate_retrieval(queries, chunks, graph, args.top_k)
    write_csv(args.details, results, [
        "query_id", "language", "approach", "answerable", "precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "retrieved_count", "relevant_chunk_ids", "retrieved_chunk_ids",
    ])
    summary = summarize_results(results)
    write_csv(args.comparison, summary, [
        "approach", "query_count", "mean_precision_at_k", "mean_recall_at_k", "mean_mrr", "mean_ndcg_at_k",
    ])
    print(json.dumps({"queries": len(queries), "chunks": len(chunks), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
