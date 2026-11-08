import argparse
import csv
import json
import unicodedata
from pathlib import Path

from src.rag.hybrid_retrieval import HybridRetriever
from src.security.unicode_robustness import compare_retrieval_consistency


def main():
    parser = argparse.ArgumentParser(description="Report Unicode controls and retrieval consistency on controlled variants.")
    parser.add_argument("--input", default="adversarial_tests/unicode/telugu_adversarial_testset.jsonl")
    parser.add_argument("--chunks", default="data/embedded_chunks/tldc_embedded_chunks.jsonl")
    parser.add_argument("--output", default="results/unicode_robustness.csv")
    args = parser.parse_args()
    input_path = Path(args.input)
    tests = []
    if input_path.exists():
        with input_path.open(encoding="utf-8") as source:
            tests = [json.loads(line) for line in source if line.strip()]
    chunks_path = Path(args.chunks)
    chunks = []
    if chunks_path.exists():
        with chunks_path.open(encoding="utf-8") as source:
            chunks = [json.loads(line) for line in source if line.strip()]
    retriever = HybridRetriever(chunks)
    rows = []
    for test in tests:
        result = compare_retrieval_consistency(
            retriever,
            test.get("clean_text", ""),
            test.get("perturbed_text", ""),
            relevant_chunk_ids=test.get("relevant_chunk_ids", []),
        )
        clean_norm = unicodedata.normalize("NFC", test.get("clean_text", ""))
        perturbed_norm = unicodedata.normalize("NFC", test.get("perturbed_text", ""))
        rows.append({
            "test_id": test.get("test_id", ""),
            "attack_type": test.get("attack_type", ""),
            "normalization_equivalent": clean_norm == perturbed_norm,
            "perturbed_requires_review": result["perturbed_flags"]["requires_review"],
            "invisible_format_characters": result["perturbed_flags"]["invisible_format_characters"],
            "bidirectional_controls": result["perturbed_flags"]["bidirectional_controls"],
            "top_k_jaccard": result["top_k_jaccard"],
            "clean_recall_at_k": result["clean_recall_at_k"],
            "perturbed_recall_at_k": result["perturbed_recall_at_k"],
            "review_status": test.get("review_status", "PENDING"),
        })
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["test_id", "attack_type", "normalization_equivalent", "perturbed_requires_review", "invisible_format_characters", "bidirectional_controls", "top_k_jaccard", "clean_recall_at_k", "perturbed_recall_at_k", "review_status"]
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print("Unicode test variants evaluated:", len(rows), "retrievable chunks:", len(retriever.chunks))


if __name__ == "__main__":
    main()
