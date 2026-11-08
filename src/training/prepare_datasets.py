import argparse
from collections import Counter
from pathlib import Path

from src.training.dataset_builder import build_week5_examples, split_by_case, write_jsonl_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare privacy-cleared, case-group-aware Week 5 datasets.")
    parser.add_argument("--segments", default="annotations/document_segmentation/segmented_documents.jsonl")
    parser.add_argument("--entities", default="annotations/legal_ner/legal_entities.jsonl")
    parser.add_argument("--register", default="metadata/document_register.csv")
    parser.add_argument("--output", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = build_week5_examples(args.segments, args.register, args.entities)
    splits = split_by_case(examples, random_state=args.seed) if examples else {
        "train": [], "validation": [], "test": []
    }
    write_jsonl_splits(splits, args.output)
    print(f"Eligible examples: {len(examples)}")
    print(f"By task: {dict(Counter(row['task'] for row in examples))}")
    print(f"Split files written under: {Path(args.output).resolve()}")
    if not examples:
        print("No eligible examples. Approve annotation and privacy review, and populate case_group_id before training.")


if __name__ == "__main__":
    main()
