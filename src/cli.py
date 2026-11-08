import argparse
import json
from pathlib import Path
from src.extraction.pdf_extractor import extract_directory
from src.evaluation.corpus_report import build_report

def main():
    parser = argparse.ArgumentParser(description="TLDC corpus development CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract")
    extract.add_argument("--input", required=True)
    extract.add_argument("--output", required=True)
    report = sub.add_parser("report")
    report.add_argument("--input", required=True)
    report.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "extract":
        results = extract_directory(Path(args.input), Path(args.output))
        print(json.dumps({"processed": len(results), "results": results}, indent=2))
    else:
        print(f"Report written to {build_report(Path(args.input), Path(args.output))}")

if __name__ == "__main__":
    main()
