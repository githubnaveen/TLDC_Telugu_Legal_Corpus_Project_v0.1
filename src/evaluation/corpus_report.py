from pathlib import Path
import json
import pandas as pd

def build_report(input_dir: Path, output_dir: Path) -> Path:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(input_dir.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        pages = payload.get("pages", [])
        records.append({
            "filename": payload.get("filename", path.name),
            "sha256": payload.get("sha256"),
            "page_count": payload.get("page_count", len(pages)),
            "pages_requiring_ocr": sum(bool(p.get("requires_ocr")) for p in pages),
            "pages_with_text": sum(bool((p.get("raw_extracted_text") or "").strip()) for p in pages),
            "total_extracted_characters": sum(p.get("character_count", 0) for p in pages)
        })
    df = pd.DataFrame(records)
    df.to_csv(output_dir / "extraction_summary.csv", index=False)
    summary = {
        "documents_processed": len(records),
        "total_pages": int(df["page_count"].sum()) if records else 0,
        "pages_requiring_ocr": int(df["pages_requiring_ocr"].sum()) if records else 0,
        "pages_with_extracted_text": int(df["pages_with_text"].sum()) if records else 0,
        "total_extracted_characters": int(df["total_extracted_characters"].sum()) if records else 0,
        "note": "Extraction statistics only; not an OCR accuracy or legal-quality assessment."
    }
    output = output_dir / "corpus_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output
