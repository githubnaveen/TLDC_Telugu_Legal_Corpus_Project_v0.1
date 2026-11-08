from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import fitz

def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def extract_pdf(pdf_path: Path, output_dir: Path, min_chars: int = 30) -> dict:
    pdf_path, output_dir = Path(pdf_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text") or ""
            pages.append({
                "page_number": page_number,
                "raw_extracted_text": text,
                "character_count": len(text.strip()),
                "requires_ocr": len(text.strip()) < min_chars
            })
        result = {
            "filename": pdf_path.name,
            "sha256": sha256_file(pdf_path),
            "page_count": len(document),
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
            "pages": pages
        }
    output_path = output_dir / f"{pdf_path.stem}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["output_path"] = str(output_path)
    return result

def extract_directory(input_dir: Path, output_dir: Path) -> list:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    results = []
    for pdf_path in sorted(input_dir.rglob("*.pdf")):
        if output_dir == pdf_path or output_dir in pdf_path.parents:
            continue
        results.append(extract_pdf(pdf_path, output_dir))
    return results
