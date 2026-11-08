# TLDC — Telugu Legal Documents Corpus
**Research project starter kit | v0.1 | Telangana + Andhra Pradesh**

A reproducible research scaffold for a provenance-aware, privacy-reviewed Telugu / Telugu-English legal-document corpus supporting later research on explainable, verifiable, privacy-preserving, adversarially resilient and controlled self-evolving cognitive AI.

> This repository contains no court documents. It does not scrape court websites or bypass access controls. Collect only documents whose access and research use have been reviewed and approved.

## Quick start
1. Create an environment: `python -m venv .venv`
2. Activate it (`.venv\\Scripts\\activate` on Windows; `source .venv/bin/activate` on macOS/Linux).
3. Install: `pip install -r requirements.txt`
4. Install Tesseract and Telugu language data (`tel`) separately if OCR is needed.
5. Place permitted PDFs under `data/raw/Telangana/` or `data/raw/Andhra_Pradesh/`.
6. Populate `metadata/document_register.csv` using the template.
7. Extract: `python -m src.cli extract --input data/raw --output data/extracted`
8. Report: `python -m src.cli report --input data/extracted --output reports`
9. Test: `pytest`

## Included
- Page-level PDF text extraction and SHA-256 checksums.
- OCR-needed page flag (not OCR execution).
- Conservative Unicode NFC normalisation.
- Document identifier helper.
- Annotation span validation.
- Privacy review queue (administrative only; no automatic anonymisation).
- Case-group-aware train/validation/test split helper.
- Corpus extraction statistics.
- Collection, annotation, ethics and PhD-alignment documentation.

## Research safeguards
- Keep original documents in restricted storage; never commit them to Git.
- Public visibility does not imply permission to bulk collect or redistribute.
- Preserve page numbers, source identifiers and text offsets.
- Do not infer legal outcomes; use UNKNOWN or AMBIGUOUS when needed.
- Do not call data anonymised without documented privacy review.
- Keep documents from the same case in one data split.
- Corpus updates do not automatically trigger model retraining.

## Not yet implemented
Official-source collection, permissions review, Telugu OCR execution/benchmarking, Telugu-English NER, automatic anonymisation, legal expert annotation, legal knowledge graph, model training and PhD experiments.
