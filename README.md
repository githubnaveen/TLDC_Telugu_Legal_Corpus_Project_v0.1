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

## Week 5 baselines
Prepare reviewed model-ready partitions with `python -m src.training.prepare_datasets`. The builder excludes unresolved annotations, records without privacy clearance, missing case-group IDs, and NER entities without exact normalized-text offsets. It writes `data/train/tldc_train.jsonl`, `data/validation/tldc_validation.jsonl`, and `data/test/tldc_test.jsonl` using a deterministic case-group split.

Run the classical baselines after eligible data is available:

```powershell
python -m src.training.document_classification_baseline --model-output models/classical_ml/document_classifier.joblib
python -m src.training.legal_section_classifier
python -m src.training.legal_ner_baseline
```

The transformer experiment notebook is `notebooks/04_transformer_experiments.ipynb`. Transformer dependencies and model downloads are optional and intentionally not part of the default install.

## Week 6 explainability
Week 6 adds optional SHAP, LIME, token-occlusion and Integrated Gradients adapters, exact section/page evidence linking, reviewer audit records and explanation sensitivity metrics. Core dependencies remain unchanged; install optional packages with `pip install -r requirements-xai.txt`. The SHAP/LIME/transformer notebooks are opt-in and require a saved model plus approved reviewed data. The evidence schema is `schemas/evidence_attribution.schema.json`; no explanation values are prefilled as real results.

After preparing approved splits and saving a classifier, generate explanations from the held-out test partition with `python -m src.explainability.prototype --model-version tldc_baseline_v1`. Review outcomes can be recorded with `src.explainability.review.record_human_review` and stored through the review CSV workflow.

## Week 7 knowledge graph
Install the optional graph stack with `pip install -r requirements-kg.txt`. Build the graph and registries with `python -m src.graph.graph_builder`, then run SHACL with `python -m src.verification.shacl_validator`. The builder excludes unapproved or privacy-unreviewed annotations; statutory references require page-matched source text, and multilingual Act aliases are merged only when marked `VERIFIED` in `metadata/legal_reference_aliases.csv`.

## Week 8 trustworthy RAG
Install sparse retrieval dependencies with `pip install -r requirements-rag.txt`. Prepare approved page-preserving chunks with `python -m src.rag.chunking`; optionally generate vectors from a reviewed local checkpoint using `python -m src.rag.embeddings --model path/to/local/model`. Evaluate BM25/vector/hybrid/graph-enhanced retrieval with `python -m src.evaluation.rag_evaluation`, and run a local abstention-first answer path with `python -m src.rag.legal_rag_generation --question "..."`. The default path has no LLM configured and does not call hosted services. Dense embeddings are optional (`requirements-rag-embeddings.txt`) and require a pre-downloaded local model. Populate `rag_question_evidence_dataset.jsonl` only with adjudicated queries and relevant chunk IDs; do not report metrics from empty templates.

Human-reviewed trust metrics can be generated with `python -m src.evaluation.rag_trust_evaluation` after entering reviewed rows in `evaluation/human_review.csv` using the columns in `evaluation/human_review_template.csv`. Exact citation checks validate document/page/section identifiers and quote inclusion, not semantic entailment; all generated answers remain subject to human review.

## Week 9 trustworthiness evaluation
Run descriptive subgroup counts with `python -m src.evaluation.fairness_dataset_audit`; add held-out records with `label` and `prediction` in `data/test/group_predictions.jsonl` for group-wise metrics. Prepare expert-adjudicated, legally valid pairs in `counterfactual_fairness_dataset.jsonl` and run `python -m src.evaluation.counterfactual_evaluation`. Calibration requires separate labeled `data/calibration_predictions.jsonl` and `data/test_calibration_predictions.jsonl` files; run `python -m src.evaluation.uncertainty_calibration`. Risk/coverage uses labeled `data/selective_predictions.jsonl` with `correct` and `confidence` fields; run `python -m src.evaluation.selective_prediction`. Explanation summaries use `python -m src.evaluation.explanation_trust_evaluation`. Record reviewer decisions under the controlled labels in `evaluation/human_review_template.csv`, then run `python -m src.oversight.review_audit`; see `reports/human_oversight_protocol.md`.

## Week 10 adversarial resilience
Run only in this local isolated repository; do not test against live court portals or production systems. `python -m src.evaluation.prompt_injection_evaluation` evaluates the synthetic detector fixtures; `python -m src.security.corpus_integrity_validator` compares registered hashes without modifying source files; `python -m src.evaluation.data_poisoning_evaluation`, `python -m src.security.unicode_robustness_report`, `python -m src.evaluation.retrieval_attack_evaluation`, and `python -m src.security.graph_validation` run the controlled audit paths. `python -m src.evaluation.adversarial_robustness` summarizes only measured checks. Static detection rates do not equal end-to-end attack success, and no security control is a guarantee.
## Not yet implemented
Live document acquisition and permissions decisions, Telugu OCR execution/benchmarking, automatic anonymisation, legal expert adjudication, trained/evaluated NLP or RAG models, authoritative statutory-source verification, and completed PhD experiments. Weeks 5–8 contain implementation scaffolds, but current review-cleared datasets are empty and no model performance results are claimed.
