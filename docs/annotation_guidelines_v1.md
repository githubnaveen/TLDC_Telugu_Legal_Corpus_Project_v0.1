# TLDC Annotation Guidelines v1

## 1. Scope
This guideline defines how to annotate Telugu and Telugu-English legal documents for the TLDC Week 4 pilot. The objective is to produce consistent, traceable annotations while preserving legal meaning and privacy.

## 2. Section labels
- HEADER: court name, jurisdiction, case number and document metadata
- FACTS: factual timeline, allegations and procedural background
- PETITIONER_ARGUMENT: applicant or petitioner claims
- RESPONDENT_ARGUMENT: defence or opposition narrative
- LEGAL_PROVISIONS: acts, sections and legal sources
- EVIDENCE: witness, forensic or documentary evidence
- JUDICIAL_OBSERVATION: court reasoning and analysis
- DECISION: final outcome or order

## 3. Legal entity labels
- COURT
- CASE_NUMBER
- DATE
- LEGAL_ACT
- SECTION
- PETITIONER_ROLE
- RESPONDENT_ROLE
- JUDGE_ROLE
- WITNESS_ROLE
- EVIDENCE
- LOCATION
- CASE_TYPE

## 4. Annotation rules
- Keep original language in the source text when possible.
- Do not infer outcomes not explicitly stated.
- Prefer role-based anonymisation over person names where possible.
- Preserve page numbers and source provenance.
- If the text is ambiguous, add an annotation_status or confidence flag.

## 5. Ambiguous and missing information
- UNKNOWN: text does not make the decision explicit
- AMBIGUOUS: conflicting or incomplete information within the same page
- PENDING_REVIEW: requires legal or curator review

## 6. Privacy
- Remove direct personal identifiers from release copies where possible.
- Store original documents separately from processed annotations.
- Keep a record of every anonymisation and review decision.

## 7. Quality checklist
- Section boundaries are clear.
- Entities have the correct label.
- Decision label is supported by the text.
- No unsupported bail outcome is inferred.
- Annotation passes schema validation.
