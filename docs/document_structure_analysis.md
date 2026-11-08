# Document structure analysis

## Overview
This document records the structural anatomy of Telugu and Telugu-English legal orders in the TLDC pilot corpus. The goal is to reduce ambiguity in annotation and keep each legal document traceable to a verified source page.

## Common legal document structure

### 1. Header
Typical fields include:
- Court name and location
- Case number
- Filing or order date
- Jurisdiction
- Parties or roles

### 2. Facts
This section summarises the background and chronology of the case. It may include the allegations, events, procedural history and dispute initiation.

### 3. Petitioner arguments
This section states what the petitioner or applicant is requesting and why. Language may be legalistic, procedural, or both.

### 4. Respondent arguments
This section states the opposition or defence position. It often includes counter-arguments, statutory references and factual responses.

### 5. Legal provisions
This section identifies the governing legal sources: statutory sections, procedural rules and relevant case law.

### 6. Evidence
This section summarises witness statements, documentary evidence, seizure records, forensic facts and verification steps.

### 7. Judicial observation
The court explains legal reasoning, balancing of facts and statutory application. This section often contains the most interpretable explanation for a final decision.

### 8. Decision
This section records the final order: granted, rejected, partly granted or not determinable from the text alone.

## Annotation heuristics
- Preserve court and case references exactly as they appear in the document.
- Do not infer bail outcome from facts or arguments alone; only annotate if the decision text states it.
- When legal provisions are mixed-language, keep them in the original form and add a plain-language label when needed.
- Treat page boundaries as provenance anchors.
- Maintain anonymised role labels for individuals when personal names are present.

## Pilot guidance
For the Week 4 pilot, each source document should be segmented into the labels defined in the project schema: HEADER, FACTS, PETITIONER_ARGUMENT, RESPONDENT_ARGUMENT, LEGAL_PROVISIONS, EVIDENCE, JUDICIAL_OBSERVATION and DECISION.
