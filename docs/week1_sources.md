# Week 1 implementation notes

## Verified status (2026-11-08)

The repository scaffold and local validation are complete:
- Git repository created and pushed to GitHub.
- Local CLI runs successfully.
- Local test suite passes: `4 passed in 0.08s`.

## Official source inventory

### Telangana
- High Court for the State of Telangana: https://tshc.gov.in/
- High Court case status portal: https://csis.tshc.gov.in/
- High Court cause list portal: https://causelist.tshc.gov.in/
- District judiciary landing area on the official portal: https://tshc.gov.in/showChildDocTypes?id=17

These links were confirmed to be reachable from this environment and are suitable as the first official Telangana sources for later sampling and ethics review.

### Andhra Pradesh
Candidate sources that warrant verification before collection:
- https://hc.ap.nic.in/
- https://hc.ap.gov.in/
- https://e-courts.ap.gov.in/

These domains did not resolve successfully in the current environment, so they must be validated by the research team before any sample collection or processing begins.

## Remaining Week 1 checklist

- [x] Create the TLDC Git repository and folder structure.
- [x] Identify official Telangana High Court and district court document sources.
- [ ] Identify official Andhra Pradesh High Court and district court document sources.
- [ ] Review source access terms and institutional ethics requirements.
- [ ] Collect a small, permitted sample from both states.
- [ ] Record document language, court, district and document type.
- [ ] Run PDF text extraction on the sample.
- [ ] Identify scanned PDFs requiring Telugu OCR.

## Recommended next step

Before any document collection, confirm the AP official judicial domain and document access terms from the court site itself. Only then should the first small sample be downloaded and added to `data/raw/` under the permitted state folders.
