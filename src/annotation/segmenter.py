import re
from typing import List, Dict, Any

SECTION_KEYWORDS = {
    "HEADER": ["హైకోర్ట్", "కోర్ట్", "జిల్లా", "న్యాయస్థానం", "court", "high court", "district court"],
    "FACTS": ["వాస్తవాలు", "facts", "సంబంధిత వివరాలు", "case background"],
    "PETITIONER_ARGUMENT": ["పెటిషనర్", "petitioner's", "petitioner", "వాదనలు", "arguments"],
    "RESPONDENT_ARGUMENT": ["ప్రతివాది", "respondent", "ముందు వాదనలు", "counter"],
    "LEGAL_PROVISIONS": ["చట్టం", "law", "సెక్షన్", "section", "act", "legal provisions"],
    "EVIDENCE": ["సాక్షులు", "evidence", "పత్రాలు", "witness", "testimony"],
    "JUDICIAL_OBSERVATION": ["న్యాయమూర్తి", "judicial observation", "reasoning", "పరిశీలన", "ఆలోచన"],
    "DECISION": ["ఆర్డర్", "decision", "final order", "judgment", "మంజూరు", "reject"],
}


def detect_section_type(line: str) -> str:
    text = (line or "").strip().lower()
    if re.search(r"(కేసు\s+సంఖ్య|case\s+number|case\s+no|హైకోర్ట్|కోర్ట్|జిల్లా|న్యాయస్థానం|court)", text):
        return "HEADER"
    if re.search(r"(వాస్తవాలు|facts|case background|background)", text):
        return "FACTS"
    if re.search(r"(పెటిషనర్|petitioner|వాదనలు|arguments)", text):
        return "PETITIONER_ARGUMENT"
    if re.search(r"(ప్రతివాది|respondent|counter)", text):
        return "RESPONDENT_ARGUMENT"
    if re.search(r"(చట్టం|law|సెక్షన్|section|act|legal provisions)", text):
        return "LEGAL_PROVISIONS"
    if re.search(r"(సాక్షులు|evidence|పత్రాలు|witness|testimony)", text):
        return "EVIDENCE"
    if re.search(r"(న్యాయమూర్తి|judicial observation|reasoning|పరిశీలన|ఆలోచన)", text):
        return "JUDICIAL_OBSERVATION"
    if re.search(r"(ఆర్డర్|decision|final order|judgment|మంజూరు|reject)", text):
        return "DECISION"
    for label, keywords in SECTION_KEYWORDS.items():
        if any(keyword.lower() in text for keyword in keywords):
            return label
    return "FACTS"


def segment_document(document_text: str) -> List[Dict[str, Any]]:
    text = (document_text or "").strip()
    if not text:
        return []

    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n") if line.strip()]
    if not lines:
        return []

    sections: List[Dict[str, Any]] = []
    current_label = None
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_label, current_lines
        if current_label is None:
            return
        chunk = " ".join(current_lines).strip()
        if chunk:
            sections.append({
                "section_id": f"S{len(sections) + 1:03d}",
                "section_type": current_label,
                "text": chunk,
                "page_number": len(sections) + 1
            })
        current_label = None
        current_lines = []

    for line in lines:
        detected = detect_section_type(line)
        if current_label is None:
            current_label = detected
            current_lines = [line]
            continue

        if detected == current_label:
            current_lines.append(line)
            continue

        flush()
        current_label = detected
        current_lines = [line]

    flush()

    if not sections:
        sections.append({
            "section_id": "S001",
            "section_type": "HEADER",
            "text": text,
            "page_number": 1,
        })

    return sections
