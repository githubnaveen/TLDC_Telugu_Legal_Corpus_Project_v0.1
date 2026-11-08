import re
from typing import List, Dict, Any

ENTITY_PATTERNS = [
    ("COURT", re.compile(r"(?:హైదరాబాద్|తిరుపతి|విశాఖ|హైకోర్ట్|సుప్రీంకోర్ట్|కోర్ట్|జిల్లా న్యాయస్థానం|district court|high court)", re.IGNORECASE)),
    ("CASE_NUMBER", re.compile(r"(?:CRLP|Crlp|CMA|C.R.P|CMA|BAIL|A.S|S.C|CASE|కేసు సంఖ్య)\s*[:#-]?\s*\d+[A-Za-z0-9/-]*", re.IGNORECASE)),
    ("LEGAL_ACT", re.compile(r"(?:IPC|CrPC|Indian Penal Code|Code of Criminal Procedure|Evidence Act|IPC\s*సెక్షన్|క్రిమినల్ ప్రొసీడర్ కోడ్|సెక్షన్)", re.IGNORECASE)),
    ("SECTION", re.compile(r"(?:Section|సెక్షన్)\s*[:#-]?\s*\d+[A-Za-z0-9.]*", re.IGNORECASE)),
    ("DATE", re.compile(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}")),
]


def extract_legal_entities(text: str) -> List[Dict[str, Any]]:
    sample = text or ""
    entities: List[Dict[str, Any]] = []
    for label, pattern in ENTITY_PATTERNS:
        for match in pattern.finditer(sample):
            entities.append({
                "label": label,
                "text": match.group(0).strip(),
                "start": match.start(),
                "end": match.end(),
            })
    return entities
