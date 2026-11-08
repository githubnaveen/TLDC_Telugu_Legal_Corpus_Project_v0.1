import re

TELUGU_RANGE = re.compile(r"[\u0C00-\u0C7F]")
ENGLISH_RANGE = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    """Classify a short text as Telugu, English or Mixed."""
    sample = (text or "").strip()
    if not sample:
        return "Unknown"

    telugu_chars = len(TELUGU_RANGE.findall(sample))
    english_chars = len(ENGLISH_RANGE.findall(sample))

    if telugu_chars and english_chars:
        return "Mixed"
    if telugu_chars:
        return "Telugu"
    if english_chars:
        return "English"
    return "Unknown"


def estimate_telugu_ratio(text: str) -> float:
    sample = text or ""
    telugu_chars = len(TELUGU_RANGE.findall(sample))
    total = len(sample)
    if total == 0:
        return 0.0
    return telugu_chars / total
