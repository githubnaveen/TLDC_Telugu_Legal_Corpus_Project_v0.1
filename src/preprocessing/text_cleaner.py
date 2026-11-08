import re
import unicodedata

def normalise_telugu_text(text: str) -> str:
    # Conservative normalisation; no spelling correction or uncertain character replacement.
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
