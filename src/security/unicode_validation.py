import unicodedata

ZERO_WIDTH_CODEPOINTS = {0x200B, 0x200C, 0x200D, 0xFEFF}
BIDI_CODEPOINTS = set(range(0x202A, 0x202F)) | set(range(0x2066, 0x206A)) | {0x200E, 0x200F}


def analyze_unicode_text(text):
    source = str(text or "")
    normalized = unicodedata.normalize("NFC", source)
    zero_width = [index for index, char in enumerate(source) if ord(char) in ZERO_WIDTH_CODEPOINTS]
    bidi = [index for index, char in enumerate(source) if ord(char) in BIDI_CODEPOINTS]
    suspicious_controls = [
        index for index, char in enumerate(source)
        if unicodedata.category(char) == "Cc" and char not in "\r\n\t"
    ]
    return {
        "source_text": source,
        "normalized_text": normalized,
        "normalization_changed": normalized != source,
        "invisible_format_characters": len(zero_width),
        "invisible_positions": zero_width,
        "bidirectional_controls": len(bidi),
        "bidirectional_positions": bidi,
        "unexpected_control_characters": len(suspicious_controls),
        "control_positions": suspicious_controls,
        "requires_review": bool(zero_width or bidi or suspicious_controls),
        "policy": "NFC form is reported separately; source text is preserved and no characters are silently removed.",
    }
