import re

STATE_CODES = {"Telangana": "TS", "Andhra Pradesh": "AP"}

def make_document_id(state: str, district_code: str, year: int, sequence: int) -> str:
    if state not in STATE_CODES:
        raise ValueError(f"Unsupported state: {state}")
    if not re.fullmatch(r"[A-Z0-9]{2,8}", district_code.upper()):
        raise ValueError("district_code must be 2-8 alphanumeric characters")
    if not 1900 <= int(year) <= 2100 or int(sequence) < 1:
        raise ValueError("Invalid year or sequence")
    return f"TLDC-{STATE_CODES[state]}-{district_code.upper()}-{year}-{sequence:06d}"
