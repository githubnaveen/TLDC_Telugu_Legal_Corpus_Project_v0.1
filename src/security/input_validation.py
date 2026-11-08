import unicodedata


def validate_user_query(query, maximum_characters=4000):
    value = str(query or "")
    errors = []
    if not value.strip():
        errors.append("empty_query")
    if len(value) > maximum_characters:
        errors.append("query_too_long")
    controls = [char for char in value if unicodedata.category(char) == "Cc" and char not in "\r\n\t"]
    if controls:
        errors.append("unexpected_control_characters")
    return {"valid": not errors, "errors": errors, "source_preserved": True}
