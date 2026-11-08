from src.preprocessing.text_cleaner import normalise_telugu_text

def test_normalisation_preserves_telugu():
    result = normalise_telugu_text("తెలుగు  పాఠ్యం\\n\\n\\nపరీక్ష")
    assert "తెలుగు" in result
    assert "\\n\\n\\n" not in result
