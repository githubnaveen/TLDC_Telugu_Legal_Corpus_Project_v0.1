from src.preprocessing.text_cleaner import normalise_telugu_text

def test_normalisation_preserves_telugu():
    result = normalise_telugu_text("తెలుగు  text\tvalue\u00a0ok\n   line\n\n\nపరీక్ష")
    assert "తెలుగు" in result
    assert result == "తెలుగు text value ok\nline\n\nపరీక్ష"
