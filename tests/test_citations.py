from app.generation.citation_validator import FALLBACK, enforce_grounding


def test_invalid_citation_falls_back():
    answer, grounded, citations = enforce_grounding(
        "The answer is X [S99].",
        {"S1"},
    )
    assert answer == FALLBACK
    assert grounded is False
    assert citations == ["S99"]


def test_valid_citation_passes():
    answer, grounded, citations = enforce_grounding(
        "The answer is X [S1].",
        {"S1"},
    )
    assert answer == "The answer is X [S1]."
    assert grounded is True
    assert citations == ["S1"]
