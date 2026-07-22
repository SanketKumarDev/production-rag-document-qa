import re

FALLBACK = "I could not find sufficient information in the provided documents."


def validate_citations(answer: str, allowed_ids: set[str]) -> tuple[bool, list[str]]:
    cited_ids = re.findall(r"\[(S\d+)\]", answer)
    unique_ids = sorted(set(cited_ids))
    if not cited_ids:
        return False, []
    return all(x in allowed_ids for x in unique_ids), unique_ids


def enforce_grounding(
    answer: str,
    allowed_ids: set[str],
) -> tuple[str, bool, list[str]]:
    if FALLBACK.lower() in answer.lower():
        return FALLBACK, True, []

    valid, citations = validate_citations(answer, allowed_ids)
    if not valid:
        return FALLBACK, False, citations

    return answer, True, citations
