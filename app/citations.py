from __future__ import annotations

import re

CITATION = re.compile(r"\[(\d{1,2})\]")


def cited_indices(text: str, source_count: int) -> list[int]:
    found: list[int] = []
    for match in CITATION.finditer(text):
        index = int(match.group(1))
        if 1 <= index <= source_count and index not in found:
            found.append(index)
    return found


def strip_invalid(text: str, source_count: int) -> str:
    """Drop a bracketed number that does not map to a retrieved source.

    A model that invents [7] with six sources produces a dead chip, so the mark
    is removed rather than rendered. Real citations are left untouched.
    """

    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        return match.group(0) if 1 <= index <= source_count else ""

    return CITATION.sub(replace, text)


def ordered_sources(sources: list[dict], cited: list[int]) -> list[dict]:
    if not cited:
        return sources
    return [source for source in sources if source.get("index") in cited] + [
        source for source in sources if source.get("index") not in cited
    ]
