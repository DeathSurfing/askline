from __future__ import annotations

from app.citations import cited_indices, ordered_sources, strip_invalid


def test_cited_indices_dedupes_and_ignores_out_of_range():
    text = "A [1], B [2], C [1], D [9]."
    assert cited_indices(text, 2) == [1, 2]


def test_strip_invalid_removes_a_dead_number_and_keeps_the_text():
    assert strip_invalid("Fact [3] with three sources.", 3) == "Fact [3] with three sources."
    assert strip_invalid("Fact [4] with three sources.", 3) == "Fact  with three sources."


def test_ordered_sources_moves_cited_results_first():
    sources = [{"index": 1}, {"index": 2}, {"index": 3}]
    assert [item["index"] for item in ordered_sources(sources, [3])] == [3, 1, 2]
    assert ordered_sources(sources, []) == sources
