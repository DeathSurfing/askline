from __future__ import annotations

import json

import pytest

from app.config import ChatConfig
from app.openui import OpenUIError, build_lang, extract_series, generate_ui, normalize, parse_json, pick_layout
from app.schemas import SearchOutcome, Source

CONFIG = ChatConfig(
    provider="openai",
    base_url="http://127.0.0.1:9/v1",
    model="test-model",
    api_key="test-key",
)

DATA = {
    "title": "Python 3.15",
    "subtitle": "The latest stable release",
    "answer": "Python 3.15.2 is the latest stable release [1]. It adds ordered dictionaries and a thousands separator [2]. The release also improves the C API and typing [1].",
    "leftTitle": "Logical",
    "rightTitle": "Physical",
    "items": [
        {"label": "Scope", "text": "Row changes for selected tables.", "left": "Per table", "right": "Whole cluster"},
        {"label": "Version", "text": "Tolerant of version differences.", "left": "Tolerant", "right": "Identical"},
        {"label": "Cost", "text": "Lower bandwidth for a few tables.", "left": "Selective", "right": "Complete"},
    ],
    "followups": ["Which one fits a zero-downtime upgrade?", "How do I monitor lag?"],
}


class FakeEngine:
    def __init__(self, results: list[Source]) -> None:
        self.results = results

    async def search(self, query: str, count: int = 6) -> SearchOutcome:
        return SearchOutcome(query=query, results=self.results, cached=False, backend="searxng")


def source(index: int = 1) -> Source:
    return Source(index, "Python release", f"https://example.com/{index}", "Notes", "searxng")


def reply(data: dict) -> str:
    return json.dumps(data)


def test_parse_json_accepts_fences_and_leading_prose():
    assert parse_json('Here you go:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('{"a": 1} trailing') == {"a": 1}


def test_parse_json_rejects_text_without_an_object():
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_pick_layout_routes_comparisons():
    assert pick_layout("Compare logical and physical replication") == "comparison"
    assert pick_layout("Postgres vs MySQL") == "comparison"
    assert pick_layout("What changed in Python 3.15?") == "summary"


def test_normalize_drops_out_of_range_citations():
    data = normalize({**DATA, "answer": DATA["answer"] + " See [9]."}, "summary", [source(1), source(2)])
    assert "[9]" not in data["answer"]
    assert len(data["items"]) == 3


def test_normalize_rejects_equal_comparison_titles():
    with pytest.raises(ValueError, match="different option names"):
        normalize({**DATA, "leftTitle": "Replication", "rightTitle": "Replication"}, "comparison", [source(1)])


def test_normalize_rejects_too_few_items():
    with pytest.raises(ValueError, match="too few"):
        normalize({**DATA, "items": DATA["items"][:1]}, "summary", [source(1)])


def test_build_lang_composes_a_valid_summary_program():
    lang = build_lang(normalize(DATA, "summary", [source(1)]), "summary", [source(1)])
    assert "root = Card([header, answer, body, followups])" in lang
    assert "body = Table([c1, c2], rows)" in lang
    assert 'c1 = Col("Point")' in lang
    assert 'rows = [["Scope", "Row changes for selected tables."]' in lang
    assert lang.count("\n") >= 9


def test_build_lang_composes_a_table_for_comparisons():
    lang = build_lang(normalize(DATA, "comparison", [source(1)]), "comparison", [source(1)])
    assert "body = Table([c1, c2, c3], rows)" in lang
    assert 'c1 = Col("Criterion")' in lang
    assert 'c2 = Col("Logical")' in lang
    assert 'rows = [["Scope", "Per table", "Whole cluster"]' in lang


def test_build_lang_adds_a_chart_when_the_data_has_one():
    data = normalize(
        {**DATA, "chart": {"kind": "bar", "title": "Patch releases", "labels": ["3.15.0", "3.15.1"], "values": [1, 2], "unit": ""}},
        "summary",
        [source(1)],
    )
    lang = build_lang(data, "summary", [source(1)])
    assert 'chart = BarChart(["3.15.0", "3.15.1"], [s1], "grouped", "", "")' in lang
    assert 's1 = Series("Patch releases", [1.0, 2.0])' in lang


def test_build_lang_adds_a_donut_chart():
    data = normalize(
        {**DATA, "chart": {"kind": "donut", "title": "Share", "labels": ["A", "B"], "values": [3, 7], "unit": "%"}},
        "summary",
        [source(1)],
    )
    lang = build_lang(data, "summary", [source(1)])
    assert "chart = PieChart([p1, p2], true)" in lang
    assert 'p1 = Slice("A", 3.0)' in lang


@pytest.mark.asyncio
async def test_generate_ui_returns_a_program_and_sources(monkeypatch):
    from app import openui

    async def fake_complete(cfg, messages, timeout=90.0):
        return reply(DATA)

    monkeypatch.setattr(openui, "complete", fake_complete)
    result = await generate_ui(None, FakeEngine([source(1), source(2)]), CONFIG, "what changed?")
    assert "root = Card(" in result["lang"]
    assert result["layout"] == "summary"
    assert len(result["sources"]) == 2
    assert result["elapsed_ms"] >= 0


@pytest.mark.asyncio
async def test_generate_ui_retries_with_feedback(monkeypatch):
    from app import openui

    calls: list[list] = []

    async def fake_complete(cfg, messages, timeout=90.0):
        calls.append(messages)
        return "not json" if len(calls) == 1 else reply(DATA)

    monkeypatch.setattr(openui, "complete", fake_complete)
    result = await generate_ui(None, FakeEngine([source()]), CONFIG, "what changed?")
    assert "root = Card(" in result["lang"]
    assert len(calls) == 2
    assert "not usable" in calls[1][-1].content


@pytest.mark.asyncio
async def test_generate_ui_fails_after_two_bad_replies(monkeypatch):
    from app import openui

    async def fake_complete(cfg, messages, timeout=90.0):
        return "not json"

    monkeypatch.setattr(openui, "complete", fake_complete)
    with pytest.raises(OpenUIError, match="usable interface"):
        await generate_ui(None, FakeEngine([source()]), CONFIG, "what changed?")


@pytest.mark.asyncio
async def test_generate_ui_reports_empty_search(monkeypatch):
    from app import openui

    async def fake_complete(cfg, messages, timeout=90.0):
        return reply(DATA)

    monkeypatch.setattr(openui, "complete", fake_complete)
    with pytest.raises(OpenUIError, match="No results came back"):
        await generate_ui(None, FakeEngine([]), CONFIG, "what changed?")


def test_openui_endpoint_returns_interface(client, monkeypatch):
    from app import main as main_module

    async def fake_generate(store, engine, cfg, question):
        return {
            "lang": "root = Card([], sources)",
            "layout": "summary",
            "sources": [{"url": "https://example.com/1", "title": "T", "sourceName": "example.com"}],
            "question": question,
            "elapsed_ms": 12,
        }

    monkeypatch.setattr(main_module, "generate_ui", fake_generate)
    response = client.post("/api/openui", json={"message": "what changed?"})
    assert response.status_code == 200
    body = response.json()
    assert body["lang"].startswith("root = Card(")
    assert body["model"] == "test-model"


def test_openui_page_serves_the_view(client):
    response = client.get("/openui")
    assert response.status_code == 200
    assert "OpenUI" in response.text


async def test_compose_ui_keeps_the_answer_verbatim(monkeypatch):
    from app.openui import compose_ui

    async def fake_complete(cfg, messages, timeout=90.0):
        return (
            '{"title":"T","subtitle":"S","answer":"rewritten",'
            '"items":[{"label":"A","text":"One."},{"label":"B","text":"Two."}],'
            '"followups":["Q1?","Q2?"]}'
        )

    monkeypatch.setattr("app.openui.complete", fake_complete)
    result = await compose_ui(
        ChatConfig(provider="openai", base_url="http://x/v1", model="m", api_key="k"),
        "Why?",
        "The real answer with [1].",
        [Source(1, "T", "https://example.com", "s", "searxng")],
    )
    assert "The real answer with [1]." in result["answer"]
    assert result["layout"] == "summary"


async def test_compose_ui_falls_back_without_a_model(monkeypatch):
    from app.openui import compose_ui

    async def boom(cfg, messages, timeout=90.0):
        raise RuntimeError("offline")

    monkeypatch.setattr("app.openui.complete", boom)
    result = await compose_ui(
        ChatConfig(provider="openai", base_url="http://x/v1", model="m", api_key="k"),
        "Why is America broke?",
        "A first sentence that is long enough. A second sentence that is also long enough.",
        [],
    )
    assert result["layout"] == "summary"
    assert "root = Card(" in result["lang"]
    assert result["answer"].startswith("A first sentence")


def test_extract_series_reads_year_and_percentage():
    text = "Turnout was 39.5% in 2002, 41.1% in 2006, and 65.3% in 2024."
    chart = extract_series(text)
    assert chart["labels"] == ["2002", "2006", "2024"]
    assert chart["values"] == [39.5, 41.1, 65.3]
    assert chart["unit"] == "%"


def test_extract_series_needs_three_points():
    assert extract_series("Turnout was 65.3% in 2024.") is None


def test_line_chart_uses_the_library_argument_count():
    from app.openui import build_lang

    data = {
        "title": "Turnout",
        "subtitle": "By year",
        "answer": "Turnout rose over the period [1].",
        "items": [{"label": "A", "text": "One."}, {"label": "B", "text": "Two."}],
        "followups": ["Q1?", "Q2?"],
        "chart": {"kind": "line", "title": "Turnout", "labels": ["2000", "2004", "2008"], "values": [50.1, 55.2, 58.3], "unit": "%"},
    }
    line = next(item for item in build_lang(data, "summary", []).splitlines() if item.startswith("chart = LineChart"))
    assert '"grouped"' not in line
    assert line.endswith('"%" )'.replace(" )", ")"))
