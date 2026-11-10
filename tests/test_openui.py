from __future__ import annotations

import json

import pytest

from app.config import ChatConfig
from app.openui import OpenUIError, build_lang, generate_ui, normalize, parse_json, pick_layout
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
    assert lang.startswith("root = Card([header, answer, body, followups], sources)")
    assert 'body = ListBlock([i1, i2, i3], "number")' in lang
    assert 'i1 = ListItem("Scope", "Row changes for selected tables.")' in lang
    assert 'sources = [{url: "https://example.com/1"' in lang
    assert lang.count("\n") >= 9


def test_build_lang_composes_a_table_for_comparisons():
    lang = build_lang(normalize(DATA, "comparison", [source(1)]), "comparison", [source(1)])
    assert "body = Table([c1, c2, c3])" in lang
    assert 'c1 = Col("Criterion", ["Scope", "Version", "Cost"])' in lang
    assert 'c2 = Col("Logical", ["Per table", "Tolerant", "Selective"])' in lang


@pytest.mark.asyncio
async def test_generate_ui_returns_a_program_and_sources(monkeypatch):
    from app import openui

    async def fake_complete(cfg, messages, timeout=90.0):
        return reply(DATA)

    monkeypatch.setattr(openui, "complete", fake_complete)
    result = await generate_ui(None, FakeEngine([source(1), source(2)]), CONFIG, "what changed?")
    assert result["lang"].startswith("root = Card(")
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
    assert result["lang"].startswith("root = Card(")
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
