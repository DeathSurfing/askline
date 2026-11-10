from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from app.memory import Store
from app.schemas import SearchOutcome, Source
from app.search import SearchEngine, dedupe, parse_bing, parse_ddg, parse_searxng, relevant

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_bing_extracts_title_url_and_snippet():
    results = parse_bing((FIXTURES / "bing.html").read_text())
    assert 1 <= len(results) <= 3
    first = results[0]
    assert first.title
    assert first.url.startswith("http")
    assert "bing.com" not in first.url
    assert first.backend == "bing"


def test_parse_ddg_decodes_redirect_links():
    markup = """
    <div class="result">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage">Example</a>
      <a class="result__snippet">A snippet</a>
    </div>
    """
    results = parse_ddg(markup)
    assert results[0].url == "https://example.com/page"
    assert results[0].snippet == "A snippet"


def test_dedupe_collapses_same_page_and_numbers_results():
    results = [
        Source(0, "One", "https://example.com/a", "", "bing"),
        Source(0, "One again", "https://example.com/a/", "", "bing"),
        Source(0, "Two", "https://other.example/b", "", "bing"),
    ]
    unique = dedupe(results, limit=10)
    assert [source.index for source in unique] == [1, 2]
    assert [source.url for source in unique] == ["https://example.com/a", "https://other.example/b"]


def test_relevant_drops_results_that_share_no_distinctive_word():
    results = [
        Source(0, "Hindustan Times", "https://news.example/latest", "Breaking news", "bing"),
        Source(0, "LangGraph docs", "https://docs.example/langgraph", "Overview", "bing"),
    ]
    kept = relevant(results, "latest LangGraph release")
    assert [source.url for source in kept] == ["https://docs.example/langgraph"]
    assert relevant(results, "latest news") == results


def test_parse_searxng_reads_json_and_skips_bad_urls():
    payload = json.dumps(
        {
            "results": [
                {"title": "LangGraph docs", "url": "https://docs.example/langgraph", "content": "Overview"},
                {"title": "empty", "url": "", "content": "x"},
                {"title": "script", "url": "javascript:alert(1)", "content": "x"},
            ]
        }
    )
    results = parse_searxng(payload)
    assert len(results) == 1
    assert results[0].url == "https://docs.example/langgraph"
    assert results[0].snippet == "Overview"
    assert results[0].backend == "searxng"
    assert parse_searxng("not json") == []


@pytest.mark.asyncio
async def test_searxng_is_tried_first_and_cached(tmp_path):
    store = Store(tmp_path / "t.db")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search"
        assert request.url.params["format"] == "json"
        return httpx.Response(
            200,
            json={"results": [{"title": "LangGraph", "url": "https://example.com/lg", "content": "release notes"}]},
        )

    engine = SearchEngine(store, transport=httpx.MockTransport(handler), url="http://search.local")
    first = await engine.search("langgraph release")
    assert first.backend == "searxng"
    assert first.results[0].url == "https://example.com/lg"
    assert (await engine.search("langgraph release")).cached is True


@pytest.mark.asyncio
async def test_searxng_failure_falls_back_to_bing(tmp_path):
    store = Store(tmp_path / "t.db")
    bing = (FIXTURES / "bing.html").read_text()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "search.local":
            return httpx.Response(500, text="down")
        return httpx.Response(200, text=bing)

    engine = SearchEngine(store, transport=httpx.MockTransport(handler), url="http://search.local")
    outcome = await engine.search("langgraph release")
    assert outcome.backend == "bing"
    assert outcome.results


@pytest.mark.asyncio
async def test_search_falls_through_to_the_second_backend_and_caches(tmp_path):
    store = Store(tmp_path / "t.db")
    bing = (FIXTURES / "bing.html").read_text()
    ddg = (FIXTURES / "ddg.html").read_text()

    def handler(request: httpx.Request) -> httpx.Response:
        if "bing.com" in request.url.host:
            return httpx.Response(503, text="blocked")
        return httpx.Response(200, text=ddg)

    engine = SearchEngine(store, transport=httpx.MockTransport(handler))
    first = await engine.search("example result")
    assert first.backend == "duckduckgo"
    assert not first.cached
    assert first.results
    assert first.results[0].url == "https://example.com/first"

    second = await engine.search("example result")
    assert second.cached is True
    assert len(second.results) == len(first.results)


@pytest.mark.asyncio
async def test_search_reports_every_failure_instead_of_raising(tmp_path):
    store = Store(tmp_path / "t.db")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="slow down")

    engine = SearchEngine(store, transport=httpx.MockTransport(handler))
    outcome = await engine.search("anything")
    assert isinstance(outcome, SearchOutcome)
    assert outcome.results == []
    assert "bing" in outcome.error and "duckduckgo" in outcome.error


@pytest.mark.asyncio
async def test_empty_query_short_circuits(tmp_path):
    engine = SearchEngine(Store(tmp_path / "t.db"))
    outcome = await engine.search("   ")
    assert outcome.error == "empty query"
