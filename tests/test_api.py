from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app import agent as agent_module
from app.main import app, resolve_zone
from app.memory import format_time
from app.schemas import SearchOutcome, Source


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    with TestClient(app) as test_client:
        yield test_client


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for frame in body.split("\n\n"):
        if not frame.strip():
            continue
        name = ""
        data = ""
        for line in frame.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        if name and data:
            events.append((name, json.loads(data)))
    return events


def scripted_factory(scripts: list[list[AIMessage]]):
    from tests.fakes import ScriptedModel

    def build(_cfg):
        return ScriptedModel(script=scripts.pop(0))

    return build


def fake_search(results: list[Source]):
    async def search(query: str, count: int = 6) -> SearchOutcome:
        return SearchOutcome(query=query, results=results, cached=False, backend="bing")

    return search


FIRST_ANSWER = "LangGraph 1.0 shipped in October 2025 [1]."


def test_chat_streams_sources_and_persists_the_turn(client, monkeypatch):
    client.app.state.engine.search = fake_search(
        [Source(1, "LangGraph", "https://example.com/langgraph", "Release notes", "bing")]
    )
    monkeypatch.setattr(
        agent_module,
        "build_chat_model",
        scripted_factory(
            [
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "web_search", "args": {"query": "langgraph 1.0"}, "id": "c1", "type": "tool_call"}
                        ],
                    ),
                    AIMessage(content=FIRST_ANSWER),
                ]
            ]
        ),
    )

    with client.stream("POST", "/api/chat", json={"message": "When did LangGraph 1.0 ship?"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = parse_sse("".join(response.iter_text()))

    names = [name for name, _ in events]
    assert names[0] == "meta"
    assert "search_start" in names
    assert "search" in names
    assert "done" in names

    meta = dict(events)["meta"]
    assert meta["model"] == "test-model"
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", meta["created_at"])

    search = dict(events)["search"]
    assert search["results"][0]["url"] == "https://example.com/langgraph"

    done = dict(events)["done"]
    assert done["text"] == FIRST_ANSWER
    assert done["cited"] == [1]
    assert done["sources"][0]["index"] == 1
    streamed = "".join(payload["delta"] for name, payload in events if name == "token")
    assert streamed == FIRST_ANSWER

    thread_id = meta["thread_id"]
    stored = client.get(f"/api/threads/{thread_id}").json()
    assert [turn["role"] for turn in stored["turns"]] == ["user", "assistant"]
    assert stored["turns"][1]["sources"][0]["url"] == "https://example.com/langgraph"
    assert stored["thread"]["title"] == "When did LangGraph 1.0 ship?"


def test_second_question_receives_the_first_one_as_memory(client, monkeypatch):
    client.app.state.engine.search = fake_search([])
    monkeypatch.setattr(
        agent_module,
        "build_chat_model",
        scripted_factory(
            [
                [AIMessage(content=FIRST_ANSWER)],
                [AIMessage(content="You asked that a moment ago.")],
            ]
        ),
    )
    with client.stream("POST", "/api/chat", json={"message": "When did LangGraph 1.0 ship?"}) as response:
        thread_id = dict(parse_sse("".join(response.iter_text())))["meta"]["thread_id"]

    with client.stream(
        "POST",
        "/api/chat",
        json={"thread_id": thread_id, "message": "What did I ask you just now?", "timezone": "Asia/Kolkata"},
    ) as response:
        events = parse_sse("".join(response.iter_text()))

    memory = dict(events).get("memory")
    assert memory is not None
    assert memory["items"][0]["text"] == "When did LangGraph 1.0 ship?"
    assert memory["items"][0]["local_time"] == format_time(memory["items"][0]["created_at"], "Asia/Kolkata")

    timeline = client.get("/api/timeline").json()
    assert timeline["days"][0]["asks"][0]["text"] == "What did I ask you just now?"
    assert timeline["days"][0]["asks"][0]["id"]


def test_legacy_timezone_alias_is_resolved():
    assert resolve_zone("Asia/Calcutta") == "Asia/Calcutta"
    assert resolve_zone("Asia/Kolkata") == "Asia/Kolkata"
    assert resolve_zone("Nowhere/City") is None


def test_chat_rejects_an_unknown_timezone(client):
    assert client.post("/api/chat", json={"message": "hi", "timezone": "Mars/Olympus"}).status_code == 400


def test_chat_without_a_key_returns_a_clear_400(client, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.post("/api/chat", json={"message": "hello"})
    assert response.status_code == 400
    assert "API key" in response.json()["detail"]


def test_settings_round_trip_masks_the_key(client):
    response = client.post(
        "/api/settings",
        json={"provider": "anthropic", "base_url": "https://api.anthropic.com", "model": "claude-sonnet-4-5",
              "api_key": "sk-ant-secret", "temperature": 0.4, "max_tokens": 600},
    )
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "model": "claude-sonnet-4-5",
        "temperature": 0.4,
        "max_tokens": 600,
        "has_api_key": True,
    }
    assert "sk-ant-secret" not in response.text

    kept = client.post("/api/settings", json={"model": "claude-haiku-4-5", "api_key": ""}).json()
    assert kept["model"] == "claude-haiku-4-5"
    assert kept["has_api_key"] is True


def test_settings_reject_an_invalid_provider(client):
    assert client.post("/api/settings", json={"provider": "gemini"}).status_code == 400


def test_settings_test_uses_submitted_overrides(client, monkeypatch):
    seen = {}

    async def fake_test(cfg):
        seen.update(cfg.public())
        seen["api_key"] = cfg.api_key
        return {"ok": True, "latency_ms": 12}

    monkeypatch.setattr("app.main.test_connection", fake_test)
    response = client.post(
        "/api/settings/test",
        json={"provider": "anthropic", "base_url": "https://example.test", "model": "claude-x",
              "temperature": 0.9, "max_tokens": 100},
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True, "latency_ms": 12}
    assert seen["provider"] == "anthropic"
    assert seen["model"] == "claude-x"
    assert seen["temperature"] == 0.9
    assert seen["api_key"] == "test-key"


def test_missing_thread_is_a_404(client):
    assert client.post("/api/chat", json={"thread_id": "nope", "message": "hi"}).status_code == 404


def test_health_never_calls_the_model(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["model"] == "test-model"
    assert body["has_api_key"] is True
