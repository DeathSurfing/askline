from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from app import subagents
from app.config import ChatConfig
from app.schemas import SearchOutcome, Source
from app.subagents import MAX_DEPTH, MAX_NODES, parse_node, parse_plan, run_delegation

CONFIG = ChatConfig(
    provider="openai",
    base_url="http://127.0.0.1:9/v1",
    model="test-model",
    api_key="test-key",
)


class FakeModel:
    async def ainvoke(self, messages):
        prompt = messages[-1].content
        if "Answer the user's question from these subagent findings" in prompt:
            return AIMessage(content="Final answer from the tree [1].")
        return AIMessage(content='{"findings": ["A fact [1]."], "followups": []}')


class FakeEngine:
    def __init__(self, count: int = 1) -> None:
        self.count = count

    async def search(self, query: str, count: int = 6) -> SearchOutcome:
        return SearchOutcome(
            query=query,
            results=[Source(index, f"Result {index}", f"https://example.com/{index}", "Snippet", "searxng") for index in range(1, self.count + 1)],
            cached=False,
            backend="searxng",
        )


def test_parse_plan_accepts_roles_and_falls_back():
    plan = parse_plan('{"tasks": [{"role": "design", "brief": "Outline the answer"}, {"role": "unknown", "brief": "Check facts"}]}')
    assert plan == [
        {"role": "design", "brief": "Outline the answer"},
        {"role": "research", "brief": "Check facts"},
    ]


def test_parse_plan_rejects_empty():
    with pytest.raises(ValueError):
        parse_plan('{"tasks": []}')


def test_parse_node_keeps_citations_and_caps_followups():
    followup_json = ", ".join(f'{{"role": "research", "brief": "q{index}"}}' for index in range(5))
    findings, followups = parse_node(
        '{"findings": ["One [1].", "Two [2]."], "followups": [' + followup_json + "]}",
        allow_followups=True,
    )
    assert findings == ["One [1].", "Two [2]."]
    assert len(followups) == 3


@pytest.mark.asyncio
async def test_delegation_yields_a_tree_and_a_final_answer(monkeypatch):
    async def fake_ask(model, prompt, timeout=45.0):
        if "Plan a small task tree" in prompt:
            return '{"tasks": [{"role": "research", "brief": "Find the release date"}]}'
        return '{"findings": ["The release is 3.15.2 [1]."], "followups": []}'

    monkeypatch.setattr(subagents, "text_model", lambda cfg: FakeModel())
    monkeypatch.setattr(subagents, "ask_json", fake_ask)

    events = [item async for item in run_delegation(None, FakeEngine(2), CONFIG, "What is the latest Python?")]
    names = [name for name, _ in events]
    final = events[-1][1]

    assert names[0] == "task"
    assert "task_done" in names
    assert names[-1] == "final"
    assert final["text"] == "Final answer from the tree [1]."
    assert len(final["sources"]) == 2
    assert [task["role"] for task in final["trace"]] == ["research"]


@pytest.mark.asyncio
async def test_delegation_stops_at_the_depth_cap(monkeypatch):
    async def fake_ask(model, prompt, timeout=45.0):
        if "Plan a small task tree" in prompt:
            return '{"tasks": [{"role": "research", "brief": "root"}]}'
        return '{"findings": ["A fact [1]."], "followups": [{"role": "research", "brief": "deeper"}]}'

    monkeypatch.setattr(subagents, "text_model", lambda cfg: FakeModel())
    monkeypatch.setattr(subagents, "ask_json", fake_ask)

    events = [item async for item in run_delegation(None, FakeEngine(), CONFIG, "question")]
    trace = events[-1][1]["trace"]
    assert len(trace) <= MAX_NODES
    assert max(task["depth"] for task in trace) == MAX_DEPTH
    assert not any(task["depth"] > MAX_DEPTH for task in trace)
