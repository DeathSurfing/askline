from __future__ import annotations

from datetime import datetime, timezone

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from app.graph import build_graph, system_prompt
from app.schemas import SearchOutcome, Source
from app.search import SearchEngine


class StubEngine:
    def __init__(self, results: list[Source] | None = None):
        self.results = results or [
            Source(1, "LangGraph 1.0", "https://example.com/langgraph", "LangGraph 1.0 shipped.", "bing")
        ]
        self.queries: list[str] = []

    async def search(self, query: str, count: int = 6) -> SearchOutcome:
        self.queries.append(query)
        return SearchOutcome(query=query, results=self.results, cached=False, backend="bing")


def tool_call(call_id: str, query: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "web_search", "args": {"query": query}, "id": call_id, "type": "tool_call"}],
    )


def initial_state(question: str = "When did LangGraph 1.0 ship?") -> dict:
    return {
        "messages": [HumanMessage(content=question)],
        "system": "You are a test agent.",
        "sources": [],
        "rounds": 0,
        "handled_tools": [],
    }


@pytest.mark.asyncio
async def test_graph_searches_then_answers_with_indices():
    from tests.fakes import ScriptedModel

    engine = StubEngine()
    model = ScriptedModel(
        script=[tool_call("c1", "langgraph 1.0 release"), AIMessage(content="It shipped in October 2025 [1].")]
    )
    graph = build_graph(model, engine)  # type: ignore[arg-type]
    result = await graph.ainvoke(initial_state())

    assert engine.queries == ["langgraph 1.0 release"]
    assert result["sources"][0]["url"] == "https://example.com/langgraph"
    assert result["sources"][0]["index"] == 1
    assert result["messages"][-1].content == "It shipped in October 2025 [1]."


@pytest.mark.asyncio
async def test_graph_caps_search_rounds_at_two():
    from tests.fakes import ScriptedModel

    engine = StubEngine()
    model = ScriptedModel(
        script=[tool_call("c1", "first"), tool_call("c2", "second"), tool_call("c3", "third")]
    )
    graph = build_graph(model, engine)  # type: ignore[arg-type]
    await graph.ainvoke(initial_state())

    assert engine.queries == ["first", "second"]


@pytest.mark.asyncio
async def test_graph_merges_and_dedupes_sources_across_rounds():
    from tests.fakes import ScriptedModel

    engine = StubEngine()
    model = ScriptedModel(script=[tool_call("c1", "a"), tool_call("c2", "b"), AIMessage(content="Done [2].")])
    graph = build_graph(model, engine)  # type: ignore[arg-type]
    result = await graph.ainvoke(initial_state())

    assert len(result["sources"]) == 1
    assert result["rounds"] == 2


def test_system_prompt_states_the_date_and_the_memory_rule():
    now = datetime(2026, 11, 10, 18, 30, tzinfo=timezone.utc)
    prompt = system_prompt("- 2026-11-03 09:12 (7 days ago) what is langgraph", now, "Asia/Kolkata")
    assert "10 November 2026" in prompt
    assert "Asia/Kolkata" in prompt
    assert "MEMORY" in prompt
    assert "square brackets" in prompt


def test_system_prompt_without_memory_has_no_memory_rule():
    now = datetime(2026, 11, 10, 18, 30, tzinfo=timezone.utc)
    prompt = system_prompt("", now, "local time")
    assert "MEMORY" not in prompt


def test_search_tool_is_the_only_tool_bound():
    from langchain_core.tools import StructuredTool

    from tests.fakes import ScriptedModel

    captured = {}

    class CapturingModel(ScriptedModel):
        def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN201
            captured["tools"] = tools
            return self

    build_graph(CapturingModel(script=[AIMessage(content="ok")]), StubEngine())  # type: ignore[arg-type]
    tools = captured["tools"]
    assert len(tools) == 1
    assert isinstance(tools[0], StructuredTool)
    assert tools[0].name == "web_search"
