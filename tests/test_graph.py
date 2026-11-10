from __future__ import annotations

from datetime import datetime, timezone

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph import (
    build_graph,
    parse_markup_tool_call,
    strip_tool_markup,
    system_prompt,
)
from app.schemas import SearchOutcome, Source


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
async def test_graph_forces_an_answer_when_tool_calls_exhaust_the_rounds():
    from tests.fakes import ScriptedModel

    engine = StubEngine()
    model = ScriptedModel(
        script=[tool_call("c1", "first"), tool_call("c2", "second"), tool_call("c3", "third"), AIMessage(content="Answer [1].")]
    )
    graph = build_graph(model, engine)  # type: ignore[arg-type]
    result = await graph.ainvoke(initial_state())

    assert engine.queries == ["first", "second"]
    assert result["messages"][-1].content == "Answer [1]."


@pytest.mark.asyncio
async def test_graph_merges_and_dedupes_sources_across_rounds():
    from tests.fakes import ScriptedModel

    engine = StubEngine()
    model = ScriptedModel(script=[tool_call("c1", "a"), tool_call("c2", "b"), AIMessage(content="Done [2].")])
    graph = build_graph(model, engine)  # type: ignore[arg-type]
    result = await graph.ainvoke(initial_state())

    assert len(result["sources"]) == 1
    assert result["rounds"] == 2


def markup_call(query: str) -> AIMessage:
    return AIMessage(
        content=(
            "<tool_call> <function=web_search> <parameter=query> "
            f"{query} </parameter> </function> </tool_call>"
        )
    )


@pytest.mark.asyncio
async def test_graph_runs_a_tool_call_written_as_text():
    from tests.fakes import ScriptedModel

    engine = StubEngine()
    model = ScriptedModel(
        script=[markup_call("langgraph 1.0 release"), AIMessage(content="It shipped in October 2025 [1].")]
    )
    graph = build_graph(model, engine)  # type: ignore[arg-type]
    result = await graph.ainvoke(initial_state())

    assert engine.queries == ["langgraph 1.0 release"]
    assert result["messages"][-1].content == "It shipped in October 2025 [1]."


@pytest.mark.asyncio
async def test_graph_forces_an_answer_when_markup_follows_a_search():
    from tests.fakes import ScriptedModel

    engine = StubEngine()
    model = ScriptedModel(
        script=[
            tool_call("c1", "first"),
            markup_call("again"),
            AIMessage(content="Final answer [1]."),
        ]
    )
    graph = build_graph(model, engine)  # type: ignore[arg-type]
    result = await graph.ainvoke(initial_state())

    assert engine.queries == ["first"]
    assert result["messages"][-1].content == "Final answer [1]."


@pytest.mark.asyncio
async def test_graph_falls_back_to_the_results_when_markup_persists():
    from tests.fakes import ScriptedModel

    engine = StubEngine()
    model = ScriptedModel(script=[tool_call("c1", "first"), markup_call("again"), markup_call("again")])
    graph = build_graph(model, engine)  # type: ignore[arg-type]
    result = await graph.ainvoke(initial_state())

    assert "Here is what the search found" in result["messages"][-1].content
    assert "LangGraph 1.0 shipped." in result["messages"][-1].content


@pytest.mark.asyncio
async def test_graph_uses_the_closer_model_at_the_cap():
    from tests.fakes import ScriptedModel

    engine = StubEngine()
    model = ScriptedModel(script=[tool_call("c1", "first"), tool_call("c2", "second")])
    closer = ScriptedModel(script=[AIMessage(content="Closed from results [1].")])
    graph = build_graph(model, engine, closer=closer)  # type: ignore[arg-type]
    result = await graph.ainvoke(initial_state())

    assert result["messages"][-1].content == "Closed from results [1]."
    assert closer.calls


@pytest.mark.asyncio
async def test_graph_falls_back_to_results_when_the_cap_model_is_empty():
    from tests.fakes import ScriptedModel

    engine = StubEngine()
    model = ScriptedModel(script=[tool_call("c1", "first"), tool_call("c2", "second"), AIMessage(content="")])
    graph = build_graph(model, engine)  # type: ignore[arg-type]
    result = await graph.ainvoke(initial_state())

    assert "Here is what the search found" in result["messages"][-1].content


def test_markup_parsing_and_stripping():
    text = str(markup_call("langgraph 1.0 release").content)
    call = parse_markup_tool_call(text)

    assert call is not None
    assert call["name"] == "web_search"
    assert call["args"] == {"query": "langgraph 1.0 release"}
    assert strip_tool_markup(text) == ""
    assert parse_markup_tool_call("plain answer") is None


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
