from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from .search import SearchEngine

MAX_SEARCH_ROUNDS = 2

_MARKUP_CALL = re.compile(
    r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>\s*</tool_call>", re.DOTALL
)
_MARKUP_PARAM = re.compile(r"<parameter=([\w.-]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def has_tool_markup(text: str | None) -> bool:
    return bool(text and "<tool_call>" in text)


def parse_markup_tool_call(text: str | None) -> dict | None:
    """Read the tool call a small model wrote as text instead of a structured call."""
    match = _MARKUP_CALL.search(text or "")
    if not match or match.group(1).strip() != "web_search":
        return None
    params = {key.strip(): value.strip() for key, value in _MARKUP_PARAM.findall(match.group(2))}
    query = params.get("query", "").strip()
    if not query:
        return None
    digest = hashlib.md5(query.encode()).hexdigest()[:8]
    return {"name": "web_search", "args": {"query": query}, "id": f"call_{digest}", "type": "tool_call"}


def strip_tool_markup(text: str | None) -> str:
    return _MARKUP_CALL.sub("", text or "").strip()


def closing_prompt(question: str, sources: list[dict]) -> str:
    lines = [f"Question: {question}", "", "Search results:"]
    for source in sources[:5]:
        snippet = str(source.get("snippet") or "").strip()[:300]
        title = source.get("title") or source.get("url") or "source"
        lines.append(f"[{source.get('index', '')}] {title}: {snippet}")
    lines += ["", "Answer in three to six sentences using only these results. Cite claims as [n]."]
    return "\n".join(lines)


def results_summary(sources: list[dict]) -> str:
    if not sources:
        return "I could not reach a clear answer. Try asking again."
    lines = ["Here is what the search found."]
    for source in sources[:3]:
        snippet = str(source.get("snippet") or "").strip()
        title = source.get("title") or source.get("url") or "source"
        lines.append(f"[{source.get('index', '')}] {title}" + (f": {snippet}" if snippet else ""))
    return "\n".join(lines)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    system: str
    sources: list[dict]
    rounds: int
    handled_tools: list[str]


class SearchArgs(BaseModel):
    query: str = Field(
        description="A short, specific web search query built from the distinctive words: names, "
        "products, versions. Drop filler such as latest, news, best, and question words."
    )


def system_prompt(memory_block: str, now: datetime, zone_label: str) -> str:
    parts = [
        "You are Askline, a fast, careful research assistant with exactly one tool: web_search.",
        f"The current local date and time is {now.strftime('%A, %d %B %Y, %H:%M')} ({zone_label}).",
        "",
        "How to work:",
        "1. Lead with the answer in the first sentence. Then give the few facts that support it.",
        "2. Search when the question depends on current events, prices, releases, schedules, people, "
        "or any fact you are not certain about. A version number, release date, price, or schedule is "
        "never something you know from memory, so always search for it. Search once; search again only "
        "if the first results miss the question or contradict each other. Write the query with the "
        "distinctive words only: names, products, versions. Drop filler like latest, news, and best, "
        "and drop question words.",
        "3. Use what the results say, not your memory of the topic, whenever you searched. "
        "Cite each claim taken from a result with its number in square brackets, like [1] or [2][3]. "
        "Never cite a number that is not in the results and never invent a URL.",
        "4. If the results do not answer the question, say what you found and what is still unknown. "
        "Do not fill the gap with a guess.",
        "5. Be concise. Four to eight sentences, or a short bullet list. No preamble, no restating the question.",
        "6. Match the user's language and register. Use plain words rather than jargon.",
    ]
    if memory_block:
        parts += [
            "",
            memory_block,
            "7. The MEMORY list is what this user asked before, with local timestamps. When they ask "
            "what they asked earlier, or when, answer from it with the dates and times exactly as shown. "
            "Do not answer a question about their history from web results.",
        ]
    return "\n".join(parts)


def _parse_tool_payload(content: Any) -> dict:
    if isinstance(content, list):
        content = "".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block) for block in content
        )
    if not isinstance(content, str):
        return {}
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def build_graph(
    model: BaseChatModel,
    engine: SearchEngine,
    max_rounds: int = MAX_SEARCH_ROUNDS,
    closer: BaseChatModel | None = None,
):
    async def run_search(query: str) -> str:
        outcome = await engine.search(query)
        return json.dumps(outcome.to_dict(), ensure_ascii=False)

    tool = StructuredTool.from_function(
        coroutine=run_search,
        name="web_search",
        description=(
            "Search the public web. Returns numbered results, each with a title, url, and snippet. "
            "Use for current or uncertain facts, then cite result numbers as [1], [2] in the answer."
        ),
        args_schema=SearchArgs,
    )
    bound = model.bind_tools([tool])
    plain = closer or model
    tool_node = ToolNode([tool])

    async def agent(state: AgentState) -> dict:
        rounds = state.get("rounds", 0)
        messages = [SystemMessage(content=state["system"]), *state["messages"]]
        if rounds >= max_rounds:
            response = await plain.ainvoke(messages)
            text = response.content if isinstance(response.content, str) else ""
            if not text.strip() and not getattr(response, "tool_calls", None):
                return {"messages": [AIMessage(content=results_summary(state.get("sources") or []))]}
        else:
            response = await bound.ainvoke(messages)
            text = response.content if isinstance(response.content, str) else ""
        structured = bool(getattr(response, "tool_calls", None))
        markup = has_tool_markup(text)
        if structured and rounds < max_rounds:
            return {"messages": [response]}
        if not structured and markup:
            leaked = parse_markup_tool_call(text)
            if leaked and rounds == 0:
                return {"messages": [AIMessage(content="", tool_calls=[leaked])]}
        if not structured and not markup:
            return {"messages": [response]}

        question = next(
            (str(message.content) for message in reversed(state["messages"]) if isinstance(message, HumanMessage)),
            "Answer the question.",
        )
        retry = await plain.ainvoke(
            [HumanMessage(content=closing_prompt(question, state.get("sources") or []))]
        )
        retry_text = retry.content if isinstance(retry.content, str) else ""
        answer = strip_tool_markup(retry_text) if not getattr(retry, "tool_calls", None) else ""
        if not answer.strip():
            answer = results_summary(state.get("sources") or [])
        return {"messages": [AIMessage(content=answer)]}

    def collect(state: AgentState) -> dict:
        sources = list(state.get("sources") or [])
        handled = list(state.get("handled_tools") or [])
        seen_urls = {source["url"] for source in sources}
        for message in state["messages"]:
            if not isinstance(message, ToolMessage) or message.tool_call_id in handled:
                continue
            payload = _parse_tool_payload(message.content)
            handled.append(message.tool_call_id)
            for result in payload.get("results") or []:
                url = str(result.get("url", ""))
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                sources.append({**result, "index": len(sources) + 1})
        return {"sources": sources, "rounds": state.get("rounds", 0) + 1, "handled_tools": handled}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls and state.get("rounds", 0) < max_rounds:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_node("collect", collect)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "collect")
    graph.add_edge("collect", "agent")
    return graph.compile()
