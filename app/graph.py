from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from .search import SearchEngine

MAX_SEARCH_ROUNDS = 2


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
        "or any fact you are not certain about. Search once; search again only if the first results "
        "miss the question or contradict each other. Write the query with the distinctive words only: "
        "names, products, versions. Drop filler like latest, news, and best, and drop question words.",
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


def build_graph(model: BaseChatModel, engine: SearchEngine, max_rounds: int = MAX_SEARCH_ROUNDS):
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
    tool_node = ToolNode([tool])

    async def agent(state: AgentState) -> dict:
        messages = [SystemMessage(content=state["system"]), *state["messages"]]
        if state.get("rounds", 0) >= max_rounds:
            response = await model.ainvoke(messages)
        else:
            response = await bound.ainvoke(messages)
        return {"messages": [response]}

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
