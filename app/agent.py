from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import AsyncIterator
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .citations import cited_indices, strip_invalid
from .config import ChatConfig, timezone_name
from .graph import _parse_tool_payload, build_graph, system_prompt
from .llm import build_chat_model, chunk_text
from .memory import Store, format_time
from .schemas import Source
from .search import SearchEngine

HISTORY_TURNS = 12
TURN_TIMEOUT = 120.0


def local_now(tz: str | None = None) -> tuple[datetime, str]:
    zone = (tz or timezone_name()).strip() or "local"
    if zone == "local":
        return datetime.now().astimezone(), "local time"
    return datetime.now(ZoneInfo(zone)), zone


def _history(store: Store, thread_id: str) -> list:
    messages: list = []
    for turn in store.recent_turns(thread_id, limit=HISTORY_TURNS):
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.text))
        elif turn.role == "assistant":
            messages.append(AIMessage(content=turn.text))
    return messages


def _from_agent(metadata: dict) -> bool:
    if metadata.get("langgraph_node"):
        return metadata["langgraph_node"] == "agent"
    return "agent" in (metadata.get("tags") or [])


def _merge_sources(existing: list[dict], incoming: list[dict]) -> list[dict]:
    merged = list(existing)
    seen = {item["url"] for item in merged}
    for result in incoming:
        url = str(result.get("url", ""))
        if not url or url in seen:
            continue
        seen.add(url)
        merged.append(dict(result))
    return [{**item, "index": position} for position, item in enumerate(merged, start=1)]


async def stream_turn(
    store: Store,
    engine: SearchEngine,
    cfg: ChatConfig,
    thread_id: str,
    user_text: str,
    user_turn,
    tz: str | None = None,
) -> AsyncIterator[tuple[str, dict]]:
    """Run one turn and yield (event, payload) pairs for the SSE endpoint."""
    started = time.perf_counter()
    memory_block, memory_items = store.memory_block(user_text, thread_id, exclude_turn=user_turn.id, tz=tz)
    now, zone_label = local_now(tz)
    yield (
        "meta",
        {
            "thread_id": thread_id,
            "turn_id": user_turn.id,
            "created_at": user_turn.created_at,
            "model": cfg.model,
            "provider": cfg.provider,
        },
    )
    if memory_items:
        yield (
            "memory",
            {
                "items": [
                    {
                        "text": item["text"],
                        "created_at": item["created_at"],
                        "local_time": format_time(item["created_at"], tz),
                        "thread_id": item["thread_id"],
                        "thread_title": item.get("thread_title") or "",
                    }
                    for item in memory_items
                ],
                "count": len(memory_items),
            },
        )

    graph = build_graph(build_chat_model(cfg), engine)
    state = {
        "messages": _history(store, thread_id),
        "system": system_prompt(memory_block, now, zone_label),
        "sources": [],
        "rounds": 0,
        "handled_tools": [],
    }

    tokens: list[str] = []
    sources: list[dict] = []
    fallback = ""

    async def consume() -> AsyncIterator[tuple[str, dict]]:
        nonlocal fallback
        async for mode, payload in graph.astream(state, stream_mode=["messages", "updates"]):
            if mode == "messages":
                chunk, metadata = payload
                if not _from_agent(metadata):
                    continue
                delta = chunk_text(chunk)
                if delta:
                    tokens.append(delta)
                    yield "token", {"delta": delta}
                continue

            for node, update in payload.items():
                if not isinstance(update, dict):
                    continue
                if node == "agent":
                    last = (update.get("messages") or [None])[-1]
                    for call in getattr(last, "tool_calls", None) or []:
                        yield "search_start", {"query": (call.get("args") or {}).get("query", "")}
                    text = chunk_text(last) if last else ""
                    if text:
                        fallback = text
                elif node == "tools":
                    for message in update.get("messages") or []:
                        if not isinstance(message, ToolMessage):
                            continue
                        tool_payload = _parse_tool_payload(message.content)
                        sources[:] = _merge_sources(sources, tool_payload.get("results") or [])
                        yield (
                            "search",
                            {
                                "query": tool_payload.get("query", ""),
                                "results": sources,
                                "cached": bool(tool_payload.get("cached")),
                                "backend": tool_payload.get("backend", ""),
                                "error": tool_payload.get("error", ""),
                            },
                        )

    try:
        async with asyncio.timeout(TURN_TIMEOUT):
            async for item in consume():
                yield item
    except TimeoutError:
        yield "error", {"message": "The model took too long to answer. Try a shorter question."}
        return
    except Exception as exc:
        yield "error", {"message": f"{type(exc).__name__}: {exc}"}
        return

    text = "".join(tokens) or fallback
    if not text.strip():
        yield "error", {"message": "The model returned an empty answer. Try again or switch models."}
        return
    text = strip_invalid(text, len(sources)).strip()
    cited = cited_indices(text, len(sources))
    message = store.add_turn(
        thread_id, "assistant", text, [Source.from_dict(item) for item in sources]
    )
    yield (
        "done",
        {
            "message_id": message.id,
            "text": text,
            "sources": sources,
            "cited": cited,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        },
    )


def sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
