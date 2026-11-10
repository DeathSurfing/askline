from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import AsyncIterator
from zoneinfo import ZoneInfo

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .citations import cited_indices, merge_sources, strip_invalid
from .config import ChatConfig, is_bundled, timezone_name
from .graph import _parse_tool_payload, build_graph, has_tool_markup, system_prompt
from .llm import build_chat_model, chunk_text
from .memory import Store, format_time
from .openui import compose_ui
from .schemas import Source
from .search import SearchEngine
from .subagents import DelegateError, run_delegation

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


def _strip_stream_markup(buffer: str) -> tuple[str, str]:
    """Hide tool calls a model writes as text, and keep the unfinished tail buffered."""
    if len(buffer) < len("<tool_call>") and "<tool_call>".startswith(buffer):
        return "", buffer
    out = ""
    while buffer.startswith("<tool_call>"):
        end = buffer.find("</tool_call>")
        if end == -1:
            return out, buffer
        buffer = buffer[end + len("</tool_call>") :]
    return out + buffer, ""


async def stream_turn(
    store: Store,
    engine: SearchEngine,
    cfg: ChatConfig,
    thread_id: str,
    user_text: str,
    user_turn,
    tz: str | None = None,
    delegate: bool = False,
    visual: bool = False,
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

    history = [
        message for message in _history(store, thread_id)
        if not (isinstance(message, HumanMessage) and message.content == user_text)
    ][-8:]
    history_text = "\n".join(
        f"{'User' if isinstance(message, HumanMessage) else 'Askline'}: {message.content}"
        for message in history
    )

    text = ""
    sources: list[dict] = []
    trace: list[dict] = []
    token_count = 0
    rate = 0.0

    if delegate:
        yield "delegate_start", {}
        result = None
        try:
            async for event, body in run_delegation(store, engine, cfg, user_text, history=history_text):
                if event == "final":
                    result = body
                else:
                    yield event, body
        except DelegateError as exc:
            yield "error", {"message": str(exc)}
            return
        except Exception as exc:
            yield "error", {"message": f"{type(exc).__name__}: {exc}"}
            return
        if not result:
            yield "error", {"message": "The subagents returned no answer."}
            return
        text = strip_invalid(result["text"], len(result["sources"])).strip()
        sources = result["sources"]
        trace = list(result["trace"])
    else:
        model = build_chat_model(cfg)
        closer = (
            build_chat_model(cfg, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            if is_bundled(cfg)
            else None
        )
        graph = build_graph(model, engine, closer=closer)
        state = {
            "messages": _history(store, thread_id),
            "system": system_prompt(memory_block, now, zone_label),
            "sources": [],
            "rounds": 0,
            "handled_tools": [],
        }

        tokens: list[str] = []
        fallback = ""
        markup_buffer = ""
        first_token_at: float | None = None
        last_stats_at = 0.0

        async def consume() -> AsyncIterator[tuple[str, dict]]:
            nonlocal fallback, first_token_at, last_stats_at, token_count, markup_buffer
            async for mode, payload in graph.astream(state, stream_mode=["messages", "updates"]):
                if mode == "messages":
                    chunk, metadata = payload
                    if not _from_agent(metadata):
                        continue
                    delta, markup_buffer = _strip_stream_markup(markup_buffer + chunk_text(chunk))
                    if delta:
                        tokens.append(delta)
                        token_count += 1
                        now_mono = time.perf_counter()
                        if first_token_at is None:
                            first_token_at = now_mono
                        yield "token", {"delta": delta}
                        elapsed = now_mono - first_token_at
                        if elapsed >= 0.25 and now_mono - last_stats_at >= 0.25:
                            last_stats_at = now_mono
                            yield "stats", {
                                "tokens": token_count,
                                "seconds": round(elapsed, 2),
                                "tok_per_s": round(token_count / elapsed, 1) if elapsed > 0 else 0.0,
                            }
                    continue

                for node, update in payload.items():
                    if not isinstance(update, dict):
                        continue
                    if node == "agent":
                        last = (update.get("messages") or [None])[-1]
                        calls = getattr(last, "tool_calls", None) or []
                        for call in calls:
                            yield "search_start", {"query": (call.get("args") or {}).get("query", "")}
                        active = chunk_text(last) if last else ""
                        if (calls or has_tool_markup(active)) and tokens:
                            tokens.clear()
                            fallback = ""
                            token_count = 0
                            first_token_at = None
                            last_stats_at = 0.0
                            yield "reset", {}
                        elif active and not has_tool_markup(active):
                            fallback = active
                    elif node == "tools":
                        for message in update.get("messages") or []:
                            if not isinstance(message, ToolMessage):
                                continue
                            tool_payload = _parse_tool_payload(message.content)
                            sources[:] = merge_sources(sources, tool_payload.get("results") or [])
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

        text = (fallback or "".join(tokens)).strip()
        if not text:
            yield "error", {"message": "The model returned an empty answer. Try again or switch models."}
            return
        text = strip_invalid(text, len(sources)).strip()
        generation_seconds = (time.perf_counter() - first_token_at) if first_token_at else 0.0
        rate = round(token_count / generation_seconds, 1) if generation_seconds > 0 else 0.0

    if visual:
        yield "ui_start", {}
        try:
            card = await compose_ui(
                cfg,
                user_text,
                text,
                [Source.from_dict(item) for item in sources[:5]],
            )
            yield "ui", {
                "lang": card["lang"],
                "layout": card["layout"],
                "title": card["title"],
                "sources": sources,
            }
            trace = trace + [{"kind": "ui", "lang": card["lang"], "layout": card["layout"], "title": card["title"]}]
        except Exception as exc:
            yield "ui_error", {"message": f"Visual skipped: {exc}"}

    cited = cited_indices(text, len(sources))
    message = store.add_turn(
        thread_id,
        "assistant",
        text,
        [Source.from_dict(item) for item in sources],
        trace=trace,
    )
    yield (
        "done",
        {
            "message_id": message.id,
            "text": text,
            "sources": sources,
            "cited": cited,
            "trace": trace,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "tokens": token_count,
            "tok_per_s": rate,
        },
    )


def sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
