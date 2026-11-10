"""Turn a question and search results into an OpenUI Lang program.

Small local models cannot write OpenUI Lang directly. Asking the model for a
small JSON object and composing the program here keeps every generated
interface valid, while the model still chooses the words, the facts, and the
follow-up questions.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain_core.messages import HumanMessage, SystemMessage

from .citations import strip_invalid
from .config import ChatConfig, is_bundled
from .llm import build_chat_model, chunk_text
from .memory import Store
from .search import SearchEngine, Source

MAX_SOURCES = 5
ATTEMPTS = 2
TIMEOUT = 90.0


class OpenUIError(Exception):
    """The model or the search step could not produce an interface."""


def host_of(url: str) -> str:
    host = urlparse(url).hostname or ""
    return re.sub(r"^www\.", "", host)


SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Two to five words."},
        "subtitle": {"type": "string", "description": "Three to seven words."},
        "answer": {"type": "string", "description": "Three to five sentences. Every factual sentence ends with citations such as [1] or [2]."},
        "leftTitle": {"type": "string", "description": "Short name of the left option."},
        "rightTitle": {"type": "string", "description": "Short name of the right option."},
        "items": {
            "type": "array",
            "minItems": 3,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "A short row or point name."},
                    "text": {"type": "string", "description": "One sentence of detail."},
                    "left": {"type": "string", "description": "A short fact for the left option, not its name."},
                    "right": {"type": "string", "description": "A different fact for the right option, not its name."},
                },
                "required": ["label", "text", "left", "right"],
                "additionalProperties": False,
            },
        },
        "followups": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {"type": "string"},
        },
    },
    "required": ["title", "subtitle", "answer", "leftTitle", "rightTitle", "items", "followups"],
    "additionalProperties": False,
}


def pick_layout(question: str) -> str:
    return "comparison" if re.search(r"\b(compare|comparison|versus|vs|difference|trade-?offs?)\b", question, re.I) else "summary"


def system_prompt(layout: str) -> str:
    shape = (
        "Each item has a label for the criterion, such as Scope, Performance, or Flexibility, never an option name. "
        "leftTitle and rightTitle are the two different option names. Left and right are different short facts about the criterion for each option."
        if layout == "comparison"
        else "Each item has a label and one sentence of detail."
    )
    return (
        "You answer a question from numbered web results and return JSON only. "
        "Never write prose outside the JSON object. "
        "Cite claims inside the answer as [1], [2], using only the supplied numbers. Every factual sentence carries a citation. "
        "Never invent facts, numbers, or URLs. "
        f"{shape} "
        "The answer is three to five sentences. Followups are two useful next questions."
    )


def research_brief(question: str, layout: str, results: list[Source]) -> str:
    lines = [f"Question: {question}", f"Layout: {layout}", "Results:"]
    for item in results[:MAX_SOURCES]:
        lines.append(f"[{item.index}] {item.title} ({host_of(item.url)})")
        if item.snippet:
            lines.append(item.snippet[:400])
        lines.append("")
    return "\n".join(lines)


def parse_json(text: str) -> dict[str, Any]:
    body = text.strip()
    body = re.sub(r"^```(?:json)?\s*", "", body)
    body = re.sub(r"\s*```$", "", body)
    start = body.find("{")
    if start == -1:
        raise ValueError("no JSON object")
    data, _ = json.JSONDecoder().raw_decode(body[start:])
    if not isinstance(data, dict):
        raise ValueError("not an object")
    return data


def _text(value: Any, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def titles_equal(raw: dict[str, Any]) -> bool:
    left = _text(raw.get("leftTitle"), 40).lower()
    right = _text(raw.get("rightTitle"), 40).lower()
    return bool(left) and left == right


def normalize(raw: dict[str, Any], layout: str, sources: list[Source]) -> dict[str, Any]:
    count = len(sources)
    answer = strip_invalid(_text(raw.get("answer"), 1600), count)
    if len(answer) < 40:
        raise ValueError("answer too short")
    if count and not re.search(r"\[\d+\]", answer):
        raise ValueError("answer needs at least one [n] citation")
    items: list[dict[str, str]] = []
    for item in raw.get("items") or []:
        if not isinstance(item, dict):
            continue
        entry = {
            "label": strip_invalid(_text(item.get("label"), 60), count),
            "text": strip_invalid(_text(item.get("text"), 240), count),
            "left": strip_invalid(_text(item.get("left"), 220), count),
            "right": strip_invalid(_text(item.get("right"), 220), count),
        }
        if entry["label"]:
            items.append(entry)
    if len(items) < 2:
        raise ValueError("too few items")
    if layout == "comparison" and titles_equal(raw):
        raise ValueError("leftTitle and rightTitle must be different option names")
    followups = [strip_invalid(_text(q, 120), count) for q in raw.get("followups") or []]
    followups = [q for q in followups if 8 <= len(q) <= 120][:2]
    while len(followups) < 2:
        followups.append("What else is worth checking?")
    return {
        "title": strip_invalid(_text(raw.get("title"), 70), count) or "Search result",
        "subtitle": strip_invalid(_text(raw.get("subtitle"), 90), count),
        "answer": answer,
        "leftTitle": strip_invalid(_text(raw.get("leftTitle"), 40), count) or "First",
        "rightTitle": strip_invalid(_text(raw.get("rightTitle"), 40), count) or "Second",
        "items": items[:4],
        "followups": followups,
    }


def quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def source_literal(sources: list[Source]) -> str:
    parts = [
        f"{{url: {quote(item.url)}, title: {quote(item.title)}, sourceName: {quote(host_of(item.url))}}}"
        for item in sources[:MAX_SOURCES]
    ]
    return "sources = [" + ", ".join(parts) + "]"


def build_lang(data: dict[str, Any], layout: str, sources: list[Source]) -> str:
    lines = [
        'root = Card([header, answer, body, followups], sources)',
        f"header = CardHeader({quote(data['title'])}, {quote(data['subtitle'])})",
        f"answer = MarkDownRenderer({quote(data['answer'])})",
    ]
    items = data["items"]
    if layout == "comparison":
        labels = ", ".join(quote(item["label"]) for item in items)
        lefts = ", ".join(quote(item["left"] or item["text"]) for item in items)
        rights = ", ".join(quote(item["right"]) for item in items)
        lines += [
            "body = Table([c1, c2, c3])",
            f'c1 = Col("Criterion", [{labels}])',
            f"c2 = Col({quote(data['leftTitle'])}, [{lefts}])",
            f"c3 = Col({quote(data['rightTitle'])}, [{rights}])",
        ]
    else:
        refs = ", ".join(f"i{index}" for index in range(1, len(items) + 1))
        lines += [
            f'body = ListBlock([{refs}], "number")',
            *(f"i{index} = ListItem({quote(item['label'])}, {quote(item['text'])})" for index, item in enumerate(items, 1)),
        ]
    lines += [
        "followups = FollowUpBlock([q1, q2])",
        f"q1 = FollowUpItem({quote(data['followups'][0])})",
        f"q2 = FollowUpItem({quote(data['followups'][1])})",
        source_literal(sources),
    ]
    return "\n".join(lines)


async def complete(cfg: ChatConfig, messages: list[Any], timeout: float = TIMEOUT) -> str:
    """Ask for JSON. llama.cpp enforces the schema; other hosts best-effort it."""
    if cfg.provider == "anthropic":
        model = build_chat_model(cfg)
        reply = await asyncio.wait_for(model.ainvoke(messages), timeout=timeout)
        return chunk_text(reply)

    url = cfg.base_url.rstrip("/") + "/chat/completions"
    body: dict[str, Any] = {
        "model": cfg.model,
        "messages": [{"role": "user" if message.type == "human" else "system", "content": message.content} for message in messages],
        "temperature": 0.1,
        "max_tokens": 700,
        "stream": False,
    }
    if is_bundled(cfg):
        body["chat_template_kwargs"] = {"enable_thinking": False}
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "askline_ui", "schema": SCHEMA}}
    else:
        body["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=body, headers={"Authorization": f"Bearer {cfg.api_key}"})
        if response.status_code == 400 and "response_format" in response.text:
            body.pop("response_format", None)
            response = await client.post(url, json=body, headers={"Authorization": f"Bearer {cfg.api_key}"})
        response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"].get("content") or ""


async def generate_ui(store: Store, engine: SearchEngine, cfg: ChatConfig, question: str) -> dict[str, Any]:
    started = time.monotonic()
    outcome = await engine.search(question, count=MAX_SOURCES + 1)
    if not outcome.results:
        raise OpenUIError(outcome.error or "No results came back for that question.")
    sources = outcome.results[:MAX_SOURCES]
    layout = pick_layout(question)
    messages: list[Any] = [
        SystemMessage(content=system_prompt(layout) + "\n\nJSON shape:\n" + json.dumps(SCHEMA, indent=2)),
        HumanMessage(content=research_brief(question, layout, sources)),
    ]
    feedback = ""
    for _ in range(ATTEMPTS):
        try:
            text = await asyncio.wait_for(complete(cfg, messages), timeout=TIMEOUT + 15)
            data = normalize(parse_json(text), layout, sources)
            return {
                "lang": build_lang(data, layout, sources),
                "layout": layout,
                "sources": [item.to_dict() for item in sources],
                "question": question,
                "elapsed_ms": int((time.monotonic() - started) * 1000),
            }
        except (ValueError, json.JSONDecodeError) as error:
            feedback = str(error)
            messages.append(HumanMessage(content=f"That reply was not usable ({feedback}). Return the JSON object only."))
        except (asyncio.TimeoutError, httpx.HTTPError):
            break
    raise OpenUIError("The model did not return a usable interface. Try a narrower question.")
