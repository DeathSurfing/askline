"""A small task tree that lets the agent delegate research and design work.

The root model plans one to three tasks. Research tasks search the web and may
spawn follow-ups, design tasks turn findings into an outline, and critic tasks
look for gaps. The tree is capped by depth and node count so a slow local model
cannot wander. Every source is numbered once, globally, so citations stay valid
across tasks.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage

from .citations import append_sources
from .config import ChatConfig
from .llm import chunk_text, text_model
from .openui import parse_json
from .search import SearchEngine, Source

MAX_DEPTH = 3
MAX_CHILDREN = 3
MAX_NODES = 7
NODE_TIMEOUT = 45.0
ROLES = ("research", "design", "critic")
PLAN_SCHEMA = (
    '{"tasks": [{"role": "research" | "design" | "critic", "brief": "one sentence"}]}'
)
NODE_SCHEMA = '{"findings": ["one or two sentences", "..."], "followups": [{"role": "...", "brief": "..."}]}'


class DelegateError(Exception):
    """The task tree could not produce an answer."""


@dataclass
class Task:
    id: str
    parent_id: str | None
    depth: int
    role: str
    brief: str
    status: str = "queued"
    findings: list[str] = field(default_factory=list)
    source_indexes: list[int] = field(default_factory=list)
    children: list["Task"] = field(default_factory=list)
    elapsed_ms: int = 0
    error: str = ""

    def trace(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "role": self.role,
            "brief": self.brief,
            "status": self.status,
            "findings": self.findings,
            "sources": self.source_indexes,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }


def _text(value: Any, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def parse_plan(text: str, limit: int = MAX_CHILDREN) -> list[dict[str, str]]:
    data = parse_json(text)
    tasks = data.get("tasks") if isinstance(data, dict) else None
    if not isinstance(tasks, list):
        raise ValueError("plan has no tasks")
    plan: list[dict[str, str]] = []
    for item in tasks:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "research").lower()
        if role not in ROLES:
            role = "research"
        brief = _text(item.get("brief"), 240)
        if brief:
            plan.append({"role": role, "brief": brief})
    if not plan:
        raise ValueError("plan has no usable tasks")
    return plan[:limit]


def parse_node(text: str, allow_followups: bool) -> tuple[list[str], list[dict[str, str]]]:
    data = parse_json(text)
    findings = [_text(item, 320) for item in data.get("findings") or []]
    findings = [item for item in findings if len(item) >= 8][:4]
    if not findings:
        raise ValueError("node has no findings")
    followups: list[dict[str, str]] = []
    if allow_followups:
        for item in data.get("followups") or []:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "research").lower()
            if role not in ROLES:
                role = "research"
            brief = _text(item.get("brief"), 240)
            if brief:
                followups.append({"role": role, "brief": brief})
    return findings, followups[:MAX_CHILDREN]


def plan_prompt(question: str, history: str = "") -> str:
    context = f"Conversation so far:\n{history}\n\n" if history else ""
    return (
        "Plan a small task tree that answers the user's question. "
        "Use one to three tasks. Research tasks search the web, design tasks turn findings into an outline, "
        "critic tasks look for gaps. Most questions need one research task. "
        f"Return JSON only: {PLAN_SCHEMA}\n\n{context}Question: {question}\n\nRewrite the task briefs so they stand alone."
    )


def node_prompt(task: Task, question: str, results: list[Source], findings: list[str]) -> str:
    lines = [f"Question: {question}", f"Role: {task.role}", f"Task: {task.brief}"]
    if results:
        lines.append("Search results:")
        for item in results:
            lines.append(f"[{item.index}] {item.title}")
            if item.snippet:
                lines.append(item.snippet[:280])
    if findings:
        lines.append("Findings from the parent task:")
        lines.extend(f"- {item}" for item in findings[:6])
    lines.append(
        "Do the task and return JSON only. Findings are one or two sentences each, with [n] citations "
        "when a result supports them. Do not invent facts. "
        f"Shape: {NODE_SCHEMA}"
    )
    if task.depth < MAX_DEPTH and task.role == "research":
        lines.append("Add up to two follow-ups only when they are needed to answer the question.")
    else:
        lines.append("Return an empty followups array.")
    return "\n".join(lines)


async def ask_json(model, prompt: str, timeout: float = NODE_TIMEOUT) -> str:
    reply = await asyncio.wait_for(
        model.ainvoke([SystemMessage(content="Return JSON only. Never write prose outside the object."), HumanMessage(content=prompt)]),
        timeout=timeout,
    )
    return chunk_text(reply)


def tidy_answer(text: str) -> str:
    """Drop reasoning asides and repeated drafts that small models leave in."""
    text = re.sub(
        r"(?i)^\s*(the user asked|the user wants|the conversation so far|likely they want|i need to|we must)\b[^.]*\.\s*",
        "",
        text,
    ).strip()
    paragraphs: list[str] = []
    seen: set[str] = set()
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if re.match(r"(?i)^(final answer|answer)\s*:", block) and paragraphs:
            continue
        key = re.sub(r"\W+", "", block.lower())[:140]
        if key in seen:
            continue
        seen.add(key)
        paragraphs.append(block)
    return "\n\n".join(paragraphs)


async def resolve_question(model, question: str, history: str) -> str:
    """Turn a follow-up into a standalone search question using the thread."""
    if not history:
        return question
    prompt = (
        "Rewrite the user's latest message as one standalone search question. "
        "Use the conversation only to resolve references, and keep every entity and constraint. "
        "Return JSON only: {\"query\": \"...\"}\n\n"
        f"Conversation:\n{history}\n\nLatest message: {question}"
    )
    try:
        data = parse_json(await ask_json(model, prompt))
        query = _text(data.get("query"), 220)
        return query or question
    except Exception:
        return question


async def run_delegation(
    store,
    engine: SearchEngine,
    cfg: ChatConfig,
    question: str,
    history: str = "",
) -> AsyncIterator[tuple[str, dict]]:
    """Yield task events and a final result with text, sources, and the trace."""
    started = time.perf_counter()
    model = text_model(cfg)
    closer = text_model(cfg)
    sources: list[dict] = []
    tasks: list[Task] = []
    counter = 0

    def new_task(parent: Task | None, role: str, brief: str) -> Task:
        nonlocal counter
        counter += 1
        return Task(
            id=f"t{counter}",
            parent_id=parent.id if parent else None,
            depth=(parent.depth + 1) if parent else 1,
            role=role,
            brief=brief,
        )

    resolved = await resolve_question(model, question, history)
    try:
        plan_text = await ask_json(model, plan_prompt(resolved, history))
        plan = parse_plan(plan_text)
    except Exception:
        plan = [{"role": "research", "brief": question}]

    roots = [new_task(None, item["role"], item["brief"]) for item in plan]

    async def run_task(task: Task) -> AsyncIterator[tuple[str, dict]]:
        if len(tasks) >= MAX_NODES:
            return
        tasks.append(task)
        task.status = "running"
        node_started = time.perf_counter()
        payload = {"id": task.id, "parent_id": task.parent_id, "depth": task.depth, "role": task.role, "brief": task.brief}
        yield "task", {**payload, "status": "running"}
        found: list[Source] = []
        try:
            if task.role == "research":
                outcome = await engine.search(task.brief, count=4)
                if outcome.results:
                    before = {item["url"] for item in sources}
                    sources[:] = append_sources(sources, [item.to_dict() for item in outcome.results[:4]])
                    task.source_indexes = [item["index"] for item in sources if item["url"] not in before]
                    found = [Source.from_dict(item) for item in sources if item["index"] in task.source_indexes]
            parent_findings: list[str] = []
            if task.parent_id:
                parent = next((item for item in tasks if item.id == task.parent_id), None)
                parent_findings = parent.findings if parent else []
            text = await ask_json(model, node_prompt(task, resolved, found, parent_findings))
            task.findings, followups = parse_node(text, allow_followups=task.role == "research")
            for item in followups:
                if len(tasks) + len(task.children) >= MAX_NODES or task.depth >= MAX_DEPTH:
                    break
                task.children.append(new_task(task, item["role"], item["brief"]))
            task.status = "done"
        except Exception as error:
            task.status = "failed"
            task.error = f"{type(error).__name__}: {error}"
        task.elapsed_ms = int((time.perf_counter() - node_started) * 1000)
        yield "task_done", {
            **payload,
            "status": task.status,
            "findings": task.findings,
            "elapsed_ms": task.elapsed_ms,
            "error": task.error,
        }

    async def walk(task: Task) -> AsyncIterator[tuple[str, dict]]:
        async for event in run_task(task):
            yield event
        for child in task.children:
            if len(tasks) >= MAX_NODES:
                break
            async for event in walk(child):
                yield event

    for root in roots:
        if len(tasks) >= MAX_NODES:
            break
        async for event in walk(root):
            yield event

    findings = [item for task in tasks for item in task.findings]
    if not findings:
        raise DelegateError("The subagents returned no findings. Try a narrower question.")
    listing = "\n".join(f"- {item}" for task in tasks for item in task.findings)
    source_list = "\n".join(f"[{item['index']}] {item['title']}" for item in sources[:12])
    answer_prompt = (
        "Answer the user's question from these subagent findings. "
        "Use four to eight sentences. Cite claims with [n] using only the numbers shown. "
        "Prefer concrete figures, dates, and names over advice about where to look. "
        "If the question asks for a dashboard, chart, or trend, list the actual values with their years or categories, one per line. "
        "Never repeat a sentence or a paragraph. Do not write 'Final Answer', 'Answer:', or 'Thus'. "
        "Do not describe the request, the conversation, or your reasoning. "
        "Do not invent facts or URLs. Do not mention the subagents or the task tree.\n\n"
        f"Conversation so far:\n{history or '(new conversation)'}\n\nQuestion: {question}\n\nFindings:\n{listing}\n\nSources:\n{source_list}"
    )
    reply = await asyncio.wait_for(
        closer.ainvoke([SystemMessage(content="You are Askline. Write the final answer only."), HumanMessage(content=answer_prompt)]),
        timeout=NODE_TIMEOUT,
    )
    text = tidy_answer(chunk_text(reply).strip())
    if len(text) < 20:
        text = " ".join(item for task in tasks for item in task.findings)
    yield "final", {
        "text": text,
        "sources": sources,
        "trace": [task.trace() for task in tasks],
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }
