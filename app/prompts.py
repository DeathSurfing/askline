import asyncio
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import ChatConfig, is_bundled
from app.llm import build_chat_model
from app.memory import Store

DEFAULT_PROMPTS = [
    "What changed in the latest Python release?",
    "Compare Postgres logical and physical replication",
    "Summarize this week's notable open-source security advisories",
]
CACHE_KEY = "prompt_suggestions_v3"
LEAD = re.compile(r"^(?:[-*•]|\d+[.)])\s*")
SYSTEM = (
    "You suggest follow-up questions for a research assistant. "
    "Given the user's recent questions, propose 3 questions they are likely to ask next. "
    "Return one per line, no numbering, no commentary, under 70 characters."
)


def parse_suggestions(text: str, limit: int = 3) -> list[str]:
    seen: set[str] = set()
    prompts: list[str] = []
    for raw in (text or "").splitlines():
        line = LEAD.sub("", raw.strip()).strip('"“”\' ')
        if not line.endswith("?") or not 8 <= len(line) <= 90 or line in seen:
            continue
        seen.add(line)
        prompts.append(line)
        if len(prompts) == limit:
            break
    return prompts


def _fill(prompts: list[str]) -> list[str]:
    for fallback in DEFAULT_PROMPTS:
        if len(prompts) >= 3:
            break
        if fallback not in prompts:
            prompts.append(fallback)
    return prompts[:3]


async def suggested_prompts(store: Store, cfg: ChatConfig, timeout: float = 15.0) -> tuple[list[str], str]:
    count = store.ask_count()
    cached = store.setting(CACHE_KEY)
    if cached:
        try:
            data = json.loads(cached)
            if data.get("count") == count and data.get("prompts"):
                return list(data["prompts"])[:3], "history"
        except (ValueError, TypeError):
            pass

    questions = store.recent_questions(20)
    if len(questions) < 2 or not cfg.configured:
        return list(DEFAULT_PROMPTS), "defaults"

    extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if is_bundled(cfg) else None
    model = build_chat_model(cfg, extra_body=extra_body)
    history = "\n".join(f"- {question}" for question in questions)
    try:
        async with asyncio.timeout(timeout):
            reply = await model.ainvoke(
                [SystemMessage(content=SYSTEM), HumanMessage(content=history)]
            )
    except Exception:
        return list(DEFAULT_PROMPTS), "defaults"

    text = reply.content if isinstance(reply.content, str) else ""
    prompts = parse_suggestions(text)
    if not prompts:
        return list(DEFAULT_PROMPTS), "defaults"

    prompts = _fill(prompts)
    store.save_setting(CACHE_KEY, json.dumps({"count": count, "prompts": prompts}))
    return prompts, "history"
