from __future__ import annotations

import time
from typing import Any

import httpx
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .config import ChatConfig, is_bundled


def text_model(cfg: ChatConfig) -> BaseChatModel:
    """A model that answers in text, with reasoning turned off on the bundled server."""
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if is_bundled(cfg) else None
    return build_chat_model(cfg, extra_body=extra)


def build_chat_model(cfg: ChatConfig, extra_body: dict | None = None) -> BaseChatModel:
    if not cfg.api_key:
        raise RuntimeError("No API key configured. Open Settings and add one, or set LLM_API_KEY.")
    if cfg.provider == "anthropic":
        return ChatAnthropic(
            model=cfg.model,
            base_url=cfg.base_url or None,
            api_key=cfg.api_key,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=60,
            max_retries=1,
        )
    return ChatOpenAI(
        model=cfg.model,
        base_url=cfg.base_url or None,
        api_key=cfg.api_key,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=60,
        max_retries=1,
        extra_body=extra_body,
    )


def chunk_text(chunk: Any) -> str:
    """Chat models return str content, or a list of content blocks on newer APIs."""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in ("text", "text_delta"):
                parts.append(str(block.get("text", "")))
    return "".join(parts)


async def test_connection(cfg: ChatConfig) -> dict:
    try:
        model = build_chat_model(cfg)
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    started = time.perf_counter()
    try:
        await model.ainvoke("Reply with the single word: ok")
    except Exception as exc:
        return {"ok": False, "error": str(exc).strip()[:400]}
    return {"ok": True, "latency_ms": int((time.perf_counter() - started) * 1000)}


async def list_models(cfg: ChatConfig) -> dict:
    if not cfg.api_key:
        return {"models": [], "error": "No API key configured"}
    if cfg.provider == "anthropic":
        url = f"{cfg.base_url}/v1/models"
        headers = {"x-api-key": cfg.api_key, "anthropic-version": "2023-06-01"}
    else:
        url = f"{cfg.base_url}/models"
        headers = {"Authorization": f"Bearer {cfg.api_key}"}
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(url, headers=headers)
        if response.status_code >= 400:
            return {"models": [], "error": f"HTTP {response.status_code}: {response.text[:200]}"}
        payload = response.json()
        raw = payload.get("data") or payload.get("models") or []
        models = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id") or item.get("name")
            if not model_id:
                continue
            models.append({"id": str(model_id), "label": str(item.get("display_name") or model_id)})
        models.sort(key=lambda item: item["id"])
        return {"models": models, "error": ""}
    except Exception as exc:
        return {"models": [], "error": f"{type(exc).__name__}: {exc}"[:300]}
