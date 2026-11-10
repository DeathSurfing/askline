from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import sse, stream_turn
from .config import ChatConfig, db_path, load_config, save_overrides
from .llm import list_models, test_connection
from .memory import Store
from .prompts import suggested_prompts
from .search import SearchEngine

STATIC_DIR = Path(__file__).parent / "static"
MAX_MESSAGE_CHARS = 8000


class ChatRequest(BaseModel):
    thread_id: str | None = None
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    timezone: str | None = None


class SettingsRequest(BaseModel):
    provider: str | None = None
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = Store(db_path())
    engine = SearchEngine(store)
    engine.prune()
    app.state.store = store
    app.state.engine = engine
    yield


app = FastAPI(title="Askline", version="1.0.0", lifespan=lifespan)


def store(request: Request) -> Store:
    return request.app.state.store


def engine(request: Request) -> SearchEngine:
    return request.app.state.engine


@app.get("/")
async def index():
    page = STATIC_DIR / "index.html"
    if not page.exists():
        return JSONResponse({"error": "UI not built"}, status_code=503)
    return FileResponse(page)


@app.get("/api/health")
async def health(request: Request):
    with store(request).connection() as conn:
        cfg = load_config(conn)
    return {
        "ok": True,
        "provider": cfg.provider,
        "model": cfg.model,
        "base_url": cfg.base_url,
        "has_api_key": bool(cfg.api_key),
        "search_backend": engine(request).last_backend,
        "search_error": engine(request).last_error,
        "searxng": bool(engine(request).searxng_url),
    }


@app.get("/api/settings")
async def get_settings(request: Request):
    with store(request).connection() as conn:
        return load_config(conn).public()


def validated_settings(payload: SettingsRequest) -> dict[str, str]:
    values: dict[str, str] = {}
    if payload.provider is not None:
        if payload.provider not in ("openai", "anthropic"):
            raise HTTPException(400, "provider must be openai or anthropic")
        values["provider"] = payload.provider
    if payload.base_url is not None:
        if not payload.base_url.startswith(("http://", "https://")):
            raise HTTPException(400, "base_url must start with http:// or https://")
        values["base_url"] = payload.base_url
    if payload.model is not None:
        if not payload.model.strip():
            raise HTTPException(400, "model cannot be empty")
        values["model"] = payload.model
    if payload.api_key is not None and payload.api_key.strip():
        values["api_key"] = payload.api_key.strip()
    if payload.temperature is not None:
        if not 0 <= payload.temperature <= 2:
            raise HTTPException(400, "temperature must be between 0 and 2")
        values["temperature"] = f"{payload.temperature}"
    if payload.max_tokens is not None:
        if not 64 <= payload.max_tokens <= 8000:
            raise HTTPException(400, "max_tokens must be between 64 and 8000")
        values["max_tokens"] = f"{payload.max_tokens}"
    return values


@app.post("/api/settings")
async def put_settings(payload: SettingsRequest, request: Request):
    values = validated_settings(payload)
    with store(request).connection() as conn:
        save_overrides(conn, values)
        return load_config(conn).public()


@app.post("/api/settings/test")
async def settings_test(request: Request, payload: SettingsRequest | None = None):
    values = validated_settings(payload) if payload else {}
    with store(request).connection() as conn:
        cfg = load_config(conn)
    updates: dict = {key: value for key, value in values.items() if key not in ("temperature", "max_tokens")}
    if "temperature" in values:
        updates["temperature"] = float(values["temperature"])
    if "max_tokens" in values:
        updates["max_tokens"] = int(values["max_tokens"])
    return await test_connection(replace(cfg, **updates))


@app.get("/api/models")
async def models(request: Request):
    with store(request).connection() as conn:
        cfg = load_config(conn)
    return await list_models(cfg)


@app.get("/api/prompts")
async def prompts(request: Request):
    with store(request).connection() as conn:
        cfg = load_config(conn)
    items, source = await suggested_prompts(store(request), cfg)
    return {"prompts": items, "source": source}


@app.get("/api/threads")
async def list_threads(request: Request):
    return {"threads": store(request).list_threads()}


@app.post("/api/threads")
async def create_thread(request: Request):
    return store(request).create_thread("New chat")


@app.get("/api/threads/{thread_id}")
async def get_thread(thread_id: str, request: Request):
    thread = store(request).get_thread(thread_id)
    if not thread:
        raise HTTPException(404, "thread not found")
    return {"thread": thread, "turns": store(request).thread_turns(thread_id)}


@app.delete("/api/threads/{thread_id}")
async def delete_thread(thread_id: str, request: Request):
    if not store(request).delete_thread(thread_id):
        raise HTTPException(404, "thread not found")
    return {"ok": True}


@app.get("/api/timeline")
async def timeline(request: Request, limit: int = 100, tz: str | None = None):
    limit = max(1, min(limit, 500))
    zone_name = resolve_zone(tz) or _zone_name()
    rows = store(request).timeline(limit)
    days: dict[str, list[dict]] = {}
    for row in rows:
        moment = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
        local = _to_local(moment, zone_name)
        days.setdefault(local.date().isoformat(), []).append(
            {
                "id": row["id"],
                "thread_id": row["thread_id"],
                "thread_title": row.get("thread_title") or "",
                "text": row["text"],
                "created_at": row["created_at"],
                "local_time": local.strftime("%H:%M"),
            }
        )
    return {
        "days": [{"date": date, "asks": asks} for date, asks in days.items()],
        "timezone": zone_name,
        "now": datetime.now(timezone.utc).isoformat(),
    }


def _zone_name() -> str:
    from .config import timezone_name

    return timezone_name()


def resolve_zone(tz: str | None) -> str | None:
    if not tz:
        return None
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError):
        return None
    return tz


def _to_local(moment: datetime, zone_name: str) -> datetime:
    if zone_name == "local":
        return moment.astimezone()
    from zoneinfo import ZoneInfo

    return moment.astimezone(ZoneInfo(zone_name))


@app.post("/api/chat")
async def chat(payload: ChatRequest, request: Request):
    message = payload.message.strip()
    if not message:
        raise HTTPException(400, "message cannot be empty")
    with store(request).connection() as conn:
        cfg = load_config(conn)
    if not cfg.api_key:
        raise HTTPException(400, "No API key configured. Open Settings and add one.")
    if payload.timezone and not resolve_zone(payload.timezone):
        raise HTTPException(400, "timezone must be an IANA name such as Europe/London")

    data = store(request)
    thread = data.get_thread(payload.thread_id) if payload.thread_id else None
    if payload.thread_id and not thread:
        raise HTTPException(404, "thread not found")
    if not thread:
        thread = data.create_thread(message[:70])
    elif thread["title"] in ("", "New chat") and thread["turn_count"] == 0:
        data.rename_thread(thread["id"], message[:70])

    user_turn = data.add_turn(thread["id"], "user", message)
    turn_stream = stream_turn(
        store=data,
        engine=engine(request),
        cfg=cfg,
        thread_id=thread["id"],
        user_text=message,
        user_turn=user_turn,
        tz=resolve_zone(payload.timezone),
    )

    async def events():
        try:
            async for event, body in turn_stream:
                yield sse(event, body)
        except Exception as exc:
            yield sse("error", {"message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
