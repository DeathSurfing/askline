from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Provider = Literal["openai", "anthropic"]

DEFAULT_BASE_URLS: dict[Provider, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
}
DEFAULT_MODELS: dict[Provider, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-5",
}

SETTING_KEYS = ("provider", "base_url", "model", "api_key", "temperature", "max_tokens")


@dataclass(frozen=True)
class ChatConfig:
    provider: Provider
    base_url: str
    model: str
    api_key: str
    temperature: float = 0.2
    max_tokens: int = 800

    @property
    def configured(self) -> bool:
        return bool(self.model and self.api_key)

    def public(self) -> dict:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "has_api_key": bool(self.api_key),
        }


def data_dir() -> Path:
    path = Path(os.environ.get("DATA_DIR", "./data")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def db_path() -> Path:
    return data_dir() / "chat.db"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def resolve_provider(provider: str, base_url: str) -> Provider:
    value = provider.strip().lower()
    if value == "auto":
        if "anthropic" in base_url.lower() or _env("ANTHROPIC_API_KEY"):
            return "anthropic"
        return "openai"
    if value not in ("openai", "anthropic"):
        raise ValueError("provider must be openai, anthropic, or auto")
    return value  # type: ignore[return-value]


def read_overrides(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute("SELECT key, value FROM settings").fetchall()
    return {row["key"]: row["value"] for row in rows}


def load_config(conn: sqlite3.Connection | None = None) -> ChatConfig:
    stored = read_overrides(conn) if conn is not None else {}
    provider_raw = stored.get("provider") or _env("LLM_PROVIDER", "openai")
    base_url = stored.get("base_url") or _env("LLM_BASE_URL")
    provider = resolve_provider(provider_raw, base_url)

    if not base_url:
        base_url = DEFAULT_BASE_URLS[provider]
    api_key = (
        stored.get("api_key")
        or _env("LLM_API_KEY")
        or _env("OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY")
    )
    model = stored.get("model") or _env("LLM_MODEL") or DEFAULT_MODELS[provider]
    temperature = float(stored.get("temperature") or _env("LLM_TEMPERATURE", "0.2"))
    max_tokens = int(stored.get("max_tokens") or _env("LLM_MAX_TOKENS", "800"))

    return ChatConfig(
        provider=provider,
        base_url=base_url.rstrip("/"),
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def save_overrides(conn: sqlite3.Connection, values: dict[str, str]) -> None:
    for key in SETTING_KEYS:
        if key not in values:
            continue
        value = str(values[key]).strip()
        if key == "api_key" and value == "":
            continue
        conn.execute(
            "INSERT INTO settings(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
    conn.commit()


def timezone_name() -> str:
    return _env("AGENT_TZ", "local")


def is_bundled(cfg: ChatConfig) -> bool:
    """True for the compose model server, which accepts chat_template_kwargs."""
    import os
    from urllib.parse import urlparse

    if os.environ.get("LLM_DISABLE_THINKING", "").strip().lower() in {"1", "true", "yes"}:
        return True
    return (urlparse(cfg.base_url).hostname or "") == "llm"
