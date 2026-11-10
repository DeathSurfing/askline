from __future__ import annotations

import re
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .config import timezone_name
from .schemas import Source, Turn, iso, utc_now

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS threads(
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS turns(
  id TEXT PRIMARY KEY,
  thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK(role IN ('user','assistant')),
  text TEXT NOT NULL,
  created_at TEXT NOT NULL,
  sources TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS turns_thread_time ON turns(thread_id, created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS asks USING fts5(
  text,
  turn_id UNINDEXED,
  thread_id UNINDEXED,
  created_at UNINDEXED,
  tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS search_cache(
  key TEXT PRIMARY KEY,
  payload TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "in", "is", "it", "me", "my",
    "of", "on", "or", "that", "the", "this", "to", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "with", "you", "your",
}


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9_]{2,}", text.lower())
    return [word for word in words if word not in STOPWORDS][:12]


def local_moment(created_at: str, tz: str | None = None) -> datetime:
    zone = (tz or timezone_name()).strip() or "local"
    moment = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    if zone == "local":
        return moment.astimezone()
    from zoneinfo import ZoneInfo

    return moment.astimezone(ZoneInfo(zone))


def format_time(created_at: str, tz: str | None = None) -> str:
    return local_moment(created_at, tz).strftime("%Y-%m-%d %H:%M")


def relative_day(created_at: str, now: datetime | None = None, tz: str | None = None) -> str:
    local = local_moment(created_at, tz)
    zone = (tz or timezone_name()).strip() or "local"
    if zone == "local":
        today = (now or utc_now()).astimezone().date()
    else:
        from zoneinfo import ZoneInfo

        today = (now or utc_now()).astimezone(ZoneInfo(zone)).date()
    days = (today - local.date()).days
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    if days < 30:
        weeks = days // 7
        return f"{weeks} week{'s' if weeks > 1 else ''} ago"
    return f"{days // 30} month{'s' if days // 30 > 1 else ''} ago"


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
        finally:
            conn.close()

    # threads -----------------------------------------------------------------

    def create_thread(self, title: str = "") -> dict:
        now = iso(utc_now())
        thread_id = uuid.uuid4().hex[:12]
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO threads(id, title, created_at, updated_at) VALUES(?,?,?,?)",
                (thread_id, title or "New chat", now, now),
            )
            conn.commit()
        return self.get_thread(thread_id)  # type: ignore[return-value]

    def get_thread(self, thread_id: str) -> dict | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT t.*, (SELECT COUNT(*) FROM turns WHERE thread_id=t.id) AS turn_count "
                "FROM threads t WHERE t.id=?",
                (thread_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_threads(self, limit: int = 50) -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT t.*, (SELECT COUNT(*) FROM turns WHERE thread_id=t.id) AS turn_count "
                "FROM threads t ORDER BY t.rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def rename_thread(self, thread_id: str, title: str) -> None:
        with self.connection() as conn:
            conn.execute("UPDATE threads SET title=?, updated_at=? WHERE id=?", (title, iso(utc_now()), thread_id))
            conn.commit()

    def delete_thread(self, thread_id: str) -> bool:
        with self.connection() as conn:
            conn.execute("DELETE FROM asks WHERE thread_id=?", (thread_id,))
            cursor = conn.execute("DELETE FROM threads WHERE id=?", (thread_id,))
            conn.commit()
        return cursor.rowcount > 0

    def touch_thread(self, thread_id: str) -> None:
        with self.connection() as conn:
            conn.execute("UPDATE threads SET updated_at=? WHERE id=?", (iso(utc_now()), thread_id))
            conn.commit()

    # turns -------------------------------------------------------------------

    def add_turn(self, thread_id: str, role: str, text: str, sources: list[Source] | None = None) -> Turn:
        import json

        turn = Turn(
            id=uuid.uuid4().hex[:16],
            thread_id=thread_id,
            role=role,
            text=text,
            created_at=iso(utc_now()),
            sources=sources or [],
        )
        payload = json.dumps([source.to_dict() for source in turn.sources])
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO turns(id, thread_id, role, text, created_at, sources) VALUES(?,?,?,?,?,?)",
                (turn.id, turn.thread_id, turn.role, turn.text, turn.created_at, payload),
            )
            if role == "user":
                conn.execute(
                    "INSERT INTO asks(text, turn_id, thread_id, created_at) VALUES(?,?,?,?)",
                    (text, turn.id, thread_id, turn.created_at),
                )
            conn.execute("UPDATE threads SET updated_at=? WHERE id=?", (turn.created_at, thread_id))
            conn.commit()
        return turn

    def recent_turns(self, thread_id: str, limit: int = 16) -> list[Turn]:
        import json

        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM turns WHERE thread_id=? ORDER BY rowid DESC LIMIT ?",
                (thread_id, limit),
            ).fetchall()
        turns = []
        for row in reversed(rows):
            raw = json.loads(row["sources"] or "[]")
            turns.append(
                Turn(
                    id=row["id"],
                    thread_id=row["thread_id"],
                    role=row["role"],
                    text=row["text"],
                    created_at=row["created_at"],
                    sources=[Source.from_dict(item) for item in raw],
                )
            )
        return turns

    def thread_turns(self, thread_id: str) -> list[dict]:
        return [turn.to_dict() for turn in self.recent_turns(thread_id, limit=200)]

    # memory ------------------------------------------------------------------

    def last_asks(
        self, limit: int = 6, exclude_thread: str | None = None, exclude_turn: str | None = None
    ) -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT a.text, a.created_at, a.thread_id, t.title AS thread_title "
                "FROM asks a LEFT JOIN threads t ON t.id=a.thread_id "
                "WHERE a.thread_id IS NOT ? AND a.turn_id IS NOT ? "
                "ORDER BY a.rowid DESC LIMIT ?",
                (exclude_thread, exclude_turn, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def matching_asks(
        self,
        query: str,
        limit: int = 6,
        exclude_thread: str | None = None,
        exclude_turn: str | None = None,
    ) -> list[dict]:
        tokens = _tokens(query)
        if not tokens:
            return []
        match = " OR ".join(f'"{token}"' for token in tokens)
        with self.connection() as conn:
            try:
                rows = conn.execute(
                    "SELECT a.text, a.created_at, a.thread_id, t.title AS thread_title "
                    "FROM asks a LEFT JOIN threads t ON t.id=a.thread_id "
                    "WHERE asks MATCH ? AND a.thread_id IS NOT ? AND a.turn_id IS NOT ? "
                    "ORDER BY bm25(asks) LIMIT ?",
                    (match, exclude_thread, exclude_turn, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [dict(row) for row in rows]

    def recall(
        self, query: str, thread_id: str | None = None, limit: int = 8, exclude_turn: str | None = None
    ) -> list[dict]:
        seen: set[str] = set()
        items: list[dict] = []
        for row in self.matching_asks(query, limit=limit, exclude_thread=thread_id, exclude_turn=exclude_turn):
            key = f"{row['created_at']}|{row['text']}"
            if key in seen:
                continue
            seen.add(key)
            items.append(row)
        for row in self.last_asks(limit=limit, exclude_thread=None, exclude_turn=exclude_turn):
            key = f"{row['created_at']}|{row['text']}"
            if key in seen:
                continue
            seen.add(key)
            items.append(row)
        items.sort(key=lambda row: row["created_at"], reverse=True)
        return items[:limit]

    def memory_block(
        self,
        query: str,
        thread_id: str | None,
        budget: int = 1600,
        exclude_turn: str | None = None,
        tz: str | None = None,
    ) -> tuple[str, list[dict]]:
        items = self.recall(query, thread_id=thread_id, exclude_turn=exclude_turn)
        if not items:
            return "", []
        lines = ["MEMORY (the user's earlier questions, with local time)"]
        used = []
        for item in items:
            text = " ".join(item["text"].split())
            if len(text) > 180:
                text = text[:177] + "…"
            line = f"- {format_time(item['created_at'], tz)} ({relative_day(item['created_at'], tz=tz)}) {text}"
            if sum(len(part) for part in lines) + len(line) > budget:
                break
            lines.append(line)
            used.append(item)
        if not used:
            return "", []
        return "\n".join(lines), used

    def timeline(self, limit: int = 100) -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT a.turn_id AS id, a.text, a.created_at, a.thread_id, t.title AS thread_title "
                "FROM asks a LEFT JOIN threads t ON t.id=a.thread_id "
                "ORDER BY a.rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    # search cache ------------------------------------------------------------

    def cache_get(self, key: str) -> dict | None:
        import json

        with self.connection() as conn:
            row = conn.execute(
                "SELECT payload, created_at FROM search_cache WHERE key=?", (key,)
            ).fetchone()
        if not row:
            return None
        created = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
        return {"payload": json.loads(row["payload"]), "created_at": created}

    def cache_put(self, key: str, payload: list[dict]) -> None:
        import json

        with self.connection() as conn:
            conn.execute(
                "INSERT INTO search_cache(key, payload, created_at) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET payload=excluded.payload, created_at=excluded.created_at",
                (key, json.dumps(payload), iso(utc_now())),
            )
            conn.commit()

    def cache_prune(self, older_than: datetime) -> None:
        with self.connection() as conn:
            conn.execute("DELETE FROM search_cache WHERE created_at < ?", (iso(older_than),))
            conn.commit()
