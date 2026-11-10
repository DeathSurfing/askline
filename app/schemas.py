from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class Source:
    index: int
    title: str
    url: str
    snippet: str
    backend: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Source":
        return cls(
            index=int(data.get("index", 0)),
            title=str(data.get("title", "")),
            url=str(data.get("url", "")),
            snippet=str(data.get("snippet", "")),
            backend=str(data.get("backend", "")),
        )


@dataclass
class Turn:
    id: str
    thread_id: str
    role: str
    text: str
    created_at: str
    sources: list[Source] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["sources"] = [source.to_dict() for source in self.sources]
        return data


@dataclass
class SearchOutcome:
    query: str
    results: list[Source]
    cached: bool
    backend: str
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "results": [source.to_dict() for source in self.results],
            "cached": self.cached,
            "backend": self.backend,
            "error": self.error,
        }
