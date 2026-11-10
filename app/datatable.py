"""Turn a reference table into the numbers a chart needs."""

from __future__ import annotations

import re

import httpx
from selectolax.lexbor import LexborHTMLParser

from .schemas import Source
from .search import WIKI_AGENT, search_terms

YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
CITE = re.compile(r"\[[^\]]*\]")
WORD = re.compile(r"[a-z][a-z0-9-]{3,}")


def keywords(question: str) -> set[str]:
    return {word for word in WORD.findall(question.lower()) if word not in {"the", "and", "for", "with", "from", "dashboard"}}


PLACE = re.compile(r"(?i)\b(united states|u\.s\.a?|us|usa|america)\b")


def place(question: str) -> str | None:
    return "United States" if PLACE.search(question) else None


def in_place(title: str, expected: str | None) -> bool:
    if expected is None:
        return True
    lowered = title.lower().replace("_", " ")
    return any(word in lowered for word in ("united states", "american", "u.s.", "usa", "america"))


def header_label(text: str) -> str:
    clean = CITE.sub("", text).replace("%", " % ")
    return re.sub(r"\s+", " ", clean).strip()


def value_column(headers: list[str], words: set[str]) -> int | None:
    best: tuple[int, int] | None = None
    for index, header in enumerate(headers):
        if "%" not in header:
            continue
        score = len(words & set(WORD.findall(header.lower())))
        if best is None or score > best[0]:
            best = (score, index)
    return None if best is None else best[1]


def number(text: str) -> float | None:
    match = NUMBER.search(text.replace(",", ""))
    return None if match is None else float(match.group(0))


def parse_series(html: str, question: str, title: str = "") -> dict | None:
    """Pick the first table with a year column and a percentage column."""
    tree = LexborHTMLParser(html)
    if title and not in_place(title, place(question)):
        return None
    words = keywords(question)
    for table in tree.css("table.wikitable"):
        rows = [row for row in table.css("tr") if row.css("th,td")]
        if len(rows) < 4:
            continue
        headers = [header_label(cell.text(strip=True)) for cell in rows[0].css("th,td")]
        column = value_column(headers, words)
        if column is None:
            continue
        pairs: dict[int, float] = {}
        for row in rows[1:]:
            cells = row.css("th,td")
            if len(cells) <= column:
                continue
            year = YEAR.search(cells[0].text(strip=True))
            value = number(cells[column].text(strip=True))
            if year and value is not None:
                pairs.setdefault(int(year.group(1)), value)
        ordered = sorted(pairs.items())
        if len(ordered) < 3:
            continue
        return {
            "kind": "line",
            "title": headers[column][:60] or "Reported trend",
            "labels": [str(year) for year, _ in ordered],
            "values": [value for _, value in ordered],
            "unit": "%",
        }
    return None


def score(page: Source, words: set[str]) -> int:
    return len(words & set(WORD.findall(f"{page.title} {page.url}".lower())))


async def series_from_sources(question: str, sources: list[Source], timeout: float = 8.0) -> dict | None:
    words = keywords(question)
    pages = sorted(
        (item for item in sources if "wikipedia.org/wiki/" in item.url),
        key=lambda item: score(item, words),
        reverse=True,
    )[:3]
    if not pages:
        return None
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for page in pages:
            try:
                response = await client.get(page.url, headers={"User-Agent": WIKI_AGENT})
                response.raise_for_status()
            except httpx.HTTPError:
                continue
            chart = parse_series(response.text, question, page.title)
            if chart:
                return chart
    return None


async def wiki_pages(question: str, client: httpx.AsyncClient) -> list[str]:
    response = await client.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": search_terms(question),
            "format": "json",
            "srlimit": 3,
        },
        headers={"User-Agent": WIKI_AGENT, "Accept": "application/json"},
    )
    response.raise_for_status()
    hits = response.json().get("query", {}).get("search", [])
    return [f"https://en.wikipedia.org/wiki/{hit['title'].replace(' ', '_')}" for hit in hits]


async def series_for_question(question: str, sources: list[Source], timeout: float = 8.0) -> dict | None:
    """Find a chartable table even when the search results missed the reference page."""
    chart = await series_from_sources(question, sources, timeout)
    if chart:
        return chart
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        try:
            pages = await wiki_pages(question, client)
        except httpx.HTTPError:
            return None
        for url in pages:
            try:
                response = await client.get(url, headers={"User-Agent": WIKI_AGENT})
                response.raise_for_status()
            except httpx.HTTPError:
                continue
            chart = parse_series(response.text, question, url)
            if chart:
                return chart
    return None
