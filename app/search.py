from __future__ import annotations

import hashlib
import html
import json
import os
import re
from base64 import urlsafe_b64decode
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, quote, quote_plus, unquote, urlparse

import httpx
from selectolax.lexbor import LexborHTMLParser as HTMLParser

from .schemas import SearchOutcome, Source, iso

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
CACHE_TTL = timedelta(minutes=10)
_WORD = re.compile(r"[a-z0-9][a-z0-9+#.-]*")
GENERIC = frozenset(
    "latest new news today current release releases update updates feature features best top "
    "what when where who why how is are was were the a an about and or for of to in on with".split()
)


def _words(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def relevant(results: list[Source], query: str) -> list[Source]:
    """Drop results that share no distinctive word with the query.

    Engines sometimes latch onto a generic word such as "latest" and return
    unrelated news. A result must match at least one word the query would still
    contain after filler is removed. Queries made only of filler keep every result.
    """
    salient = _words(query) - GENERIC
    if not salient:
        return results
    return [result for result in results if salient & _words(f"{result.title} {result.snippet} {result.url}")]


def _clean(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


def _decode_bing(href: str) -> str:
    if "bing.com/ck/a" not in href:
        return href
    params = parse_qs(urlparse(href).query)
    raw = (params.get("u") or [""])[0]
    if not raw.startswith("a1"):
        return ""
    encoded = raw[2:]
    padded = encoded + "=" * (-len(encoded) % 4)
    try:
        return urlsafe_b64decode(padded).decode("utf-8", "ignore")
    except Exception:
        return ""


def _decode_ddg(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    if "duckduckgo.com/l/" in href:
        params = parse_qs(urlparse(href).query)
        return unquote((params.get("uddg") or [""])[0])
    return href


def _usable(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    if not host or host.endswith("bing.com") or host.endswith("duckduckgo.com"):
        return False
    return url.startswith("http://") or url.startswith("https://")


def _is_web_url(url: str) -> bool:
    return url.startswith(("http://", "https://")) and bool(urlparse(url).netloc)


def parse_searxng(payload: str) -> list[Source]:
    """Read the JSON that SearXNG returns for format=json."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    results: list[Source] = []
    for item in data.get("results") or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not _is_web_url(url):
            continue
        results.append(
            Source(
                index=0,
                title=_clean(str(item.get("title") or "")),
                url=url,
                snippet=_clean(str(item.get("content") or item.get("snippet") or "")),
                backend="searxng",
            )
        )
    return results


def searxng_url() -> str:
    return (os.environ.get("SEARXNG_URL") or "").strip().rstrip("/")


def parse_bing(markup: str) -> list[Source]:
    tree = HTMLParser(markup)
    results: list[Source] = []
    for block in tree.css("li.b_algo"):
        link = block.css_first("h2 a")
        if not link:
            continue
        url = _decode_bing(link.attributes.get("href", ""))
        if not _usable(url):
            continue
        snippet_node = block.css_first("p") or block.css_first(".b_caption")
        results.append(
            Source(
                index=0,
                title=_clean(link.text(strip=True)),
                url=url,
                snippet=_clean(snippet_node.text(strip=True) if snippet_node else ""),
                backend="bing",
            )
        )
    return results


def parse_ddg(markup: str) -> list[Source]:
    tree = HTMLParser(markup)
    results: list[Source] = []
    for block in tree.css(".result"):
        link = block.css_first("a.result__a")
        if not link:
            continue
        url = _decode_ddg(link.attributes.get("href", ""))
        if not _usable(url):
            continue
        snippet_node = block.css_first(".result__snippet")
        results.append(
            Source(
                index=0,
                title=_clean(link.text(strip=True)),
                url=url,
                snippet=_clean(snippet_node.text(strip=True) if snippet_node else ""),
                backend="duckduckgo",
            )
        )
    return results


def parse_mojeek(markup: str) -> list[Source]:
    tree = HTMLParser(markup)
    results: list[Source] = []
    for block in tree.css("ul.results-standard li"):
        link = block.css_first("a.title") or block.css_first("h2 a")
        if not link:
            continue
        url = link.attributes.get("href", "")
        if not _usable(url):
            continue
        snippet_node = block.css_first("p.s") or block.css_first("p")
        results.append(
            Source(
                index=0,
                title=_clean(link.text(strip=True)),
                url=url,
                snippet=_clean(snippet_node.text(strip=True) if snippet_node else ""),
                backend="mojeek",
            )
        )
    return results


def parse_wikipedia(payload: dict, extract: str = "") -> list[Source]:
    pages = payload.get("query", {}).get("search", [])
    results: list[Source] = []
    for item in pages:
        title = item.get("title", "").strip()
        if not title:
            continue
        snippet = extract if extract and not results else re.sub(r"<[^>]+>", "", item.get("snippet", ""))
        results.append(
            Source(
                index=0,
                title=title,
                url=f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                snippet=" ".join(snippet.split())[:2600],
                backend="wikipedia",
            )
        )
    return results


def wants_current(query: str) -> bool:
    return bool(re.search(r"(?i)\b(latest|current|today|news|price|release|version|now|2026)\b", query))


def dedupe(results: list[Source], limit: int) -> list[Source]:
    seen: set[str] = set()
    unique: list[Source] = []
    for result in results:
        parsed = urlparse(result.url)
        key = f"{parsed.netloc.lower()}{parsed.path.rstrip('/')}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            Source(
                index=len(unique) + 1,
                title=result.title[:180] or parsed.netloc,
                url=result.url,
                snippet=result.snippet[:400],
                backend=result.backend,
            )
        )
        if len(unique) >= limit:
            break
    return unique


class SearchEngine:
    """One web search tool. SearXNG when configured, then Bing, DuckDuckGo, and Mojeek."""

    def __init__(
        self,
        store,
        timeout: float = 7.0,
        ttl: timedelta = CACHE_TTL,
        transport=None,
        url: str | None = None,
    ):
        self.store = store
        self.timeout = timeout
        self.ttl = ttl
        self.transport = transport
        self.searxng_url = (url if url is not None else searxng_url()).rstrip("/")
        self.last_backend = ""
        self.last_error = ""

    def _cache_key(self, query: str, count: int) -> str:
        return hashlib.sha256(f"v1|{query.strip().lower()}|{count}".encode()).hexdigest()

    def _finished(self, key: str, query: str, results: list[Source], backend: str) -> SearchOutcome:
        self.last_backend = backend
        self.last_error = ""
        self.store.cache_put(key, [source.to_dict() for source in results])
        return SearchOutcome(query=query, results=results, cached=False, backend=backend)

    async def search(self, query: str, count: int = 6) -> SearchOutcome:
        query = " ".join(query.split())
        if not query:
            return SearchOutcome(query=query, results=[], cached=False, backend="", error="empty query")
        key = self._cache_key(query, count)
        cached = self.store.cache_get(key)
        if cached and datetime.now(timezone.utc) - cached["created_at"] < self.ttl:
            results = [Source.from_dict(item) for item in cached["payload"]]
            return SearchOutcome(query=query, results=results, cached=True, backend=results[0].backend if results else "")

        errors: list[str] = []
        wikipedia_first = not wants_current(query)
        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
            transport=self.transport,
        ) as client:
            async def wikipedia() -> list[Source]:
                response = await client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": count},
                )
                if response.status_code >= 400:
                    return []
                payload = response.json()
                pages = payload.get("query", {}).get("search", [])
                extract = ""
                if pages:
                    detail = await client.get(
                        "https://en.wikipedia.org/w/api.php",
                        params={
                            "action": "query",
                            "prop": "extracts",
                            "explaintext": "1",
                            "titles": pages[0]["title"],
                            "format": "json",
                            "redirects": "1",
                        },
                    )
                    if detail.status_code < 400:
                        blocks = detail.json().get("query", {}).get("pages", {})
                        extract = next(iter(blocks.values()), {}).get("extract", "")[:2600]
                return relevant(dedupe(parse_wikipedia(payload, extract), count), query)

            if wikipedia_first:
                try:
                    results = await wikipedia()
                    if results:
                        return self._finished(key, query, results, "wikipedia")
                    errors.append("wikipedia: no results")
                except Exception as exc:
                    errors.append(f"wikipedia: {type(exc).__name__}")

            if self.searxng_url:
                try:
                    response = await client.get(
                        f"{self.searxng_url}/search",
                        params={"q": query, "format": "json", "language": "en"},
                        timeout=8.0,
                    )
                    if response.status_code >= 400:
                        errors.append(f"searxng: HTTP {response.status_code}")
                    else:
                        results = relevant(dedupe(parse_searxng(response.text), count), query)
                        if results:
                            return self._finished(key, query, results, "searxng")
                        errors.append("searxng: no relevant results")
                except Exception as exc:
                    errors.append(f"searxng: {type(exc).__name__}")

            backends = (
                ("bing", f"https://www.bing.com/search?q={quote_plus(query)}&count={count}&setlang=en", parse_bing),
                ("duckduckgo", f"https://html.duckduckgo.com/html/?q={quote_plus(query)}", parse_ddg),
                ("mojeek", f"https://www.mojeek.com/search?q={quote_plus(query)}", parse_mojeek),
            )
            if not wikipedia_first:
                try:
                    results = await wikipedia()
                    if results:
                        return self._finished(key, query, results, "wikipedia")
                    errors.append("wikipedia: no results")
                except Exception as exc:
                    errors.append(f"wikipedia: {type(exc).__name__}")

            for name, url, parser in backends:
                try:
                    response = await client.get(url)
                    if response.status_code >= 400:
                        errors.append(f"{name}: HTTP {response.status_code}")
                        continue
                    results = relevant(dedupe(parser(response.text), count), query)
                    if not results:
                        errors.append(f"{name}: no relevant results")
                        continue
                    return self._finished(key, query, results, name)
                except Exception as exc:  # network errors and parser failures share one retry path
                    errors.append(f"{name}: {type(exc).__name__}")
        self.last_error = "; ".join(errors)
        return SearchOutcome(query=query, results=[], cached=False, backend="", error=self.last_error)

    def prune(self) -> None:
        self.store.cache_prune(datetime.now(timezone.utc) - timedelta(days=7))
