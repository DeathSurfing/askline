# Askline design

Askline is a chat agent with exactly one tool: web search. It answers questions in a
streaming column, cites the pages it used, and keeps a timestamped record of every
question the user asks so a later question such as "what did I ask about Lisbon last
week" can be answered from that record.

The design favours three properties over feature count:

1. **Speed.** The common turn is one model call. A search adds one HTTP round trip and
   a second model call, not an open-ended agent loop.
2. **Correctness.** The model may only use the search results it received, and every
   claim taken from a result is expected to carry that result's number. Numbers that do
   not map to a retrieved source are stripped before the answer is stored or shown.
3. **Memory.** Every user question is written to an append-only log with a UTC
   timestamp. Recall reads that log, not a summary written by the model.

## Goals and non-goals

The primary user is one person running the app on a laptop or a small server, talking to
whichever model endpoint they already pay for. The app should be openable in one command
and should not require an account, a vector database, a message queue, or a build step.

Non-goals: multi-user accounts, document upload and retrieval, autonomous multi-step
tasks, code execution, and long-running background work. One tool is a constraint, not a
starting point.

## System shape

```mermaid
flowchart LR
    B[Browser<br/>vanilla JS] -->|POST /api/chat<br/>SSE| F[FastAPI]
    F --> G[LangGraph<br/>agent -> tools -> agent]
    G -->|web_search| S[SearXNG<br/>meta-search in compose]
    S -.->|blocked or empty| S2[Bing -> DDG -> Mojeek]
    G -->|chat completions| L[llama.cpp server<br/>LFM3 1.7B on the GPU]
    L -.->|any compatible endpoint| L2[OpenAI or Anthropic<br/>hosted model]
    F --> M[(SQLite<br/>turns + FTS5 + cache)]
    G --> M
    M -->|MEMORY block| G
```

The browser holds no state that matters. Threads, turns, sources, and settings live in
SQLite, so a refresh or a second tab reconstructs the same view. The model is stateless:
the graph is compiled without a checkpointer, and each turn rebuilds its message list
from the log. This removes a class of bugs where a checkpoint and the visible transcript
disagree.

## A turn, end to end

```mermaid
sequenceDiagram
    participant U as Browser
    participant A as FastAPI
    participant D as SQLite
    participant G as LangGraph
    participant S as Search
    participant L as LLM

    U->>A: POST /api/chat {thread_id, message}
    A->>D: insert user turn (UTC timestamp)
    A-->>U: event: meta
    A->>D: recall(query, exclude current turn)
    A-->>U: event: memory
    A->>G: astream(state)
    G->>L: system + history + question
    alt model requests a search
        L-->>G: tool call
        G->>S: web_search(query)
        S-->>G: numbered results
        G-->>U: event: search_start, event: search
        G->>L: results as tool message
    end
    L-->>G: answer tokens
    G-->>U: event: token (repeat)
    A->>D: insert assistant turn + sources
    A-->>U: event: done
```

A turn is capped at two search rounds. The cap is enforced in two places: the routing
edge stops sending the model back to the tool node, and the agent node stops binding
tools once the cap is reached, so the model must produce a final answer from what it has.
This bounds latency and cost at roughly three model calls and two searches in the worst
case.

## Data model

```
threads(id, title, created_at, updated_at)
turns(id, thread_id, role, text, created_at, sources_json)
asks  -- FTS5 over user turn text with turn_id, thread_id, created_at
search_cache(key, payload, created_at)
settings(key, value)
```

`turns` is append-only. `asks` is a materialised index of the user half, written in the
same transaction as the turn, so recall and the transcript cannot drift. Ordering uses
SQLite `rowid` rather than the second-resolution timestamp, because two turns in the same
second must still have a stable order.

The `sources_json` column keeps the exact result set that was shown to the model with
that answer. Reopening a thread therefore reproduces the citations, even though the
search cache may have expired.

## Why SearXNG, with a scraped chain behind it

A search API key would add a second credential to configure and a vendor to trust. A
self-hosted SearXNG instance avoids both: it aggregates several engines server-side,
returns JSON, and runs as one more container beside the app. The service is published
only on `127.0.0.1`, so the instance is not an open proxy.

The app queries `SEARXNG_URL` first. If the instance is unreachable, returns an error,
or returns no relevant result, the same `SearchEngine.search` call falls through to the
public result pages of three engines, in order:

| Engine | Parser | Why it is in the chain |
| --- | --- | --- |
| SearXNG | `results[]` from `format=json` | Primary path; one local HTTP hop, six results, no vendor key |
| Bing | `li.b_algo`, `h2 a`, `p` | Most reliable public HTML at the time of writing; redirect links decode from `u=a1<base64url>` |
| DuckDuckGo HTML | `.result`, `a.result__a` | Independent index; redirect links decode from `uddg=` |
| Mojeek | `ul.results-standard li` | Third fallback when the first two are blocked |

Each result is normalised to title, URL, snippet, and backend. Results are deduplicated
by host plus path, internal engine links are dropped, and the first six survive. A
result also has to share a distinctive word with the query; filler such as "latest" was
enough for Bing to return unrelated news, so `relevant()` filters those pages and lets
the next backend answer. A miss is not fatal: the tool returns an error payload, the
model is told the search failed through that payload, and the answer can say so. Every
backend failing produces one `search` event with an `error` field rather than an
exception.

The cache is keyed by a hash of the normalised query and result count. Its TTL is ten
minutes, which is long enough that a follow-up question reuses the same pages and short
enough that "today" questions do not go stale. Cache hits are labelled `cached: true` in
the stream.

## Memory

Memory answers two different questions, and the implementation treats them differently.

*What did I ask, and when?* is a property of the log. The `MEMORY` block always includes
the six most recent asks with their local timestamps, regardless of the current query.
The model is instructed to answer history questions from that block with the dates as
written.

*What did I say about X?* is a retrieval problem. The current query is tokenised, stop
words are dropped, and the remaining terms are OR-ed into an FTS5 match over all user
turns. Porter stemming makes "install" match "installed". The top matches are merged with
the recent six, deduplicated, sorted newest first, and cut to a 1,600 character budget.

The current turn is excluded explicitly. Without that exclusion the newest ask is always
the question being answered, and it crowds out the history the model is meant to recall.

The block that reaches the model looks like this:

```
MEMORY (the user's earlier questions, with local time)
- 2026-11-10 14:32 (today) How do I install LangGraph on Python 3.12?
- 2026-11-09 21:04 (yesterday) What changed in LangGraph 1.0?
- 2026-11-03 09:12 (7 days ago) cheap flights to Lisbon in March
```

Timestamps are stored in UTC and rendered in an IANA zone supplied by the browser with
each chat request. `AGENT_TZ` remains the fallback for API clients. The browser value
matters because a container's own clock is UTC even when the host is not: without it the
model told a user in IST that their 19:04 question arrived at 13:04. An unknown zone is
rejected at the API boundary with a 400 rather than silently falling back.

Chromium reports `Asia/Calcutta`, a legacy alias that current tzdata builds no longer
ship as a file. The image installs the `tzdata` Python package, whose backward links
cover those names, so the alias resolves instead of failing the whole turn. That failure
was observed in the browser and is why the dependency is explicit rather than assumed.

A vector store was rejected. The corpus is one person's questions, the vocabulary is
narrow, and FTS5 runs in the same file as the log with no extra service. If recall
quality becomes the bottleneck, the `recall` method is the only thing that needs to
change.

## The graph

State:

| Field | Purpose |
| --- | --- |
| `messages` | LangChain messages, reduced with `add_messages` |
| `system` | The prompt built for this turn, including the memory block and local date |
| `sources` | Deduplicated, renumbered results accumulated across rounds |
| `rounds` | Completed search rounds, used by the cap |
| `handled_tools` | Tool call ids already parsed, so a replayed state cannot double count |

Nodes are `agent`, `tools`, and `collect`. `collect` parses the tool message JSON into
`sources` and increments `rounds`; keeping it separate means the tool node stays a plain
`ToolNode` and the parsing is testable on its own.

The model is built per request from the effective settings. That is cheap, and it means
a settings change applies to the next message without a restart.

A small model sometimes keeps looking for a tool after the search budget is spent, or
writes a tool call as plain text instead of a structured call. The graph handles both.
`has_tool_markup` and `parse_markup_tool_call` recover a call written as text on the
first round. At the cap, the agent node asks the model for an answer with a short closing
prompt that has no tools in it, and `results_summary` is the fallback if even that reply
is empty or still holds a call. The stream applies the same rule through
`_strip_stream_markup`, so raw markup never reaches the interface. This is deliberately
model-specific recovery, kept at the model boundary rather than spread through the UI.

## Provider abstraction

`ChatConfig` is the single boundary. It resolves, in order: values saved from the
settings dialog, then environment variables, then defaults. `build_chat_model` maps the
provider name to `ChatOpenAI` or `ChatAnthropic`, both of which accept a `base_url`, a
model id, a key, and tool definitions.

That covers OpenAI, Anthropic, OpenRouter, Groq, Together, Fireworks, vLLM, LM Studio,
and Ollama's OpenAI-compatible endpoint. A provider that speaks a different wire format
needs a new branch in one function, not changes to the graph, the memory, or the UI.

Keys saved through the UI are stored in the `settings` table in plain text. This is a
local single-user app, and the alternative, an OS keychain, would not work in a
container without extra plumbing. The key is never returned to the browser: the settings
endpoint reports `has_api_key` only.

## Streaming contract

`POST /api/chat` returns `text/event-stream`. Frames are `event: name\ndata: json\n\n`.

| Event | Payload | Consumer behaviour |
| --- | --- | --- |
| `meta` | thread_id, turn_id, created_at, model, provider | Bind the answer to a thread, show the model |
| `memory` | items with text, created_at, local_time | Show that recall happened |
| `search_start` | query | Show "Searching the web..." |
| `search` | results, cached, backend, error | Render source chips |
| `token` | delta | Append to the answer |
| `stats` | tokens, seconds, tok_per_s | Update the live rate, at most every 250 ms |
| `reset` | (none) | Clear streamed text that the server replaced |
| `done` | text, sources, cited, elapsed_ms, tokens, tok_per_s | Finalise, show sources and timing |
| `error` | message | Inline error with retry |

The stream is produced with `graph.astream(..., stream_mode=["messages", "updates"])`.
`messages` carries token chunks; `updates` carries node outputs, which is where tool
results appear. That pairing gives live tokens and structured search events from the same
run without a second model call or a callback side channel.

## Answer shaping

The system prompt fixes the shape of the answer: lead with the answer, then the
supporting facts, four to eight sentences, no preamble. It also states the current local
date and time, which prevents "latest" questions from being answered against an old
training cut-off, and it makes the memory rule conditional on a memory block existing.

Citation handling is deliberately strict. The browser links `[n]` to result `n` only when
that result exists. `strip_invalid` removes numbers outside the retrieved range before
the answer is stored. The stored text and the rendered text therefore agree, and a model
that invents `[7]` with six sources leaves no dead link behind.

## Failure modes

| Failure | Behaviour |
| --- | --- |
| No API key | The chat endpoint returns 400 with a message that names Settings |
| Model endpoint down | SSE `error` event; the user turn stays in the log |
| All search backends blocked | Tool returns an error payload; the model is told and can say what is unknown |
| Model times out | 120 second turn timeout, then an SSE `error` |
| Model writes a tool call as text | The graph parses it, or forces an answer; the stream hides the markup |
| Model returns no text | SSE `error` names the empty answer, no assistant turn is stored |
| Stream disconnects | The async generator closes; the turn is not stored unless `done` ran |
| SQLite locked | Ten second busy timeout; each operation uses its own connection |

## Testing

The test suite runs without a model and without the network. Two seams make that
possible:

* `build_chat_model` is replaced with `ScriptedModel`, which returns a fixed sequence of
  `AIMessage` objects, including tool calls. Graph routing, the search cap, source
  merging, SSE framing, and persistence are all exercised through the real FastAPI app.
* `SearchEngine` accepts an `httpx` transport. `MockTransport` serves saved Bing and
  DuckDuckGo markup, so parser and fallback behaviour are tested against real page
  structure, and the cache path is tested without waiting for a network.

The live checks that the suite cannot replace are a real model call, a real search, and a
browser turn. Those are run manually and recorded in the README.

## Verification

Measured on the deployment host: a llama.cpp server on an RTX 3070 laptop GPU with LFM3
1.7B Q8, SearXNG, and a browser-shaped chat request. The turn was "What changed in the
latest Python release?".

| Stage | Time |
| --- | --- |
| SSE headers and `meta` event | 0.02 s |
| Model decision to call the tool | 1.7 s |
| Search results, cached | 1.7 s |
| First streamed token | 8.6 s |
| Complete turn, `done` event | 9.9 s |
| Generation rate | 134 tok/s |

Most of the wait is the model's reasoning phase, which llama.cpp does not stream. The
answer itself arrives quickly once it starts. The app's own stages stay small: `meta`
goes out before any model work, the search is a container round trip, and the cache
serves repeats.

A separate measurement decided the runtime. On the same GPU, Ollama generated 3.7 tok/s
for the same class of model, while a 120-token llama.cpp completion ran at 165 tok/s
(1785 MHz, 56 W). The stack therefore runs llama.cpp directly with the NVIDIA devices
and driver libraries mounted, and falls back to CPU when those are absent.

The test suite covers the graph, parsers, cache, API, citations, and memory without a
model or network. The container run is verified separately with `./run.sh up`, a health
check, a SearXNG JSON request, and one real chat turn.

## Suggested questions

`GET /api/prompts` returns three questions for the empty state. It reads the user's last
twenty questions, asks the model for three likely follow-ups, and falls back to the
static defaults when there is too little history, the model is not configured, or the
call fails within eight seconds. The result is cached in the `settings` table under the
current ask count, so a new turn regenerates the suggestions and repeated page loads do
not. The prompt asks for one line per question with no numbering; `parse_suggestions`
still strips bullets and numbers because small models add them anyway.

This keeps the home screen tied to the user's actual work without adding a second tool or
a background job. The trade is one extra model call after the first turn, hidden behind a
fallback.

## Answer shapes and subagents

A turn has two independent switches. **Visual** adds a generated interface below the prose.
**Delegate** researches through a subagent tree before the answer is written. Both start on
in the interface; the API defaults to off so every existing client keeps the plain path.

The delegate tree is deliberately small. `MAX_DEPTH = 3` counts the root's children as
depth one, `MAX_CHILDREN = 3` bounds each node, and `MAX_NODES = 7` bounds the whole run.
A planner returns up to three briefs with one of three roles. Research nodes search, write
findings, and may spawn follow-ups. Design and critique nodes work from the parent's
findings. The root model then writes one cited answer from all findings. The tree streams
as `task` and `task_done` events and is stored in `turns.trace`, so reopening a thread shows
the same trace.

Follow-ups are resolved before planning. When history exists, `resolve_question` asks for a
standalone search question built from the earlier turns, and that question drives the
briefs. The final synthesis still receives the user's original wording and the conversation
so the answer responds to what was actually asked.

The trade is latency. A plain turn is one or two model calls. A delegated turn is a query
resolution call, a planner call, a call per node, and a synthesis call, so the measured
turn grows from about ten seconds to about thirty on the laptop GPU. The depth cap keeps
the worst case bounded, and the UI shows each node as it lands.

## The OpenUI lab and Visual mode

The chat answer is prose. The OpenUI lab answers the same question as a small interface:
a card with a header, the answer, structured facts, follow-up buttons, sources, and an
optional chart. Visual mode composes the same card from an answer that delegation already
produced, so turning Visual on does not add a search.

The renderer is [OpenUI](https://www.openui.com/) with the shadcn design system. The island
is built once with Vite into `app/static/vendor/shadcn/askline-shadcn.js` and
`askline-shadcn.css`, and the page loads them as plain script and style tags, so the app
still has no build step. next-themes inside the island receives the resolved app theme,
which keeps the card's tokens in step with the page in light and dark mode. The app's own
tokens come from the same shadcn Tailwind primitive sheet, so there is one palette rather
than a second set of colors.

The first attempt asked the model to write OpenUI Lang. LFM3 1.7B failed in four ways:
it rambled past the program, repeated `root`, wrote named arguments where the language
takes positional ones, and emitted multiline arrays that the line-oriented parser cannot
read. The fix is constrained generation. The model returns JSON with a title, subtitle,
answer, items, two follow-ups, and an optional chart. llama.cpp enforces the schema with
`response_format`, so the reply is always valid JSON. `app/openui.py` then composes the
program with the shadcn library's positional signatures: `CardHeader`, `MarkDownRenderer`,
`Table`, `BarChart` or `LineChart` with `Series`, `FollowUpBlock`, and `Card`. A prose
answer that already names a year and a percentage per year can also become a line chart
through `extract_series`, so a dashboard request degrades to a real chart when the model
supplies the numbers.

The card has no `sources` argument because the shadcn `Card` accepts only children. The
browser renders the source chips from the same numbered list the answer cites, so the
strip and the citations still agree.

This is a deliberate trade. A larger model could write the language directly and produce
freer layouts. With a 1.7B model on a laptop GPU, a valid interface in about three seconds
beats an ambitious one that fails. The JSON step also makes the sources trustworthy,
because the model never writes a URL.

## Rejected alternatives

* **LangChain's `AgentExecutor`.** Its loop is unbounded and its intermediate steps are
  harder to stream as typed events. A three-node `StateGraph` is smaller and predictable.
* **A LangGraph checkpointer.** It would duplicate the transcript that SQLite already
  holds. One source of truth is easier to reason about and to query for memory.
* **A frontend framework and bundler for the app shell.** The UI is one page and one
  stream. The generated card is the exception: OpenUI and shadcn are built once into
  vendored assets, because reproducing that component library by hand would be more code
  than the whole chat page.
* **A search API with a key.** It would improve result quality slightly and cost setup
  friction, a second secret, and a vendor dependency. The chain is good enough and can
  be replaced behind `SearchEngine.search`.
* **HTMX or server-rendered fragments.** Token streaming and a memory drawer need
  client state. Vanilla JS is smaller than either framework at this size.
