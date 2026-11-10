/* Askline client. Classic script, no modules.
   Server text only reaches the DOM via textContent or element nodes, so the
   markdown renderer builds nodes instead of concatenating HTML. */
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const narrowQuery = window.matchMedia('(max-width: 899px)');
  const reducedQuery = window.matchMedia('(prefers-reduced-motion: reduce)');

  const els = {
    menu: $('menu-btn'), sidebar: $('sidebar'), newChat: $('new-chat-btn'),
    threadList: $('thread-list'), threadsEmpty: $('threads-empty'),
    model: $('model-chip'), banner: $('key-banner'), bannerSettings: $('banner-settings-btn'),
    settingsBtn: $('settings-btn'), memoryBtn: $('memory-btn'), memory: $('memory'),
    memoryTitle: $('memory-title'), memoryClose: $('memory-close'), memoryBody: $('memory-body'),
    shell: $('shell'), scrim: $('scrim'), notice: $('notice'), transcript: $('transcript'),
    empty: $('empty'), examples: $('examples'), turns: $('turns'), composer: $('composer'), prompt: $('prompt'), send: $('send-btn'),
    dialog: $('settings-dialog'), form: $('settings-form'), dialogStatus: $('settings-status'),
    test: $('test-btn'), save: $('save-btn'), cancel: $('cancel-btn'),
    modelHint: $('model-hint'), modelOptions: $('model-options'),
    f: {
      provider: $('f-provider'), baseUrl: $('f-base-url'), model: $('f-model'), apiKey: $('f-api-key'),
      temperature: $('f-temp'), maxTokens: $('f-max'),
    },
  };

  const state = { threadId: null, threads: [], busy: false, memoryOpen: false, sidebarOpen: false, uid: 0, noticeTimer: 0 };

  // ---------- formatting ----------

  const stampFmt = new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' });
  const clockFmt = new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' });
  const dayFmt = new Intl.DateTimeFormat(undefined, { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' });
  const relFmt = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' });
  const UNITS = [['year', 31536000], ['month', 2592000], ['week', 604800], ['day', 86400], ['hour', 3600], ['minute', 60]];

  const stamp = (iso) => (Number.isNaN(Date.parse(iso)) ? '' : stampFmt.format(new Date(iso)));
  const dayKey = (date) => `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}`;

  function relTime(iso) {
    const seconds = (Date.parse(iso) - Date.now()) / 1000;
    if (!Number.isFinite(seconds) || Math.abs(seconds) < 60) return 'just now';
    const [unit, size] = UNITS.find(([, s]) => Math.abs(seconds) >= s);
    return relFmt.format(Math.round(seconds / size), unit);
  }

  // Groups by local calendar day, so an ask made late at night stays on its local day.
  function dayHeading(date) {
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    const key = dayKey(date);
    if (key === dayKey(new Date())) return `Today · ${dayFmt.format(date)}`;
    if (key === dayKey(yesterday)) return `Yesterday · ${dayFmt.format(date)}`;
    return dayFmt.format(date);
  }

  const formatElapsed = (ms) =>
    ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;

  function hostOf(url) {
    try {
      return new URL(url).hostname.replace(/^www\./, '');
    } catch {
      return '';
    }
  }

  const sortThreads = (list) => [...list].sort((a, b) => Date.parse(b.updated_at) - Date.parse(a.updated_at));

  // ---------- DOM, network, and feedback helpers ----------

  // Attr keys: text sets textContent, class sets className, on<event> binds a listener.
  // null, undefined, and false are skipped; true becomes a boolean attribute.
  function h(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value === null || value === undefined || value === false) continue;
      if (key === 'text') node.textContent = value;
      else if (key === 'class') node.className = value;
      else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? '' : String(value));
    }
    node.append(...[].concat(children).filter((c) => c !== null && c !== undefined && c !== false));
    return node;
  }

  function errorMessage(data, status) {
    const detail = data && (data.error || data.message || data.detail);
    return typeof detail === 'string' && detail ? detail : `Request failed (${status}).`;
  }

  async function api(path, { method = 'GET', body } = {}) {
    const init = { method, headers: { Accept: 'application/json' } };
    if (body !== undefined) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(body);
    }
    const res = await fetch(path, init);
    const data = await res.json().catch(() => null);
    if (!res.ok) throw new Error(errorMessage(data, res.status));
    return data;
  }

  function notify(message) {
    els.notice.textContent = message;
    els.notice.hidden = false;
    clearTimeout(state.noticeTimer);
    state.noticeTimer = setTimeout(() => { els.notice.hidden = true; }, 6000);
  }

  // Pins the transcript to the bottom only when the reader is already there.
  function followIfPinned(update) {
    const t = els.transcript;
    const pinned = t.scrollHeight - t.scrollTop - t.clientHeight < 120;
    const result = update();
    if (pinned) t.scrollTop = t.scrollHeight;
    return result;
  }

  const scrollToEnd = () => { els.transcript.scrollTop = els.transcript.scrollHeight; };
  const updateEmpty = () => { els.empty.hidden = els.turns.childElementCount > 0; };

  function flash(button, label) {
    button.dataset.label ||= button.textContent;
    button.textContent = label;
    clearTimeout(Number(button.dataset.timer));
    button.dataset.timer = String(setTimeout(() => { button.textContent = button.dataset.label; }, 1600));
  }

  // ---------- markdown subset (built as DOM nodes) ----------

  const FENCE_RE = /^\s*```/;
  const UL_RE = /^\s*[-*+]\s+(.*)$/;
  const OL_RE = /^\s*\d{1,3}[.)]\s+(.*)$/;
  const INLINE_RE = /`([^`\n]+)`|\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)|\*\*([^*\n]+)\*\*|\*([^*\n]+)\*|\[(\d{1,2})\]/g;
  const startsBlock = (line) => FENCE_RE.test(line) || UL_RE.test(line) || OL_RE.test(line);

  function renderMarkdown(text, ctx) {
    const lines = text.replace(/\r\n?/g, '\n').split('\n');
    const nodes = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (FENCE_RE.test(line)) {
        // An unclosed fence runs to the end of the text, which suits a streaming answer.
        const code = [];
        for (i += 1; i < lines.length && !FENCE_RE.test(lines[i]); i += 1) code.push(lines[i]);
        i += 1;
        nodes.push(h('pre', {}, [h('code', { text: code.join('\n') })]));
      } else if (UL_RE.test(line) || OL_RE.test(line)) {
        const re = UL_RE.test(line) ? UL_RE : OL_RE;
        const list = h(re === UL_RE ? 'ul' : 'ol');
        for (; i < lines.length && re.test(lines[i]); i += 1) {
          list.append(h('li', {}, renderInline(lines[i].match(re)[1], ctx)));
        }
        nodes.push(list);
      } else if (!line.trim()) {
        i += 1;
      } else {
        const paragraph = h('p');
        for (let first = true; i < lines.length && lines[i].trim() && !startsBlock(lines[i]); i += 1) {
          if (!first) paragraph.append(h('br'));
          first = false;
          paragraph.append(...renderInline(lines[i], ctx));
        }
        nodes.push(paragraph);
      }
    }
    return nodes;
  }

  function renderInline(text, ctx) {
    const out = [];
    let last = 0;
    for (const m of text.matchAll(INLINE_RE)) {
      if (m.index > last) out.push(text.slice(last, m.index));
      if (m[1] !== undefined) out.push(h('code', { text: m[1] }));
      else if (m[2] !== undefined) out.push(h('a', { href: m[3], target: '_blank', rel: 'noopener noreferrer', text: m[2] }));
      else if (m[4] !== undefined) out.push(h('strong', { text: m[4] }));
      else if (m[5] !== undefined) out.push(h('em', { text: m[5] }));
      else out.push(citation(m[6], m[0], ctx));
      last = m.index + m[0].length;
    }
    if (last < text.length) out.push(text.slice(last));
    return out;
  }

  // "[n]" becomes a superscript link to chip n. With no matching source it stays literal.
  function citation(num, raw, ctx) {
    const source = ctx.sources.get(Number(num));
    if (!source) return raw;
    return h('sup', {}, [h('a', {
      class: 'cite',
      href: `#${source.domId}`,
      'aria-label': `Source ${num}`,
      text: num,
      onclick: (event) => {
        event.preventDefault();
        const node = document.getElementById(source.domId);
        ctx.openSources();
        node?.scrollIntoView({ block: 'nearest', behavior: reducedQuery.matches ? 'auto' : 'smooth' });
        node?.focus({ preventScroll: true });
      },
    })]);
  }

  // ---------- turns ----------

  const sourceId = (uid, index) => `src-${uid}-${index}`;

  function userTurn(text, iso) {
    return h('article', { class: 'turn turn--user' }, [
      h('div', { class: 'bubble', text }),
      h('time', { class: 'turn__time', datetime: iso, text: stamp(iso) }),
    ]);
  }

  function assistantView(iso) {
    const view = { uid: ++state.uid, text: '', sources: [], cited: null, frame: 0 };
    view.recall = h('details', { class: 'recall', hidden: true });
    view.status = h('p', { class: 'turn__status', 'aria-live': 'polite', hidden: true });
    view.body = h('div', { class: 'turn__body' });
    view.sourceList = h('details', { class: 'sources', hidden: true });
    view.elapsed = h('span', { class: 'turn__elapsed' });
    view.rate = h('span', { class: 'turn__rate', hidden: true });
    view.copy = h('button', {
      type: 'button',
      class: 'btn-ghost btn-small',
      'aria-label': 'Copy answer',
      hidden: true,
      text: 'Copy',
      onclick: () => copyAnswer(view),
    });
    view.article = h('article', { class: 'turn turn--assistant', 'aria-busy': 'true' }, [
      view.recall,
      view.status,
      view.body,
      view.sourceList,
      h('footer', { class: 'turn__foot' }, [h('time', { class: 'turn__time', datetime: iso, text: stamp(iso) }), view.elapsed, view.rate, view.copy]),
    ]);
    return view;
  }

  function setStatus(view, text, kind = '') {
    view.status.dataset.kind = kind;
    view.status.hidden = !text;
    view.status.replaceChildren(...(kind === 'progress' ? [h('span', { class: 'spinner', 'aria-hidden': 'true' })] : []), text);
  }

  const clearProgress = (view) => { if (view.status.dataset.kind === 'progress') setStatus(view, ''); };

  function renderAnswer(view) {
    const sources = new Map(view.sources.map((s) => [s.index, { domId: sourceId(view.uid, s.index) }]));
    const openSources = () => { view.sourceList.open = true; };
    view.body.replaceChildren(...renderMarkdown(view.text, { sources, openSources }));
  }

  function renderSources(view) {
    const items = view.sources.map((source) => {
      const dim = view.cited && !view.cited.has(source.index);
      const host = hostOf(source.url);
      return h('li', { class: dim ? 'source source--uncited' : 'source' }, [
        h('a', { href: source.url, target: '_blank', rel: 'noopener noreferrer', id: sourceId(view.uid, source.index) }, [
          h('span', { class: 'source__n', text: String(source.index) }),
          h('span', { class: 'source__title', text: source.title || source.url }),
          host ? h('span', { class: 'source__host', text: host }) : null,
        ]),
      ]);
    });
    view.sourceList.replaceChildren(
      h('summary', { text: `${items.length} ${items.length === 1 ? 'source' : 'sources'}` }),
      h('ul', {}, items),
    );
    view.sourceList.hidden = items.length === 0;
  }

  function showRecall(view, items) {
    if (!items.length) return;
    view.recall.replaceChildren(
      h('summary', { text: `${items.length} earlier ${items.length === 1 ? 'question' : 'questions'}` }),
      h('ul', {}, items.map((item) => h('li', { text: `${item.text} · ${relTime(item.created_at)}` }))),
    );
    view.recall.hidden = false;
  }

  function onSearch(view, data) {
    const results = data.results || [];
    if (results.length) {
      view.sources = results;
      clearProgress(view);
      renderSources(view);
    } else {
      setStatus(view, data.error ? `Web search unavailable: ${data.error}` : 'No web results found.', 'note');
    }
  }

  // Shared by a live "done" event and a stored turn when a thread reopens.
  function completeView(view, { text, sources, cited, elapsed_ms: elapsedMs, tok_per_s: tokPerS }) {
    cancelAnimationFrame(view.frame);
    view.frame = 0;
    view.text = text || view.text;
    if (sources?.length) view.sources = sources;
    view.cited = Array.isArray(cited) ? new Set(cited) : null;
    setStatus(view, '');
    renderAnswer(view);
    if (!view.text.trim()) view.body.replaceChildren(h('p', { class: 'field__hint', text: 'The model returned no text.' }));
    renderSources(view);
    if (Number.isFinite(elapsedMs)) view.elapsed.textContent = formatElapsed(elapsedMs);
    if (Number.isFinite(tokPerS) && tokPerS > 0) {
      view.rate.textContent = `${tokPerS.toFixed(1)} tok/s`;
      view.rate.hidden = false;
    }
    view.copy.hidden = false;
    view.article.setAttribute('aria-busy', 'false');
  }

  function failView(view, message, userText) {
    cancelAnimationFrame(view.frame);
    setStatus(view, '');
    if (view.text) renderAnswer(view);
    view.article.setAttribute('aria-busy', 'false');
    view.article.append(h('div', { class: 'turn__error', role: 'alert' }, [
      h('span', { text: `Couldn't finish this answer. ${message}` }),
      h('button', { type: 'button', class: 'btn btn-small', text: 'Retry', onclick: () => retryTurn(view, userText) }),
    ]));
  }

  function retryTurn(view, text) {
    if (state.busy) return;
    view.article.remove();
    streamAnswer(text);
  }

  function savedTurn(turn) {
    if (turn.role === 'user') return userTurn(turn.text, turn.created_at);
    const view = assistantView(turn.created_at);
    view.sources = turn.sources || [];
    view.text = turn.text || '';
    renderAnswer(view);
    renderSources(view);
    view.copy.hidden = false;
    view.article.setAttribute('aria-busy', 'false');
    return view.article;
  }

  async function copyAnswer(view) {
    try {
      await navigator.clipboard.writeText(view.text);
      flash(view.copy, 'Copied');
    } catch {
      flash(view.copy, 'Copy failed');
    }
  }

  // ---------- streaming ----------

  // Frame format: "event: <name>\ndata: <json>". Comment lines and malformed frames are skipped.
  function parseFrame(raw) {
    let event = 'message';
    const data = [];
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (!data.length) return null;
    try {
      return { event, data: JSON.parse(data.join('\n')) };
    } catch {
      return null;
    }
  }

  // Returns true at the terminal "done" event. Throws on an "error" event.
  function handleEvent(name, data, view) {
    switch (name) {
      case 'meta':
        state.threadId = data.thread_id || state.threadId;
        if (data.provider && data.model) els.model.textContent = `${data.provider} · ${data.model}`;
        refreshThreads();
        return false;
      case 'memory':
        showRecall(view, data.items || []);
        return false;
      case 'search_start':
        setStatus(view, `Searching the web for “${data.query}”…`, 'progress');
        return false;
      case 'search':
        onSearch(view, data);
        return false;
      case 'token':
        clearProgress(view);
        view.text += data.delta ?? '';
        queueRender(view);
        return false;
      case 'reset':
        view.text = '';
        view.rate.hidden = true;
        renderAnswer(view);
        return false;
      case 'stats':
        if (Number.isFinite(data.tok_per_s) && data.tok_per_s > 0) {
          view.rate.textContent = `${data.tok_per_s.toFixed(1)} tok/s`;
          view.rate.hidden = false;
        }
        return false;
      case 'done':
        completeView(view, data);
        refreshThreads();
        if (state.memoryOpen) loadMemory();
        return true;
      case 'error':
        throw new Error(data.message || 'The server reported an error.');
      default:
        return false;
    }
  }

  // Coalesces token bursts into one markdown render per frame.
  function queueRender(view) {
    if (view.frame) return;
    view.frame = requestAnimationFrame(() => {
      view.frame = 0;
      followIfPinned(() => renderAnswer(view));
    });
  }

  async function readStream(message, view) {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify({
        thread_id: state.threadId || undefined,
        message,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      }),
    });
    if (!res.ok || !res.body) throw new Error(errorMessage(await res.json().catch(() => null), res.status));
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done }).replace(/\r\n/g, '\n');
      if (done) buffer += '\n\n';
      let end = buffer.indexOf('\n\n');
      while (end !== -1) {
        const frame = parseFrame(buffer.slice(0, end));
        buffer = buffer.slice(end + 2);
        if (frame && followIfPinned(() => handleEvent(frame.event, frame.data, view))) return;
        end = buffer.indexOf('\n\n');
      }
      if (done) break;
    }
    throw new Error('The connection closed before the answer finished.');
  }

  async function streamAnswer(message) {
    const view = assistantView(new Date().toISOString());
    els.turns.append(view.article);
    updateEmpty();
    scrollToEnd();
    setStatus(view, 'Thinking…', 'progress');
    setBusy(true);
    try {
      await readStream(message, view);
    } catch (error) {
      failView(view, error.message, message);
    } finally {
      setBusy(false);
    }
  }

  function sendMessage(text) {
    if (state.busy) return;
    els.turns.append(userTurn(text, new Date().toISOString()));
    updateEmpty();
    scrollToEnd();
    streamAnswer(text);
  }

  // Thread switching and new chats are blocked while an answer streams, so events never land in the wrong thread.
  function setBusy(busy) {
    state.busy = busy;
    els.send.disabled = busy;
    els.send.classList.toggle('is-busy', busy);
    els.send.setAttribute('aria-label', busy ? 'Answering' : 'Send question');
    els.newChat.disabled = busy;
    els.threadList.inert = busy;
  }

  // ---------- threads ----------

  function renderThreads(threads) {
    state.threads = threads;
    const sorted = sortThreads(threads);
    els.threadsEmpty.hidden = sorted.length > 0;
    els.threadList.replaceChildren(...sorted.map((thread) => {
      const title = thread.title || 'New chat';
      const active = thread.id === state.threadId;
      return h('li', { class: active ? 'thread is-active' : 'thread' }, [
        h('button', {
          type: 'button',
          class: 'thread__open',
          'aria-current': active ? 'true' : null,
          onclick: () => openThread(thread.id),
        }, [
          h('span', { class: 'thread__title', text: title }),
          h('span', { class: 'thread__time', text: relTime(thread.updated_at) }),
        ]),
        h('button', {
          type: 'button',
          class: 'thread__delete',
          'aria-label': `Delete ${title}`,
          title: 'Delete thread',
          text: '×',
          onclick: () => deleteThread(thread.id, title),
        }),
      ]);
    }));
  }

  async function refreshThreads() {
    try {
      const { threads } = await api('/api/threads');
      renderThreads(threads || []);
    } catch (error) {
      notify(`Couldn't load threads: ${error.message}`);
    }
    return state.threads;
  }

  async function openThread(id) {
    if (state.busy) {
      notify('Wait for the current answer to finish.');
      return;
    }
    try {
      const { thread, turns } = await api(`/api/threads/${encodeURIComponent(id)}`);
      state.threadId = thread?.id ?? id;
      els.turns.replaceChildren(...turns.map(savedTurn));
      document.title = `${thread?.title || 'Chat'} · Askline`;
      updateEmpty();
      renderThreads(state.threads);
      scrollToEnd();
      closeSidebar();
    } catch (error) {
      notify(`Couldn't open that thread: ${error.message}`);
    }
  }

  async function newChat() {
    if (state.busy) return;
    try {
      const thread = await api('/api/threads', { method: 'POST' });
      await refreshThreads();
      await openThread(thread.id);
      loadPrompts();
      els.prompt.focus();
    } catch (error) {
      notify(`Couldn't start a chat: ${error.message}`);
    }
  }

  function renderExamples(prompts) {
    if (!els.examples || !prompts.length) return;
    els.examples.replaceChildren(
      ...prompts.slice(0, 3).map((text) => {
        const button = h('button', { type: 'button', class: 'example' }, text);
        button.addEventListener('click', () => sendMessage(text));
        const item = document.createElement('li');
        item.append(button);
        return item;
      })
    );
  }

  async function loadPrompts() {
    try {
      const { prompts } = await api('/api/prompts');
      renderExamples(prompts || []);
    } catch {
      // The static examples stay when suggestions are unavailable.
    }
  }

  async function deleteThread(id, title) {
    if (state.busy) return;
    if (!window.confirm(`Delete “${title}”? This cannot be undone.`)) return;
    try {
      await api(`/api/threads/${encodeURIComponent(id)}`, { method: 'DELETE' });
    } catch (error) {
      notify(`Couldn't delete the thread: ${error.message}`);
      return;
    }
    const threads = await refreshThreads();
    if (id === state.threadId) {
      if (threads.length) await openThread(sortThreads(threads)[0].id);
      else await newChat();
    }
    els.newChat.focus();
  }

  // ---------- memory drawer (a record of asks, grouped by local day) ----------

  function setMemoryOpen(open) {
    state.memoryOpen = open;
    if (open) closeSidebar();
    els.memory.classList.toggle('is-open', open);
    els.memoryBtn.setAttribute('aria-expanded', String(open));
    els.shell.inert = open;
    syncScrim();
    if (open) {
      loadMemory();
      els.memoryTitle.focus();
    } else {
      els.memoryBtn.focus();
    }
  }

  async function loadMemory() {
    els.memoryBody.replaceChildren(h('p', { class: 'drawer__note', role: 'status', text: 'Loading…' }));
    try {
      const { days = [] } = await api('/api/timeline?limit=100');
      renderMemory(days.flatMap((day) => day.asks || []).sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at)));
    } catch (error) {
      els.memoryBody.replaceChildren(
        h('p', { class: 'drawer__note', role: 'alert', text: `Couldn't load memory: ${error.message}` }),
        h('button', { type: 'button', class: 'btn-ghost', text: 'Try again', onclick: loadMemory }),
      );
    }
  }

  function renderMemory(asks) {
    if (!asks.length) {
      els.memoryBody.replaceChildren(h('p', { class: 'drawer__note', text: 'No questions yet. Each question you send is recorded here with the time you asked it.' }));
      return;
    }
    const groups = new Map();
    for (const ask of asks) {
      const date = new Date(ask.created_at);
      const key = dayKey(date);
      if (!groups.has(key)) groups.set(key, { date, asks: [] });
      groups.get(key).asks.push(ask);
    }
    els.memoryBody.replaceChildren(...[...groups.values()].map((group) => h('section', { class: 'day' }, [
      h('h3', { class: 'day__title', text: dayHeading(group.date) }),
      h('ul', { class: 'asks' }, group.asks.map(askNode)),
    ])));
  }

  function askNode(ask) {
    const title = ask.thread_title || 'Untitled thread';
    return h('li', { class: 'ask' }, [
      h('time', { class: 'ask__time', datetime: ask.created_at, text: clockFmt.format(new Date(ask.created_at)) }),
      h('button', {
        type: 'button',
        class: 'link-btn ask__text',
        title,
        'aria-label': `Open thread “${title}”`,
        text: ask.text,
        onclick: () => openFromMemory(ask.thread_id),
      }),
    ]);
  }

  function openFromMemory(threadId) {
    if (state.busy) {
      notify('Wait for the current answer to finish.');
      return;
    }
    setMemoryOpen(false);
    openThread(threadId);
  }

  // ---------- sidebar and overlays ----------

  function setSidebarOpen(open) {
    state.sidebarOpen = open;
    if (!open && els.sidebar.contains(document.activeElement)) els.menu.focus();
    els.sidebar.classList.toggle('is-open', open);
    els.menu.setAttribute('aria-expanded', String(open));
    els.menu.setAttribute('aria-label', open ? 'Close threads' : 'Open threads');
    syncScrim();
    if (open) els.newChat.focus();
  }

  const closeSidebar = () => { if (state.sidebarOpen) setSidebarOpen(false); };

  function closeOverlays() {
    closeSidebar();
    if (state.memoryOpen) setMemoryOpen(false);
  }

  const syncScrim = () => { els.scrim.hidden = !(state.sidebarOpen || state.memoryOpen); };

  function onKeydown(event) {
    if (event.key !== 'Escape' || els.dialog.open) return;
    if (state.memoryOpen) setMemoryOpen(false);
    else closeSidebar();
  }

  // ---------- composer ----------

  // Grows with the text up to 8 rows, then scrolls.
  function resizePrompt() {
    const box = els.prompt;
    const style = getComputedStyle(box);
    const line = parseFloat(style.lineHeight) || parseFloat(style.fontSize) * 1.5;
    const max = line * 8 + parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
    box.style.height = 'auto';
    box.style.height = `${Math.min(box.scrollHeight, max)}px`;
    box.style.overflowY = box.scrollHeight > max ? 'auto' : 'hidden';
  }

  function onPromptKey(event) {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      els.composer.requestSubmit();
    }
  }

  function onSubmit(event) {
    event.preventDefault();
    const text = els.prompt.value.trim();
    if (!text || state.busy) return;
    els.prompt.value = '';
    resizePrompt();
    sendMessage(text);
  }

  // ---------- header and settings ----------

  function applyHealth(health) {
    els.model.textContent = `${health.provider} · ${health.model}`;
    els.banner.hidden = health.has_api_key !== false;
  }

  const setDialogStatus = (message, tone = '') => {
    els.dialogStatus.textContent = message;
    els.dialogStatus.dataset.tone = tone;
  };

  function setDialogBusy(busy, label = '') {
    els.test.disabled = busy;
    els.save.disabled = busy;
    els.form.setAttribute('aria-busy', String(busy));
    if (busy) setDialogStatus(label);
  }

  function fillSettings(settings) {
    const f = els.f;
    if (![...f.provider.options].some((o) => o.value === settings.provider)) {
      f.provider.append(new Option(settings.provider, settings.provider));
    }
    f.provider.value = settings.provider;
    f.baseUrl.value = settings.base_url ?? '';
    f.model.value = settings.model ?? '';
    f.apiKey.value = '';
    f.apiKey.placeholder = settings.has_api_key ? 'keep current key' : 'Paste an API key';
    f.temperature.value = settings.temperature;
    f.maxTokens.value = settings.max_tokens;
  }

  // A blank API key is omitted, so the server keeps the stored one.
  function readSettings() {
    const f = els.f;
    const body = {
      provider: f.provider.value,
      base_url: f.baseUrl.value.trim(),
      model: f.model.value.trim(),
      temperature: Number(f.temperature.value),
      max_tokens: Number(f.maxTokens.value),
    };
    const key = f.apiKey.value.trim();
    if (key) body.api_key = key;
    return body;
  }

  async function loadModels() {
    try {
      const { models = [], error } = await api('/api/models');
      els.modelOptions.replaceChildren(...models.map((m) => h('option', { value: m.id, label: m.label || m.id })));
      els.modelHint.textContent = error ? `Model list unavailable: ${error}` : 'Pick a suggestion or type any model id.';
    } catch (error) {
      els.modelHint.textContent = `Model list unavailable: ${error.message}`;
    }
  }

  async function openSettings() {
    setDialogStatus('');
    try {
      fillSettings(await api('/api/settings'));
    } catch (error) {
      setDialogStatus(`Couldn't load settings: ${error.message}`, 'error');
    }
    els.dialog.showModal();
    loadModels();
  }

  async function testConnection() {
    if (!els.form.reportValidity()) return;
    setDialogBusy(true, 'Testing…');
    try {
      const result = await api('/api/settings/test', { method: 'POST', body: readSettings() });
      if (result.ok) setDialogStatus(`Connected. Replied in ${result.latency_ms} ms.`, 'ok');
      else setDialogStatus(`Connection failed: ${result.error || 'unknown error'}`, 'error');
    } catch (error) {
      setDialogStatus(`Connection failed: ${error.message}`, 'error');
    } finally {
      setDialogBusy(false);
    }
  }

  async function saveSettings(event) {
    event.preventDefault();
    if (!els.form.reportValidity()) return;
    setDialogBusy(true, 'Saving…');
    let saved;
    try {
      saved = await api('/api/settings', { method: 'POST', body: readSettings() });
    } catch (error) {
      setDialogStatus(`Save failed: ${error.message}`, 'error');
      setDialogBusy(false);
      return;
    }
    fillSettings(saved);
    setDialogBusy(false);
    els.dialog.close();
    try {
      applyHealth(await api('/api/health'));
    } catch (error) {
      notify(`Settings saved, but the header could not refresh: ${error.message}`);
    }
  }

  // ---------- startup ----------

  function bind() {
    els.menu.addEventListener('click', () => setSidebarOpen(!state.sidebarOpen));
    els.newChat.addEventListener('click', newChat);
    els.memoryBtn.addEventListener('click', () => setMemoryOpen(!state.memoryOpen));
    els.memoryClose.addEventListener('click', () => setMemoryOpen(false));
    els.settingsBtn.addEventListener('click', openSettings);
    els.bannerSettings.addEventListener('click', openSettings);
    els.scrim.addEventListener('click', closeOverlays);
    document.addEventListener('keydown', onKeydown);
    narrowQuery.addEventListener('change', () => setSidebarOpen(false));

    els.prompt.addEventListener('input', resizePrompt);
    els.prompt.addEventListener('keydown', onPromptKey);
    els.composer.addEventListener('submit', onSubmit);

    els.form.addEventListener('submit', saveSettings);
    els.test.addEventListener('click', testConnection);
    els.cancel.addEventListener('click', () => els.dialog.close());
    // Every close path (Cancel, Escape, Save) returns focus to the header button.
    els.dialog.addEventListener('close', () => els.settingsBtn.focus());
  }

  async function init() {
    const [health, threads] = await Promise.allSettled([api('/api/health'), api('/api/threads')]);
    if (health.status === 'fulfilled') applyHealth(health.value);
    else notify(`Couldn't reach the server: ${health.reason.message}`);

    if (threads.status === 'fulfilled') renderThreads(threads.value.threads || []);
    else notify(`Couldn't load threads: ${threads.reason.message}`);
    loadPrompts();

    if (!state.threads.length) {
      await newChat();
      return;
    }
    await openThread(sortThreads(state.threads)[0].id);
  }

  bind();
  resizePrompt();
  init().catch((error) => notify(`Startup failed: ${error.message}`));
})();
