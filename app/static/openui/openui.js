const { React, createRoot, Renderer, openuiChatLibrary } = window.__OpenUI || {};
const form = document.getElementById("form");
const input = document.getElementById("question");
const submit = document.getElementById("submit");
const submitLabel = document.getElementById("submit-label");
const starters = document.getElementById("starters");
const thread = document.getElementById("thread");

let busy = false;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function setBusy(value) {
  busy = value;
  submit.disabled = value;
  submitLabel.textContent = value ? "Composing" : "Compose";
  input.setAttribute("aria-busy", String(value));
  starters.inert = value;
}

function turnShell(question, existing) {
  if (existing) return existing;
  const turn = el("article", "turn");
  turn.append(el("p", "turn__question", question));
  thread.append(turn);
  turn.scrollIntoView({ block: "start", behavior: "smooth" });
  return turn;
}

function loadingCard() {
  const card = el("div", "result");
  const status = el("div", "result__status");
  status.append(el("div", "result__bar"), el("span", "", "Searching the web and composing an interface…"));
  card.append(status);
  return card;
}

function errorCard(turn, question, message) {
  const card = el("div", "result__error");
  card.append(el("p", "", message));
  const retry = el("button", "retry", "Try again");
  retry.type = "button";
  retry.addEventListener("click", () => {
    card.remove();
    ask(question, turn);
  });
  card.append(retry);
  turn.append(card);
}

async function readError(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((item) => item.msg).join(", ");
  } catch {
    // The body was not JSON; fall through to the status text.
  }
  return `The request failed with status ${response.status}.`;
}

function renderInterface(node, lang, turn) {
  const root = createRoot(node);
  root.render(
    React.createElement(Renderer, {
      library: openuiChatLibrary,
      response: lang,
      onAction: (event) => {
        if (event?.type === "continue_conversation" && event.humanFriendlyMessage) {
          ask(event.humanFriendlyMessage);
        }
      },
    }),
  );
}

async function ask(question, existingTurn) {
  const text = String(question || "").trim();
  if (!text || busy) return;
  if (!window.__OpenUI || !Renderer || !openuiChatLibrary) {
    const turn = turnShell(text, existingTurn);
    errorCard(turn, text, "The OpenUI bundle did not load. Reload the page.");
    return;
  }

  setBusy(true);
  input.value = "";
  const turn = turnShell(text, existingTurn);
  const card = el("div", "result");
  card.append(loadingCard());
  turn.append(card);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 120000);

  try {
    const response = await fetch("/api/openui", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(await readError(response));
    const data = await response.json();

    card.replaceChildren();
    const mount = el("div", "result__mount");
    card.append(mount);
    renderInterface(mount, data.lang, turn);

    const meta = el("div", "result__meta");
    meta.append(
      el("span", "", `${(data.elapsed_ms / 1000).toFixed(1)} s`),
      el("span", "", data.model),
      el("span", "", "OpenUI Lang"),
    );
    card.append(meta);

    const lang = el("details", "result__lang");
    lang.append(el("summary", "", "View the generated program"), el("pre", "", data.lang));
    card.append(lang);
  } catch (error) {
    card.remove();
    const message =
      error.name === "AbortError" ? "The model took too long. Try a narrower question." : error.message;
    errorCard(turn, text, message);
  } finally {
    clearTimeout(timer);
    setBusy(false);
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  ask(input.value);
});

starters.addEventListener("click", (event) => {
  const button = event.target.closest("[data-question]");
  if (button) ask(button.dataset.question);
});

ask(new URLSearchParams(window.location.search).get("q") || "");
