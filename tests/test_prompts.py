from langchain_core.messages import AIMessage

from app.config import ChatConfig
from app.memory import Store
from app.prompts import DEFAULT_PROMPTS, parse_suggestions, suggested_prompts

CFG = ChatConfig(provider="openai", base_url="http://llm/v1", model="test", api_key="test")


def test_parse_ignores_commentary_and_duplicates():
    text = "Thinking...\n- When is the next Python release?\nThese follow from the history.\nWhen is the next Python release?\nHow does replication lag work?"
    assert parse_suggestions(text) == ["When is the next Python release?", "How does replication lag work?"]


def test_parse_suggestions_cleans_lines():
    text = "1. What changed in Python 3.15?\n- How does Postgres replication work?\n• Which advisories matter this week?\nignored short\n"
    assert parse_suggestions(text) == [
        "What changed in Python 3.15?",
        "How does Postgres replication work?",
        "Which advisories matter this week?",
    ]


def test_parse_suggestions_limits_to_three():
    text = "\n".join(f"What is question number {n} about?" for n in range(6))
    assert len(parse_suggestions(text)) == 3


async def test_defaults_without_history(tmp_path, monkeypatch):
    store = Store(tmp_path / "chat.db")
    monkeypatch.setattr("app.prompts.build_chat_model", lambda cfg, **kwargs: None)
    prompts, source = await suggested_prompts(store, CFG)
    assert prompts == DEFAULT_PROMPTS
    assert source == "defaults"


async def test_generates_from_history_and_caches(tmp_path, monkeypatch):
    store = Store(tmp_path / "chat.db")
    store.add_turn(store.create_thread("One")["id"], "user", "What changed in the latest Python release?")
    store.add_turn(store.create_thread("Two")["id"], "user", "Compare Postgres logical and physical replication")
    class PromptModel:
        def __init__(self, text):
            self.text = text
            self.calls = []

        async def ainvoke(self, messages):
            self.calls.append(messages)
            return AIMessage(content=self.text)

    model = PromptModel(
        "What is new in Python 3.16?\n"
        "How does logical replication lag work?\n"
        "Which security advisories matter this week?"
    )
    monkeypatch.setattr("app.prompts.build_chat_model", lambda cfg, **kwargs: model)

    prompts, source = await suggested_prompts(store, CFG)
    assert source == "history"
    assert prompts[0] == "What is new in Python 3.16?"
    assert len(prompts) == 3
    calls = len(model.calls)

    again, source = await suggested_prompts(store, CFG)
    assert again == prompts
    assert source == "history"
    assert len(model.calls) == calls


async def test_empty_reply_uses_defaults_without_caching(tmp_path, monkeypatch):
    store = Store(tmp_path / "chat.db")
    store.add_turn(store.create_thread("One")["id"], "user", "What changed in the latest Python release?")
    store.add_turn(store.create_thread("Two")["id"], "user", "Compare Postgres replication")

    class Silent:
        async def ainvoke(self, messages):
            return AIMessage(content="")

    monkeypatch.setattr("app.prompts.build_chat_model", lambda cfg, **kwargs: Silent())
    prompts, source = await suggested_prompts(store, CFG)
    assert prompts == DEFAULT_PROMPTS
    assert source == "defaults"
    assert store.setting("prompt_suggestions_v3") is None


async def test_model_failure_falls_back(tmp_path, monkeypatch):
    store = Store(tmp_path / "chat.db")
    store.add_turn(store.create_thread("One")["id"], "user", "What changed in the latest Python release?")
    store.add_turn(store.create_thread("Two")["id"], "user", "Compare Postgres replication")

    class Broken:
        async def ainvoke(self, messages):
            raise RuntimeError("down")

    monkeypatch.setattr("app.prompts.build_chat_model", lambda cfg, **kwargs: Broken())
    prompts, source = await suggested_prompts(store, CFG)
    assert prompts == DEFAULT_PROMPTS
    assert source == "defaults"
