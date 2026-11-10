from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from app.config import ChatConfig
from app.llm import build_chat_model


def test_openai_compatible_model_builds():
    cfg = ChatConfig(
        provider="openai",
        base_url="http://127.0.0.1:11434/v1",
        model="gemma4:e2b",
        api_key="local",
    )
    model = build_chat_model(cfg)
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gemma4:e2b"
    assert model.temperature == 0.2
    assert model.max_tokens == 800


def test_anthropic_model_builds():
    cfg = ChatConfig(
        provider="anthropic",
        base_url="https://api.anthropic.com",
        model="claude-sonnet-4-5",
        api_key="sk-ant-test",
    )
    model = build_chat_model(cfg)
    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-sonnet-4-5"
    assert model.max_tokens == 800
