from __future__ import annotations

import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedModel(BaseChatModel):
    """Deterministic stand-in for a chat model that may or may not call tools."""

    script: list[AIMessage]
    calls: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN201
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,  # noqa: ANN001
        **kwargs,
    ) -> ChatResult:
        self.calls.append(list(messages))
        if not self.script:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="(script exhausted)"))])
        message = self.script.pop(0)
        if isinstance(message.content, str):
            message.content = re.sub(r"\s+", " ", message.content).strip()
        return ChatResult(generations=[ChatGeneration(message=message)])
