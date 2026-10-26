"""A scripted chat model so the agent can be tested without calling a real LLM."""

from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

Step = AIMessage | Callable[[list[BaseMessage]], AIMessage]


class ScriptedChatModel(BaseChatModel):
    """Returns the scripted responses in order. A step may be a function of the messages it receives."""

    steps: list[Any]
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    @property
    def calls(self) -> list[list[BaseMessage]]:
        return self._calls

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self._calls.append(list(messages))
        step = self.steps[len(self._calls) - 1]
        message = step(messages) if callable(step) else step
        return ChatResult(generations=[ChatGeneration(message=message)])


def tool_call(name: str, args: dict, call_id: str = "call-1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
        usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
    )


def answer(text: str) -> AIMessage:
    return AIMessage(
        content=text,
        usage_metadata={"input_tokens": 200, "output_tokens": 30, "total_tokens": 230},
    )
