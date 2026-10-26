"""The agent: a LangGraph state machine that alternates between the LLM and tool execution.

    START -> agent --(tool calls?)--> tools -> agent -> ... -> END

``request_leave`` calls ``interrupt()``, which pauses the graph and saves its state in the checkpointer
until a manager approves or rejects the request; ``OpsPilotAgent.resolve_approval`` resumes it.
"""

import logging
import re
import time
import uuid
from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from pydantic import BaseModel, Field

from opspilot.guardrails import BLOCKED_INPUT_MESSAGE, check_input, check_output, redact_pii
from opspilot.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_SOURCE_RE = re.compile(r"\[source: ([^\]]+)\]")


def build_graph(model: BaseChatModel, tools: Sequence[BaseTool], checkpointer=None):
    model_with_tools = model.bind_tools(list(tools))

    def call_model(state: MessagesState) -> dict:
        response = model_with_tools.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(list(tools)))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


class ChatResult(BaseModel):
    thread_id: str
    status: Literal["answered", "pending_approval", "blocked"]
    answer: str | None = None
    pending_action: dict[str, Any] | None = None
    tools_used: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    guardrail: str | None = None
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class OpsPilotAgent:
    def __init__(self, graph, *, max_steps: int = 12, max_input_chars: int = 2000) -> None:
        self.graph = graph
        self.max_steps = max_steps
        self.max_input_chars = max_input_chars
        self._pending: dict[str, dict[str, Any]] = {}

    def _config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": self.max_steps}

    def chat(self, message: str, thread_id: str | None = None) -> ChatResult:
        thread_id = thread_id or uuid.uuid4().hex
        verdict = check_input(message, self.max_input_chars)
        logger.info("chat thread=%s message=%r", thread_id, redact_pii(message))
        if not verdict.allowed:
            logger.warning("blocked input thread=%s reason=%s", thread_id, verdict.reason)
            return ChatResult(
                thread_id=thread_id,
                status="blocked",
                answer=BLOCKED_INPUT_MESSAGE,
                guardrail=verdict.reason,
            )
        if thread_id in self._pending:
            return ChatResult(
                thread_id=thread_id,
                status="pending_approval",
                answer="This conversation is waiting for a manager to approve a leave request.",
                pending_action=self._pending[thread_id],
            )
        return self._run({"messages": [HumanMessage(message)]}, thread_id)

    def resolve_approval(
        self, thread_id: str, approved: bool, reviewer: str, comment: str | None = None
    ) -> ChatResult:
        if thread_id not in self._pending:
            raise KeyError(thread_id)
        decision = {"approved": approved, "reviewer": reviewer, "comment": comment}
        logger.info("approval thread=%s approved=%s reviewer=%s", thread_id, approved, reviewer)
        return self._run(Command(resume=decision), thread_id)

    def pending_approvals(self) -> dict[str, dict[str, Any]]:
        return dict(self._pending)

    def _run(self, graph_input, thread_id: str) -> ChatResult:
        config = self._config(thread_id)
        before = len(self.graph.get_state(config).values.get("messages", []))
        started = time.perf_counter()
        self.graph.invoke(graph_input, config)
        latency_ms = int((time.perf_counter() - started) * 1000)

        state = self.graph.get_state(config)
        new_messages: list[BaseMessage] = state.values.get("messages", [])[before:]
        result = ChatResult(thread_id=thread_id, status="answered", latency_ms=latency_ms)
        self._collect_metrics(new_messages, result)

        if state.interrupts:
            self._pending[thread_id] = state.interrupts[0].value
            result.status = "pending_approval"
            result.pending_action = self._pending[thread_id]
            result.answer = "Your leave request has been sent to your manager for approval."
            return result

        self._pending.pop(thread_id, None)
        final = next((m for m in reversed(new_messages) if isinstance(m, AIMessage)), None)
        text, verdict = check_output(_message_text(final) if final else "")
        result.answer = text or "Sorry, I couldn't produce an answer."
        if not verdict.allowed:
            result.guardrail = verdict.reason
            logger.warning("output guardrail thread=%s reason=%s", thread_id, verdict.reason)
        return result

    @staticmethod
    def _collect_metrics(messages: list[BaseMessage], result: ChatResult) -> None:
        for m in messages:
            if isinstance(m, AIMessage):
                result.tools_used.extend(call["name"] for call in m.tool_calls)
                usage = m.usage_metadata or {}
                result.input_tokens += usage.get("input_tokens", 0)
                result.output_tokens += usage.get("output_tokens", 0)
            elif isinstance(m, ToolMessage) and m.name == "search_policies":
                for source in _SOURCE_RE.findall(_message_text(m)):
                    if source not in result.sources:
                        result.sources.append(source)


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content).strip()
