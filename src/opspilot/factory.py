"""Wires the retriever, HTTP clients, tools, model and graph together."""

import httpx
from langchain_core.language_models.chat_models import BaseChatModel

from opspilot.agent import OpsPilotAgent, build_graph
from opspilot.config import Settings
from opspilot.llm import get_chat_model, get_fallback_models
from opspilot.retrieval import Retriever, build_retriever
from opspilot.tools import make_tools


def build_agent(
    settings: Settings,
    *,
    model: BaseChatModel | None = None,
    retriever: Retriever | None = None,
    service_desk: httpx.Client | None = None,
    public_http: httpx.Client | None = None,
) -> OpsPilotAgent:
    timeout = settings.http_timeout_seconds
    tools = make_tools(
        retriever or build_retriever(settings),
        service_desk or httpx.Client(base_url=settings.ticket_api_url, timeout=timeout),
        public_http or httpx.Client(timeout=timeout),
        holiday_api_url=settings.holiday_api_url,
        default_country=settings.holiday_country,
        top_k=settings.retrieval_top_k,
    )
    if model is None:
        model, fallbacks = get_chat_model(settings), get_fallback_models(settings)
    else:
        fallbacks = []
    graph = build_graph(model, tools, fallbacks=fallbacks)
    return OpsPilotAgent(graph, max_steps=settings.max_agent_steps, max_input_chars=settings.max_input_chars)
