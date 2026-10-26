import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from opspilot import __version__
from opspilot.agent import ChatResult, OpsPilotAgent
from opspilot.config import get_settings
from opspilot.factory import build_agent
from opspilot.llm import LLMNotConfiguredError

logger = logging.getLogger("opspilot")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    if getattr(app.state, "agent", None) is None:
        try:
            app.state.agent = build_agent(settings)
        except LLMNotConfiguredError as exc:
            logger.warning("Agent disabled: %s", exc)
            app.state.agent = None
    yield


app = FastAPI(
    title="OpsPilot",
    version=__version__,
    description="Internal operations assistant: policy Q&A, IT tickets and human-approved actions.",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, examples=["How many days of annual leave do I get?"])
    thread_id: str | None = Field(default=None, description="Omit to start a new conversation.")


class ApprovalDecision(BaseModel):
    approved: bool
    reviewer: str = Field(min_length=1, examples=["Jane Manager"])
    comment: str | None = None


def _agent(request: Request) -> OpsPilotAgent:
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="LLM is not configured. See .env.example.")
    return agent


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    settings = get_settings()
    ready = getattr(request.app.state, "agent", None) is not None
    return {
        "status": "ok",
        "version": __version__,
        "env": settings.env,
        "agent": "ready" if ready else "llm_not_configured",
    }


@app.post("/chat")
def chat(body: ChatRequest, request: Request) -> ChatResult:
    return _agent(request).chat(body.message, body.thread_id)


@app.get("/approvals")
def list_approvals(request: Request) -> dict[str, dict]:
    """Leave requests waiting for a manager, keyed by conversation thread id."""
    return _agent(request).pending_approvals()


@app.post("/approvals/{thread_id}")
def resolve_approval(thread_id: str, body: ApprovalDecision, request: Request) -> ChatResult:
    try:
        return _agent(request).resolve_approval(thread_id, body.approved, body.reviewer, body.comment)
    except KeyError:
        raise HTTPException(status_code=404, detail="No pending approval for this thread") from None
