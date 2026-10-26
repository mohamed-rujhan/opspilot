"""Demo UI for OpsPilot: an employee chat on the left, a manager approval queue on the right.

streamlit run app/streamlit_app.py
"""

import html
import os
from pathlib import Path

import httpx
import streamlit as st

API_URL = os.getenv("OPSPILOT_API_URL", "http://localhost:8000")

EXAMPLES = [
    "How many days of annual leave do I get?",
    "My laptop was stolen, please raise a ticket.",
    "What are the public holidays in 2026?",
    "Book leave 21–24 Dec for a family visit.",
]

CSS = (Path(__file__).parent / "style.css").read_text()

st.set_page_config(page_title="OpsPilot", page_icon="🧭", layout="wide")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

api = httpx.Client(base_url=API_URL, timeout=90)

st.session_state.setdefault("thread_id", None)
st.session_state.setdefault("history", [])
st.session_state.setdefault("awaiting", None)


def agent_ready() -> bool:
    try:
        return api.get("/health").json().get("agent") == "ready"
    except httpx.HTTPError:
        return False


def pills(result: dict) -> str:
    parts = [f'<span class="pill tool">🛠 {html.escape(t)}</span>' for t in result.get("tools_used") or []]
    parts += [f'<span class="pill src">📄 {html.escape(s)}</span>' for s in result.get("sources") or []]
    if result.get("guardrail"):
        parts.append(f'<span class="pill guard">🛡 {html.escape(result["guardrail"])}</span>')
    if result.get("latency_ms"):
        parts.append(f'<span class="pill time">⏱ {result["latency_ms"] / 1000:.1f}s</span>')
    return f'<div class="meta">{"".join(parts)}</div>' if parts else ""


def show_result(result: dict) -> None:
    st.markdown(result.get("answer") or "")
    if meta := pills(result):
        st.markdown(meta, unsafe_allow_html=True)


def send(message: str) -> dict:
    response = api.post("/chat", json={"message": message, "thread_id": st.session_state.thread_id})
    if response.status_code != 200:
        return {"answer": f"⚠️ {response.json().get('detail', response.text)}"}
    result = response.json()
    st.session_state.thread_id = result["thread_id"]
    return result


def ask(message: str) -> None:
    st.session_state.history.append(("user", message))
    st.session_state.awaiting = message


ready = agent_ready()
status = (
    '<div class="status ok">● Agent online</div>'
    if ready
    else '<div class="status off">● LLM not configured</div>'
)
st.markdown(
    f"""
    <div class="hero">
      <div>
        <h1>🧭 OpsPilot</h1>
        <p>Internal operations assistant for Lumen Analytics Ltd — a fictional company</p>
        <div class="tags"><span>LangGraph agent</span><span>RAG with citations</span>
          <span>Human-in-the-loop</span><span>Guardrails</span></div>
      </div>
      {status}
    </div>
    """,
    unsafe_allow_html=True,
)

chat_col, manager_col = st.columns([2.1, 1], gap="medium")

with chat_col, st.container(key="chat_card"):
    header, reset = st.columns([5, 1], vertical_alignment="center")
    header.markdown('<div class="card-title">💬 Employee chat</div>', unsafe_allow_html=True)
    if reset.button("New chat", use_container_width=True):
        st.session_state.update(thread_id=None, history=[], awaiting=None)
        st.rerun()

    messages = st.container(height=340, autoscroll=True, key="messages")
    with messages:
        if not st.session_state.history:
            st.markdown(
                '<div class="hint">Ask about leave, IT, expenses — or try one of these:</div>',
                unsafe_allow_html=True,
            )
            cols = st.columns(2)
            for i, example in enumerate(EXAMPLES):
                if cols[i % 2].button(example, key=f"ex-{i}", use_container_width=True):
                    ask(example)
                    st.rerun()
        for role, payload in st.session_state.history:
            with st.chat_message(role, avatar="🧑‍💼" if role == "user" else "🧭"):
                if role == "assistant":
                    show_result(payload)
                else:
                    st.markdown(f'<div class="ububble">{html.escape(payload)}</div>', unsafe_allow_html=True)
        if st.session_state.awaiting:
            with st.chat_message("assistant", avatar="🧭"), st.spinner("Thinking…"):
                result = send(st.session_state.awaiting)
            st.session_state.history.append(("assistant", result))
            st.session_state.awaiting = None
            st.rerun()

    if prompt := st.chat_input("Ask about leave, IT, expenses… or raise a ticket"):
        ask(prompt)
        st.rerun()

with manager_col, st.container(key="manager_card"):
    try:
        pending = api.get("/approvals").json()
    except httpx.HTTPError:
        pending = {}
    badge = f'<span class="count{" zero" if not pending else ""}">{len(pending)}</span>'
    st.markdown(f'<div class="card-title">✅ Manager approvals {badge}</div>', unsafe_allow_html=True)
    if not pending:
        st.markdown('<div class="empty">No requests waiting for approval</div>', unsafe_allow_html=True)
    for thread_id, action in pending.items():
        st.markdown(
            f"""<div class="req"><b>🌴 {html.escape(action["leave_type"].title())} leave</b>
            <div class="dates">{action["start_date"]} → {action["end_date"]}</div>
            <div class="why">{html.escape(action["reason"])}</div></div>""",
            unsafe_allow_html=True,
        )
        comment = st.text_input(
            "Comment", key=f"comment-{thread_id}", placeholder="Optional note to the employee"
        )
        approve, reject = st.columns(2)
        for col, approved, label in ((approve, True, "Approve"), (reject, False, "Reject")):
            kind = "primary" if approved else "secondary"
            if col.button(label, key=f"{label}-{thread_id}", type=kind, use_container_width=True):
                with st.spinner("Resuming the agent…"):
                    result = api.post(
                        f"/approvals/{thread_id}",
                        json={"approved": approved, "reviewer": "Demo Manager", "comment": comment or None},
                    ).json()
                if thread_id == st.session_state.thread_id:
                    st.session_state.history.append(("assistant", result))
                st.rerun()

    answers = [p for role, p in st.session_state.history if role == "assistant"]
    tools = sum(len(a.get("tools_used") or []) for a in answers)
    blocked = sum(1 for a in answers if a.get("guardrail"))
    st.markdown(
        f"""<div class="stats">
        <div class="stat"><b>{len(answers)}</b><span>Answers</span></div>
        <div class="stat"><b>{tools}</b><span>Tool calls</span></div>
        <div class="stat"><b>{blocked}</b><span>Blocked</span></div></div>""",
        unsafe_allow_html=True,
    )
