"""Demo UI for OpsPilot: an employee chat on the left, a manager approval queue on the right.

streamlit run app/streamlit_app.py
"""

import os

import httpx
import streamlit as st

API_URL = os.getenv("OPSPILOT_API_URL", "http://localhost:8000")

st.set_page_config(page_title="OpsPilot", page_icon="🧭", layout="wide")
st.markdown("<style>.block-container {padding-top: 1.5rem;}</style>", unsafe_allow_html=True)
st.title("🧭 OpsPilot")
st.caption("Internal operations assistant for Lumen Analytics Ltd (a fictional company)")

api = httpx.Client(base_url=API_URL, timeout=90)

st.session_state.setdefault("thread_id", None)
st.session_state.setdefault("history", [])
st.session_state.setdefault("awaiting", None)


def show_result(result: dict) -> None:
    st.markdown(result.get("answer") or "")
    details = []
    if result.get("tools_used"):
        details.append("🛠 " + ", ".join(result["tools_used"]))
    if result.get("sources"):
        details.append("📄 " + ", ".join(result["sources"]))
    if result.get("guardrail"):
        details.append(f"🛡 guardrail: {result['guardrail']}")
    if result.get("latency_ms"):
        details.append(f"⏱ {result['latency_ms']} ms")
    if details:
        st.caption(" · ".join(details))


def send(message: str) -> dict:
    response = api.post("/chat", json={"message": message, "thread_id": st.session_state.thread_id})
    if response.status_code != 200:
        return {"answer": f"⚠️ {response.json().get('detail', response.text)}"}
    result = response.json()
    st.session_state.thread_id = result["thread_id"]
    return result


chat_col, manager_col = st.columns([2, 1], gap="large")

with chat_col:
    header, reset = st.columns([4, 1], vertical_alignment="center")
    header.subheader("Employee chat")
    if reset.button("New chat", use_container_width=True):
        st.session_state.update(thread_id=None, history=[], awaiting=None)
        st.rerun()

    messages = st.container(height=380, autoscroll=True)
    with messages:
        if not st.session_state.history:
            st.caption(
                "Try: “How many days of annual leave do I get?” or “My laptop was stolen, raise a ticket.”"
            )
        for role, payload in st.session_state.history:
            with st.chat_message(role):
                if role == "assistant":
                    show_result(payload)
                else:
                    st.markdown(payload)
        if st.session_state.awaiting:
            with st.chat_message("assistant"), st.spinner("Thinking…"):
                result = send(st.session_state.awaiting)
            st.session_state.history.append(("assistant", result))
            st.session_state.awaiting = None
            st.rerun()

    if prompt := st.chat_input("Ask about leave, IT, expenses… or raise a ticket"):
        st.session_state.history.append(("user", prompt))
        st.session_state.awaiting = prompt
        st.rerun()

with manager_col:
    st.subheader("Manager approvals")
    try:
        pending = api.get("/approvals").json()
    except httpx.HTTPError:
        pending = {}
    if not pending:
        st.info("No requests waiting for approval.")
    for thread_id, action in pending.items():
        with st.container(border=True):
            st.markdown(
                f"**{action['leave_type'].title()} leave** · "
                f"{action['start_date']} → {action['end_date']}  \n"
                f"Reason: {action['reason']}"
            )
            comment = st.text_input("Comment", key=f"comment-{thread_id}")
            approve, reject = st.columns(2)
            for col, approved, label in ((approve, True, "Approve"), (reject, False, "Reject")):
                if col.button(label, key=f"{label}-{thread_id}", use_container_width=True):
                    with st.spinner("Resuming the agent…"):
                        result = api.post(
                            f"/approvals/{thread_id}",
                            json={
                                "approved": approved,
                                "reviewer": "Demo Manager",
                                "comment": comment or None,
                            },
                        ).json()
                    if thread_id == st.session_state.thread_id:
                        st.session_state.history.append(("assistant", result))
                    st.rerun()
