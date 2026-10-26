"""Demo UI for OpsPilot: an employee chat on the left, a manager approval queue on the right.

streamlit run app/streamlit_app.py
"""

import os

import httpx
import streamlit as st

API_URL = os.getenv("OPSPILOT_API_URL", "http://localhost:8000")

st.set_page_config(page_title="OpsPilot", page_icon="🧭", layout="wide")
st.title("🧭 OpsPilot")
st.caption("Internal operations assistant for Lumen Analytics Ltd (a fictional company)")

api = httpx.Client(base_url=API_URL, timeout=60)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
    st.session_state.history = []


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


chat_col, manager_col = st.columns([2, 1])

with chat_col:
    st.subheader("Employee chat")
    if st.button("New conversation"):
        st.session_state.thread_id = None
        st.session_state.history = []

    for role, payload in st.session_state.history:
        with st.chat_message(role):
            show_result(payload) if role == "assistant" else st.markdown(payload)

    if prompt := st.chat_input("Ask about leave, IT, expenses… or raise a ticket"):
        st.session_state.history.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        response = api.post("/chat", json={"message": prompt, "thread_id": st.session_state.thread_id})
        if response.status_code != 200:
            st.error(response.json().get("detail", response.text))
        else:
            result = response.json()
            st.session_state.thread_id = result["thread_id"]
            st.session_state.history.append(("assistant", result))
            with st.chat_message("assistant"):
                show_result(result)

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
                    result = api.post(
                        f"/approvals/{thread_id}",
                        json={"approved": approved, "reviewer": "Demo Manager", "comment": comment or None},
                    ).json()
                    if thread_id == st.session_state.thread_id:
                        st.session_state.history.append(("assistant", result))
                    st.rerun()
