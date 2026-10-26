import pytest
from fastapi.testclient import TestClient

from opspilot.api.main import app
from tests.fakes import ScriptedChatModel, answer, tool_call


@pytest.fixture
def client(make_agent):
    model = ScriptedChatModel(
        steps=[
            tool_call(
                "request_leave",
                {
                    "leave_type": "annual",
                    "start_date": "2026-08-03",
                    "end_date": "2026-08-07",
                    "reason": "Summer holiday",
                },
            ),
            answer("Approved and recorded."),
        ]
    )
    app.state.agent = make_agent(model)
    yield TestClient(app)
    app.state.agent = None


def test_chat_approval_flow(client):
    pending = client.post("/chat", json={"message": "Book leave 3-7 August 2026, summer holiday"})
    assert pending.status_code == 200
    body = pending.json()
    assert body["status"] == "pending_approval"

    approvals = client.get("/approvals").json()
    assert body["thread_id"] in approvals

    decision = client.post(
        f"/approvals/{body['thread_id']}", json={"approved": True, "reviewer": "Jane Manager"}
    )
    assert decision.status_code == 200
    assert decision.json()["status"] == "answered"


def test_unknown_approval_returns_404(client):
    response = client.post("/approvals/nope", json={"approved": True, "reviewer": "Jane"})
    assert response.status_code == 404


def test_chat_returns_503_when_llm_not_configured():
    app.state.agent = None
    response = TestClient(app).post("/chat", json={"message": "hi"})
    assert response.status_code == 503
