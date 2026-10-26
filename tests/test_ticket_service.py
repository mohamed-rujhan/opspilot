def test_create_and_fetch_ticket(service_desk):
    created = service_desk.post(
        "/tickets",
        json={"title": "VPN not connecting", "description": "Error 809 since this morning", "priority": "P2"},
    )
    assert created.status_code == 201
    ticket = created.json()
    assert ticket["status"] == "open"

    fetched = service_desk.get(f"/tickets/{ticket['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["title"] == "VPN not connecting"


def test_unknown_ticket_returns_404(service_desk):
    assert service_desk.get("/tickets/999").status_code == 404


def test_invalid_priority_is_rejected(service_desk):
    response = service_desk.post(
        "/tickets", json={"title": "Printer", "description": "Jammed", "priority": "urgent"}
    )
    assert response.status_code == 422


def test_leave_end_date_must_not_be_before_start(service_desk):
    response = service_desk.post(
        "/leave",
        json={
            "leave_type": "annual",
            "start_date": "2026-12-10",
            "end_date": "2026-12-01",
            "reason": "Holiday",
            "approved_by": "Jane",
        },
    )
    assert response.status_code == 422
