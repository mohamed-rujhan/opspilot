"""Mock service desk standing in for a real enterprise system (e.g. ServiceNow / Jira Service Management).

Run it with:  uvicorn opspilot.ticket_service.app:app --port 8001
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from opspilot.config import get_settings

Priority = Literal["P1", "P2", "P3", "P4"]
TicketStatus = Literal["open", "in_progress", "resolved", "closed"]
LeaveType = Literal["annual", "sick", "unpaid"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    priority TEXT NOT NULL,
    status TEXT NOT NULL,
    requester TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS leave_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee TEXT NOT NULL,
    leave_type TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    reason TEXT NOT NULL,
    approved_by TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class TicketCreate(BaseModel):
    title: str = Field(min_length=3, max_length=120)
    description: str = Field(min_length=3, max_length=2000)
    priority: Priority = "P3"
    requester: str = "employee"


class Ticket(TicketCreate):
    id: int
    status: TicketStatus
    created_at: datetime


class LeaveCreate(BaseModel):
    employee: str = "employee"
    leave_type: LeaveType
    start_date: date
    end_date: date
    reason: str = Field(min_length=2, max_length=500)
    approved_by: str

    @model_validator(mode="after")
    def check_dates(self) -> "LeaveCreate":
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        return self


class LeaveRequest(LeaveCreate):
    id: int
    created_at: datetime


def create_app(db_path: Path | None = None) -> FastAPI:
    path = Path(db_path or get_settings().ticket_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect() -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    with connect() as conn:
        conn.executescript(SCHEMA)

    app = FastAPI(title="OpsPilot mock service desk", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/tickets", status_code=status.HTTP_201_CREATED)
    def create_ticket(body: TicketCreate) -> Ticket:
        now = datetime.now(UTC)
        with connect() as conn:
            cur = conn.execute(
                "INSERT INTO tickets (title, description, priority, status, requester, created_at)"
                " VALUES (?, ?, ?, 'open', ?, ?)",
                (body.title, body.description, body.priority, body.requester, now.isoformat()),
            )
            ticket_id = cur.lastrowid
        return Ticket(id=ticket_id, status="open", created_at=now, **body.model_dump())

    @app.get("/tickets/{ticket_id}")
    def get_ticket(ticket_id: int) -> Ticket:
        with connect() as conn:
            row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")
        return Ticket(**dict(row))

    @app.post("/leave", status_code=status.HTTP_201_CREATED)
    def create_leave(body: LeaveCreate) -> LeaveRequest:
        now = datetime.now(UTC)
        with connect() as conn:
            cur = conn.execute(
                "INSERT INTO leave_requests"
                " (employee, leave_type, start_date, end_date, reason, approved_by, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    body.employee,
                    body.leave_type,
                    body.start_date.isoformat(),
                    body.end_date.isoformat(),
                    body.reason,
                    body.approved_by,
                    now.isoformat(),
                ),
            )
            leave_id = cur.lastrowid
        return LeaveRequest(id=leave_id, created_at=now, **body.model_dump())

    return app


app = create_app()
