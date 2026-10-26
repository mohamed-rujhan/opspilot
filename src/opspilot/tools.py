"""Tools the agent can call. Each tool is a thin, validated wrapper around a retriever or a REST API."""

from datetime import date
from typing import Literal

import httpx
from langchain_core.tools import BaseTool, tool
from langgraph.types import interrupt

from opspilot.guardrails import contains_injection
from opspilot.retrieval import Retriever

# Only these tools are ever bound to the model (tool allow-list).
TOOL_NAMES = (
    "search_policies",
    "create_ticket",
    "get_ticket_status",
    "get_public_holidays",
    "request_leave",
)


def make_tools(
    retriever: Retriever,
    service_desk: httpx.Client,
    public_http: httpx.Client,
    *,
    holiday_api_url: str,
    default_country: str,
    top_k: int = 3,
) -> list[BaseTool]:
    @tool
    def search_policies(query: str) -> str:
        """Search the company HR and IT policy documents. Use for any question about leave, remote work,
        IT security, IT support or expenses. Returns the most relevant policy excerpts with their sources."""
        results = [r for r in retriever.search(query, k=top_k) if not contains_injection(r.chunk.text)]
        if not results:
            return "No relevant policy excerpts found."
        return "\n\n".join(f"[source: {r.chunk.citation}]\n{r.chunk.text}" for r in results)

    @tool
    def create_ticket(title: str, description: str, priority: Literal["P1", "P2", "P3", "P4"] = "P3") -> str:
        """Create an IT service desk ticket. Priority: P1 security incident or lost/stolen device,
        P2 cannot work, P3 broken with a workaround, P4 question or small request."""
        response = service_desk.post(
            "/tickets", json={"title": title, "description": description, "priority": priority}
        )
        response.raise_for_status()
        ticket = response.json()
        return f"Created ticket #{ticket['id']} ({ticket['priority']}): {ticket['title']}"

    @tool
    def get_ticket_status(ticket_id: int) -> str:
        """Look up the current status of an IT service desk ticket by its number."""
        response = service_desk.get(f"/tickets/{ticket_id}")
        if response.status_code == 404:
            return f"Ticket #{ticket_id} was not found."
        response.raise_for_status()
        t = response.json()
        return f"Ticket #{t['id']} '{t['title']}' is {t['status']} (priority {t['priority']})."

    @tool
    def get_public_holidays(year: int, country_code: str | None = None) -> str:
        """List public holidays for a year. country_code is an ISO 3166-1 alpha-2 code such as GB or DE;
        defaults to the company's home country."""
        country = (country_code or default_country).upper()
        response = public_http.get(f"{holiday_api_url}/PublicHolidays/{year}/{country}")
        if response.status_code in (204, 404):
            return f"No public holiday data is available for {country}."
        response.raise_for_status()
        holidays = response.json()
        lines = [f"- {h['date']}: {h['localName']}" for h in holidays]
        return f"Public holidays in {country} for {year}:\n" + "\n".join(lines)

    @tool
    def request_leave(
        leave_type: Literal["annual", "sick", "unpaid"],
        start_date: date,
        end_date: date,
        reason: str,
    ) -> str:
        """Submit a leave request. The request is paused until a manager approves or rejects it."""
        if end_date < start_date:
            return "The end date is before the start date. Please check the dates."
        decision = interrupt(
            {
                "action": "request_leave",
                "leave_type": leave_type,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "reason": reason,
            }
        )
        reviewer = decision.get("reviewer", "manager")
        if not decision.get("approved"):
            comment = decision.get("comment") or "no comment"
            return f"The leave request was rejected by {reviewer} ({comment})."
        response = service_desk.post(
            "/leave",
            json={
                "leave_type": leave_type,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "reason": reason,
                "approved_by": reviewer,
            },
        )
        response.raise_for_status()
        return f"The leave request was approved by {reviewer} and recorded as #{response.json()['id']}."

    tools = [search_policies, create_ticket, get_ticket_status, get_public_holidays, request_leave]
    assert tuple(t.name for t in tools) == TOOL_NAMES
    return tools
