import httpx
import pytest
from fastapi.testclient import TestClient

from opspilot.config import Settings
from opspilot.factory import build_agent
from opspilot.retrieval import BM25Retriever, load_policy_chunks
from opspilot.ticket_service.app import create_app

HOLIDAYS_GB_2026 = [
    {"date": "2026-01-01", "localName": "New Year's Day"},
    {"date": "2026-12-25", "localName": "Christmas Day"},
]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, ticket_db_path=tmp_path / "tickets.sqlite3")


@pytest.fixture
def service_desk(settings) -> TestClient:
    return TestClient(create_app(settings.ticket_db_path))


@pytest.fixture
def public_http() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/2026/GB"):
            return httpx.Response(200, json=HOLIDAYS_GB_2026)
        return httpx.Response(204)

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture
def retriever(settings) -> BM25Retriever:
    return BM25Retriever(load_policy_chunks(settings.policies_dir))


@pytest.fixture
def make_agent(settings, retriever, service_desk, public_http):
    def _make(model):
        return build_agent(
            settings,
            model=model,
            retriever=retriever,
            service_desk=service_desk,
            public_http=public_http,
        )

    return _make
