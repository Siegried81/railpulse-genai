"""
Offline tests for the FastAPI layer (api/main.py).

The mapping tests stub `answer_question` so each HTTP outcome is checked in
isolation. The integration test drives the real pipeline with requests.post
mocked and a tiny throwaway SQLite DB, so it proves the API-to-engine wiring,
the read-only guardrailed query, and the ungrounded check all still hold on the
HTTP path. Nothing reaches the network and the real DB is never touched.
"""

import sqlite3

import pytest
import requests
from fastapi.testclient import TestClient

from api import main as api
from app import config, llm_client
from app.llm_client import LLMConnectionError


@pytest.fixture
def client():
    return TestClient(api.app)


# --- Liveness and config -----------------------------------------------------


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200 and resp.json() == {"status": "ok"}


def test_config_reports_provider_and_no_secret(client, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(config, "GROQ_API_KEY", "sk-secret-groq")
    monkeypatch.setattr(config, "GROQ_API_KEYS", ["sk-secret-groq"])
    resp = client.get("/api/config")
    assert resp.status_code == 200
    body = resp.json()
    assert "sk-secret" not in resp.text
    assert body["llm_provider"] == "groq"
    assert body["llm_model"] == config.GROQ_MODEL


# --- /ask outcome mapping (answer_question stubbed) --------------------------


def _result(**over):
    base = {
        "answer": "Gouvy saw delays up to 10 minutes.",
        "sql": 'SELECT "Stations Name" FROM liveboard_records;',
        "rows": [{"Stations Name": "Gouvy"}],
        "sql_by": "groq · openai/gpt-oss-120b",
        "answer_by": "groq · openai/gpt-oss-120b",
        "ungrounded": [],
    }
    base.update(over)
    return base


def test_ask_answer_path(client, monkeypatch):
    monkeypatch.setattr(api, "answer_question", lambda q: _result())
    resp = client.post("/ask", json={"question": "Delays at Gouvy?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["no_query"] is False
    assert body["answer"].startswith("Gouvy saw delays")
    assert body["sql"].startswith("SELECT")
    assert body["rows"] == [{"Stations Name": "Gouvy"}]
    assert body["sql_by"] == "groq · openai/gpt-oss-120b"
    assert body["ungrounded"] == []


def test_ask_no_query_is_200_not_error(client, monkeypatch):
    monkeypatch.setattr(
        api, "answer_question",
        lambda q: _result(
            answer="I can't answer that with the data I have access to. Platform data is unavailable.",
            sql=None, rows=None, answer_by=None,
        ),
    )
    resp = client.post("/ask", json={"question": "Worst platform at Gouvy?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["no_query"] is True
    assert body["sql"] is None and body["rows"] is None
    assert "Platform data is unavailable" in body["answer"]


def test_ask_llm_unreachable_is_503(client, monkeypatch):
    def _raise(q):
        raise LLMConnectionError("backend down")

    monkeypatch.setattr(api, "answer_question", _raise)
    resp = client.post("/ask", json={"question": "Delays at Gouvy?"})
    assert resp.status_code == 503
    assert "backend down" in resp.json()["detail"]


def test_ask_guardrail_block_is_422(client, monkeypatch):
    def _raise(q):
        raise ValueError("Query blocked by guardrails: unknown column")

    monkeypatch.setattr(api, "answer_question", _raise)
    resp = client.post("/ask", json={"question": "Weather at Gouvy?"})
    assert resp.status_code == 422
    assert "guardrails" in resp.json()["detail"]


@pytest.mark.parametrize("question", ["", "x" * 501])
def test_ask_rejects_bad_length(client, question):
    """Empty and over-500-character questions fail Pydantic validation (422)."""
    resp = client.post("/ask", json={"question": question})
    assert resp.status_code == 422


# --- Integration: real engine, requests.post + DB mocked ---------------------


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} Client Error")

    def json(self):
        return self._payload


def test_ask_end_to_end_through_real_engine(client, monkeypatch, tmp_path):
    """POST /ask drives the whole pipeline: a generated SELECT is run read-only
    against a tiny DB and the consultant answer comes back, all over HTTP."""
    path = tmp_path / "tiny.db"
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE liveboard_records ("Stations Name" TEXT, delay_seconds INTEGER)')
    conn.executemany("INSERT INTO liveboard_records VALUES (?, ?)", [("Gouvy", 600), ("Gouvy", 90)])
    conn.commit()
    conn.close()
    monkeypatch.setattr(config, "DB_PATH", str(path))

    queue = [
        FakeResponse(200, {"choices": [{"message": {
            "content": 'SELECT "Stations Name", delay_seconds FROM liveboard_records;'}}]}),
        FakeResponse(200, {"choices": [{"message": {
            "content": "Gouvy saw delays up to 10 minutes; worth a look."}}]}),
    ]
    monkeypatch.setattr(llm_client.requests, "post", lambda *a, **k: queue.pop(0))

    resp = client.post("/ask", json={"question": "Delays at Gouvy?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["no_query"] is False
    assert body["rows"] == [
        {"Stations Name": "Gouvy", "delay_seconds": 600},
        {"Stations Name": "Gouvy", "delay_seconds": 90},
    ]
    assert body["ungrounded"] == []
    assert body["answer_by"] == "groq · openai/gpt-oss-120b"


# --- API protection: off by default, on via env (for a public deploy) --------


@pytest.fixture
def _reset_rate_limit():
    """Clear the in-memory rate-limit store around a test that turns it on."""
    api._rate_hits.clear()
    yield
    api._rate_hits.clear()


def test_health_stays_open_even_with_a_key_set(client, monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "secret")
    assert client.get("/api/health").status_code == 200


def test_ask_is_open_when_no_key_is_configured(client, monkeypatch):
    # The default: no key set -> the endpoint answers without a header (local dev).
    monkeypatch.setattr(config, "API_KEY", "")
    monkeypatch.setattr(api, "answer_question", lambda q: _result())
    assert client.post("/ask", json={"question": "Delays at Gouvy?"}).status_code == 200


def test_ask_401_without_or_with_wrong_key_when_configured(client, monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "secret")
    monkeypatch.setattr(api, "answer_question", lambda q: _result())
    assert client.post("/ask", json={"question": "Delays at Gouvy?"}).status_code == 401
    assert client.post(
        "/ask", json={"question": "Delays at Gouvy?"}, headers={"X-API-Key": "wrong"}
    ).status_code == 401


def test_ask_200_with_the_correct_key(client, monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "secret")
    monkeypatch.setattr(api, "answer_question", lambda q: _result())
    resp = client.post(
        "/ask", json={"question": "Delays at Gouvy?"}, headers={"X-API-Key": "secret"}
    )
    assert resp.status_code == 200
    assert resp.json()["answer"].startswith("Gouvy saw delays")


def test_rate_limit_returns_429_past_the_cap(client, monkeypatch, _reset_rate_limit):
    monkeypatch.setattr(config, "API_KEY", "")  # no key gate, so this isolates the limiter
    monkeypatch.setattr(config, "RATE_LIMIT_PER_MINUTE", 3)
    monkeypatch.setattr(api, "answer_question", lambda q: _result())
    for _ in range(3):
        assert client.post("/ask", json={"question": "Delays at Gouvy?"}).status_code == 200
    assert client.post("/ask", json={"question": "Delays at Gouvy?"}).status_code == 429
