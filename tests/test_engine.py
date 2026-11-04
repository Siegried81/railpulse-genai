"""
Tests for app/engine.py: the shared text-to-SQL + consultant pipeline.

Every LLM call is mocked at requests.post (so the real call_llm_until retry and
LAST_ANSWERED_BY wiring is exercised, nothing reaches the network), and the
database is a tiny throwaway SQLite file, never data/railpulse_ai.db.
"""

import sqlite3

import pytest
import requests

from app import config, db, llm_client
from app.engine import answer_question
from app.llm_client import LLMConnectionError


class FakeResponse:
    """Minimal stand-in for requests.Response (same shape as test_llm_client)."""

    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} Client Error")

    def json(self):
        return self._payload


def _groq_reply(text: str) -> FakeResponse:
    """A 200 Groq-shaped response carrying `text` as the message content."""
    return FakeResponse(200, {"choices": [{"message": {"content": text}}]})


@pytest.fixture
def fake_post(monkeypatch):
    """Queue of responses (or exceptions) returned by successive requests.post calls."""
    queue = []

    def _post(*args, **kwargs):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(llm_client.requests, "post", _post)
    return queue


@pytest.fixture
def tiny_db(tmp_path, monkeypatch):
    """A two-row liveboard_records table, with config.DB_PATH pointed at it."""
    path = tmp_path / "tiny.db"
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE liveboard_records ("Stations Name" TEXT, delay_seconds INTEGER)')
    conn.executemany(
        "INSERT INTO liveboard_records VALUES (?, ?)",
        [("Gouvy", 600), ("Gouvy", 90)],
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(config, "DB_PATH", str(path))
    return path


SELECT_SQL = 'SELECT "Stations Name", delay_seconds FROM liveboard_records;'


def test_answer_question_success_path(fake_post, tiny_db):
    """SQL step, read-only query, consultant step: all fields come back filled."""
    fake_post.append(_groq_reply(SELECT_SQL))
    # 10 min (600s) and 1.5 min (90s) are in the consultant input, so grounded.
    fake_post.append(_groq_reply("Gouvy saw delays up to 10 minutes; worth a look."))

    result = answer_question("Delays at Gouvy?")

    assert result["sql"] == SELECT_SQL
    assert result["rows"] == [
        {"Stations Name": "Gouvy", "delay_seconds": 600},
        {"Stations Name": "Gouvy", "delay_seconds": 90},
    ]
    assert result["answer"].startswith("Gouvy saw delays")
    assert result["sql_by"] == "groq · openai/gpt-oss-120b"
    assert result["answer_by"] == "groq · openai/gpt-oss-120b"
    assert result["ungrounded"] == []


def test_answer_question_flags_ungrounded_figures(fake_post, tiny_db):
    """A figure the model invents (not in the rows) is returned in `ungrounded`."""
    fake_post.append(_groq_reply(SELECT_SQL))
    # 42 is nowhere in the consultant input; call_llm_until exhausts its retries
    # and returns the last attempt, and the engine reports the ungrounded figure.
    bad = "Gouvy had about 42 severe incidents this week across the network."
    fake_post.extend([_groq_reply(bad), _groq_reply(bad), _groq_reply(bad)])

    result = answer_question("Delays at Gouvy?")

    assert "42" in result["ungrounded"]


def test_no_query_is_a_normal_result(fake_post, tiny_db):
    """A NO_QUERY verdict returns an explanation with null sql/rows, not an error.

    Only the SQL step runs: the consultant is never called, so a single mocked
    response is enough.
    """
    fake_post.append(_groq_reply("NO_QUERY: platform-level data is unavailable."))

    result = answer_question("Which platform at Gouvy is worst?")

    assert result["sql"] is None
    assert result["rows"] is None
    assert result["answer_by"] is None
    assert result["ungrounded"] == []
    assert "platform-level data is unavailable" in result["answer"]


def test_guardrail_block_raises_valueerror_carrying_sql(fake_post, tiny_db):
    """A SELECT that passes is_valid but references an unknown column is blocked
    by the guardrails inside execute_query; the ValueError carries the SQL."""
    blocked = 'SELECT "weather" FROM liveboard_records;'
    fake_post.append(_groq_reply(blocked))

    with pytest.raises(ValueError) as exc_info:
        answer_question("What was the weather?")

    assert "guardrails" in str(exc_info.value)
    assert exc_info.value.sql == blocked


def test_llm_unreachable_on_sql_step_raises_with_null_sql(fake_post, tiny_db):
    """If the backend can't be reached for the SQL step, no SQL ran."""
    fake_post.append(requests.exceptions.ConnectionError("down"))

    with pytest.raises(LLMConnectionError) as exc_info:
        answer_question("Delays at Gouvy?")

    assert exc_info.value.sql is None
    assert exc_info.value.rows is None


def test_llm_unreachable_on_answer_step_keeps_sql_and_rows(fake_post, tiny_db):
    """If the SQL ran but the consultant backend is unreachable, the exception
    carries the SQL and rows so the UI can still show them."""
    fake_post.append(_groq_reply(SELECT_SQL))
    fake_post.append(requests.exceptions.ConnectionError("down"))

    with pytest.raises(LLMConnectionError) as exc_info:
        answer_question("Delays at Gouvy?")

    assert exc_info.value.sql == SELECT_SQL
    assert exc_info.value.rows == [
        {"Stations Name": "Gouvy", "delay_seconds": 600},
        {"Stations Name": "Gouvy", "delay_seconds": 90},
    ]
