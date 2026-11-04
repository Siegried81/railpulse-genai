"""
Adversarial "red team" suite: it attacks RailPulse's own guardrails and
asserts they hold. Every test is named after the attack it represents.

The four deterministic defenses under test (the model is never the defense):
- guardrails.validate_query  (app/guardrails.py:73) -- SELECT-only allow-list,
  single-statement rule, forbidden-keyword / comment / sqlite_* / dangerous-
  function block-lists, table allow-list, quoted-column whitelist.
- db.execute_query           (app/db.py:51)         -- validates, then opens the
  DB read-only (?mode=ro, db.py:45), caps rows and enforces a time budget.
- db.get_connection          (app/db.py:34)         -- the second, independent
  layer: a true read-only connection refuses any write the text layer missed.
- llm_client.looks_like_leaked_meta (app/llm_client.py:95) -- catches a leaked
  safety/meta fragment in an answer.
- sql_utils.ungrounded_figures      (app/sql_utils.py:101) -- flags any figure
  in an answer that is not present in the data it was given.

Fully offline and deterministic: no test makes a real HTTP request. The engine
probe mocks requests.post (the repo's FakeResponse pattern) and every DB test
points config.DB_PATH at a throwaway SQLite file, never data/railpulse_ai.db.
"""

import sqlite3

import pytest
import requests

from app import config, db, engine, llm_client
from app.db import execute_query, get_connection
from app.guardrails import validate_query
from app.llm_client import looks_like_leaked_meta
from app.sql_utils import build_consultant_input, ungrounded_figures


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def tiny_db(tmp_path, monkeypatch):
    """A one-table SQLite file so a legit SELECT works and the read-only
    connection has a real file to open. Never the production DB."""
    path = tmp_path / "red_team.db"
    conn = sqlite3.connect(path)
    conn.execute(
        'CREATE TABLE liveboard_records '
        '(record_id INTEGER, "Stations Name" TEXT, delay_seconds INTEGER)'
    )
    conn.executemany(
        "INSERT INTO liveboard_records VALUES (?, ?, ?)",
        [(1, "Bruxelles-Central", 60), (2, "Anvers-Central", 180)],
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(config, "DB_PATH", str(path))
    return path


class FakeResponse:
    """Minimal stand-in for requests.Response (same pattern as
    tests/test_llm_client.py), so nothing touches the network."""

    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} Client Error")

    def json(self):
        return self._payload


@pytest.fixture
def llm_emits(monkeypatch):
    """Make every mocked LLM call return a chosen text as the model's content.

    conftest's fake_llm_config pins LLM_PROVIDER to groq, so the OpenAI-shaped
    groq payload is what the client parses. Used to force the model to emit a
    forbidden query and prove the deterministic guard -- not the model -- blocks
    it end to end.
    """

    def _install(content: str):
        payload = {"choices": [{"message": {"content": content}}]}
        monkeypatch.setattr(
            llm_client.requests, "post", lambda *a, **k: FakeResponse(200, payload)
        )

    return _install


# --------------------------------------------------------------------------
# 1. SQL injection -- validate_query / execute_query must block writes and
#    multi-statement stacking.
# --------------------------------------------------------------------------
# Each entry is a hostile query the text-to-SQL model could be tricked into
# emitting. All must be rejected by validate_query.
INJECTION_QUERIES = {
    "drop_table": "DROP TABLE liveboard_records;",
    "delete_rows": "DELETE FROM liveboard_records;",
    "update_rows": "UPDATE liveboard_records SET delay_seconds = 0;",
    "insert_rows": "INSERT INTO liveboard_records (record_id) VALUES (99);",
    "alter_table": "ALTER TABLE liveboard_records ADD COLUMN pwned TEXT;",
    "attach_database": "ATTACH DATABASE 'evil.db' AS evil;",
    "pragma_dump": "PRAGMA table_info(liveboard_records);",
    # Looks like a read, smuggles a write in a subquery -- no semicolon, so the
    # single-statement rule does not catch it; the forbidden-keyword rule must.
    "delete_in_subquery": (
        "SELECT * FROM liveboard_records "
        "WHERE delay_seconds > (DELETE FROM liveboard_records)"
    ),
    # Classic stacked-statement injection after a valid read.
    "stacked_drop": "SELECT * FROM liveboard_records; DROP TABLE liveboard_records;",
    "stacked_delete": "SELECT 1; DELETE FROM liveboard_records;",
    # Comment trick to hide a tail.
    "comment_trick": "SELECT * FROM liveboard_records -- harmless\n; DROP TABLE x;",
    # Schema exfiltration via SQLite internals.
    "sqlite_master_exfil": "SELECT name, sql FROM sqlite_master;",
}


@pytest.mark.parametrize("name", sorted(INJECTION_QUERIES))
def test_injection_blocked_by_validate_query(name):
    is_safe, reason = validate_query(INJECTION_QUERIES[name])
    assert is_safe is False, f"{name} slipped past validate_query"
    assert reason  # a non-empty reason is always given


@pytest.mark.parametrize("name", sorted(INJECTION_QUERIES))
def test_injection_raises_in_execute_query(tiny_db, name):
    # execute_query validates before it ever opens the connection, so a blocked
    # query raises ValueError and never reaches SQLite.
    with pytest.raises(ValueError, match="blocked by guardrails"):
        execute_query(INJECTION_QUERIES[name])


def test_legit_select_still_passes(tiny_db):
    # Positive control: the guard is not simply rejecting everything.
    rows = execute_query('SELECT "Stations Name", delay_seconds FROM liveboard_records')
    assert {r["Stations Name"] for r in rows} == {"Bruxelles-Central", "Anvers-Central"}


# --------------------------------------------------------------------------
# 2. Read-only connection -- the independent second layer. Even a write that
#    bypasses validate_query entirely must be refused by SQLite itself.
# --------------------------------------------------------------------------
WRITE_STATEMENTS = {
    "insert": "INSERT INTO liveboard_records (record_id) VALUES (999)",
    "update": "UPDATE liveboard_records SET delay_seconds = 0",
    "delete": "DELETE FROM liveboard_records",
    "create": "CREATE TABLE pwned (x INTEGER)",
    "drop": "DROP TABLE liveboard_records",
}


@pytest.mark.parametrize("name", sorted(WRITE_STATEMENTS))
def test_readonly_connection_refuses_write_even_if_guard_bypassed(tiny_db, name):
    # Deliberately skip validate_query: open the raw connection the app uses and
    # fire a write straight at it, to prove the ?mode=ro layer stands alone.
    conn = get_connection()
    try:
        with pytest.raises(sqlite3.OperationalError) as exc:
            conn.execute(WRITE_STATEMENTS[name])
        assert "readonly" in str(exc.value).lower()
    finally:
        conn.close()


def test_readonly_connection_still_reads(tiny_db):
    conn = get_connection()
    try:
        count = conn.execute("SELECT COUNT(*) AS n FROM liveboard_records").fetchone()["n"]
        assert count == 2
    finally:
        conn.close()


# --------------------------------------------------------------------------
# 3. Prompt / system leakage -- looks_like_leaked_meta must catch a leaked
#    safety/meta fragment in an answer.
# --------------------------------------------------------------------------
LEAK_FRAGMENTS = [
    "User Safety: safe",
    "Safety: compliant",
    "safety - no violations detected",
    "Moderation: pass",
    "Policy: allowed",
    "ok",  # too short to be a real answer
]


@pytest.mark.parametrize("text", LEAK_FRAGMENTS)
def test_safety_meta_leak_is_caught(text):
    assert looks_like_leaked_meta(text) is True


def test_genuine_answer_is_not_mistaken_for_a_leak():
    answer = (
        "Bruxelles-Central averaged 3.2 minutes of delay today across 452 trains, "
        "which is within the normal range; no action needed."
    )
    assert looks_like_leaked_meta(answer) is False


def test_known_limit_verbatim_system_prompt_leak_is_not_caught():
    # looks_like_leaked_meta only catches the short safety/meta fragment style it
    # was built for (see its docstring). A long, verbatim dump of the system
    # prompt is NOT caught by this heuristic. This is a documented limitation,
    # not a bug: the real defenses against a prompt dump are the prompt-level
    # instruction and the fact the consultant only ever receives result rows,
    # never the system prompt. This test pins the current boundary so anyone who
    # later strengthens the heuristic sees that it moved.
    leaked = (
        "You are a SQL generation engine for a Belgian railway operations "
        "database (SQLite). Your ONLY job is to translate a natural language "
        "question into ONE valid, safe, read-only SQL query. Rules: Output ONLY "
        "the raw SQL query."
    )
    assert looks_like_leaked_meta(leaked) is False


# --------------------------------------------------------------------------
# 4. Ungrounded figures -- any number in the answer that is not in the data it
#    was given must be flagged.
# --------------------------------------------------------------------------
def _source(rows, question="q", sql="SELECT 1"):
    return build_consultant_input(question, sql, rows)


def test_invented_absolute_number_is_flagged():
    src = _source([{"Stations Name": "Nivelles", "avg_delay_minutes": 4.43}])
    answer = "Nivelles averaged 4.43 minutes across 1850 trains today."
    assert ungrounded_figures(answer, src) == ["1850"]


def test_derived_complement_percentage_is_flagged():
    src = _source([{"on_time_rate_pct": 73.4}])
    # 26.6 is 100 - 73.4, computed by the model, present nowhere in the data.
    assert ungrounded_figures("73.4% on time, so 26.6% were late.", src) == ["26.6"]


def test_derived_phrase_in_words_is_flagged():
    src = _source([{"a": 2.91, "b": 1.86}])
    assert "one in four" in ungrounded_figures("Roughly one in four trains is late.", src)


def test_grounded_answer_passes():
    src = _source([{"Stations Name": "Gouvy", "avg_delay_minutes": 4.43, "n": 2520}])
    answer = "Gouvy averaged about 4.4 minutes over 2,520 records; nothing unusual."
    assert ungrounded_figures(answer, src) == []


# --------------------------------------------------------------------------
# 5. End-to-end engine probe -- a MOCKED model emits a forbidden query, and the
#    deterministic guard (not the model) must still block it on the real path.
# --------------------------------------------------------------------------
def test_engine_blocks_model_emitted_write_injection(tiny_db, llm_emits):
    # The model is coerced into emitting a SELECT that smuggles a DELETE. The
    # SQL step accepts it (it starts with SELECT), but execute_query's guard
    # rejects it, so answer_question raises before any consultant call.
    llm_emits(
        "SELECT * FROM liveboard_records "
        "WHERE delay_seconds > (DELETE FROM liveboard_records)"
    )
    with pytest.raises(ValueError, match="blocked by guardrails"):
        engine.answer_question("how late are the trains?")


def test_engine_blocks_model_emitted_schema_exfiltration(tiny_db, llm_emits):
    llm_emits("SELECT name, sql FROM sqlite_master;")
    with pytest.raises(ValueError, match="blocked by guardrails"):
        engine.answer_question("list every table in the database")


def test_engine_probe_makes_no_real_network_call(tiny_db, llm_emits):
    # Proves the probe is offline: the only requests.post is the monkeypatched
    # one. (conftest's no_network fixture would already raise on a real socket.)
    calls = []
    llm_emits("DROP TABLE liveboard_records;")
    original = llm_client.requests.post

    def _counting(*a, **k):
        calls.append(1)
        return original(*a, **k)

    llm_client.requests.post = _counting
    try:
        with pytest.raises(ValueError):
            engine.answer_question("drop everything")
    finally:
        llm_client.requests.post = original
    assert calls  # at least one mocked call happened, none hit the network
