"""
Tests for the two independent safety layers: validate_query() and the
read-only SQLite connection. Uses a tiny throwaway database, never
data/railpulse_ai.db.
"""

import sqlite3

import pytest

from app import config, db
from app.guardrails import validate_query


@pytest.fixture
def tiny_db(tmp_path, monkeypatch):
    path = tmp_path / "tiny.db"
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE liveboard_records ("Stations Name" TEXT, delay_seconds INTEGER)')
    conn.executemany(
        "INSERT INTO liveboard_records VALUES (?, ?)",
        [("Gouvy", i) for i in range(db.MAX_ROWS + 10)],
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(config, "DB_PATH", str(path))
    return path


@pytest.mark.parametrize(
    "sql",
    [
        'SELECT "Stations Name", AVG(delay_seconds) FROM liveboard_records GROUP BY 1;',
        "select count(*) from liveboard_records",
    ],
)
def test_valid_select_passes(sql):
    assert validate_query(sql) == (True, "")


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "DROP TABLE liveboard_records",
        "SELECT 1; DELETE FROM liveboard_records",
        "SELECT 1 -- comment",
        'SELECT "weather" FROM liveboard_records',
        "SELECT * FROM pragma_table_info('liveboard_records')",
    ],
)
def test_unsafe_or_hallucinated_query_is_blocked(sql):
    is_safe, reason = validate_query(sql)
    assert not is_safe
    assert reason


def test_execute_query_returns_dicts_capped_at_max_rows(tiny_db):
    rows = db.execute_query('SELECT "Stations Name", delay_seconds FROM liveboard_records')
    assert len(rows) == db.MAX_ROWS
    assert rows[0] == {"Stations Name": "Gouvy", "delay_seconds": 0}


def test_execute_query_raises_on_blocked_sql(tiny_db):
    with pytest.raises(ValueError, match="guardrails"):
        db.execute_query("DELETE FROM liveboard_records")


def test_connection_is_read_only_even_without_guardrails(tiny_db):
    conn = db.get_connection()
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM liveboard_records")
    finally:
        conn.close()
