"""
SQLite connection + safe execution layer.

Defense in depth: even if guardrails.validate_query() has a gap, the
connection itself is opened in TRUE read-only mode, so SQLite refuses
any write at the driver level regardless of what SQL text gets through.
"""

import sqlite3
import logging
from pathlib import Path
from app import config
from app.guardrails import validate_query

logging.basicConfig(
    filename="query_audit.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
)

MAX_ROWS = 500  # hard cap, regardless of what the LLM's SQL asks for


def get_connection() -> sqlite3.Connection:
    db_path = Path(config.DB_PATH).resolve()
    uri = db_path.as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def execute_query(sql: str) -> list[dict]:
    """
    Validate then execute a SQL query, returning rows as list of dicts.
    Raises ValueError if the query fails the safety guardrails.
    Results are capped at MAX_ROWS regardless of the query's own LIMIT.
    """
    is_safe, reason = validate_query(sql)
    if not is_safe:
        logging.info(f"BLOCKED | reason={reason} | sql={sql!r}")
        raise ValueError(f"Query blocked by guardrails: {reason}")

    conn = get_connection()
    try:
        cursor = conn.execute(sql)
        rows = [dict(row) for row in cursor.fetchmany(MAX_ROWS)]
        logging.info(f"EXECUTED | rows_returned={len(rows)} | sql={sql!r}")
        return rows
    except sqlite3.OperationalError as e:
        # Catches any residual write attempt too: read-only DB raises here
        logging.info(f"DB_ERROR | error={e} | sql={sql!r}")
        raise ValueError(f"Query execution failed: {e}")
    finally:
        conn.close()