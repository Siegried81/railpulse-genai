"""
SQLite connection + safe execution layer.

Defense in depth: even if guardrails.validate_query() has a gap, the
connection itself is opened in TRUE read-only mode, so SQLite refuses
any write at the driver level regardless of what SQL text gets through.
"""

import sqlite3
import logging
import time
from pathlib import Path
from app import config
from app.guardrails import validate_query

logging.basicConfig(
    filename="query_audit.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
)

MAX_ROWS = 500  # hard cap, regardless of what the LLM's SQL asks for
# Wall-clock budget for one query. MAX_ROWS bounds what comes back, not the
# work SQLite does to produce it: a cartesian join or an unindexed aggregate
# over the million-row polling table runs to completion before fetchmany sees
# a single row. The progress handler below interrupts it instead.
MAX_QUERY_SECONDS = 5.0
# How many SQLite virtual-machine instructions run between two deadline checks.
# Small enough that a runaway query is stopped within milliseconds of the
# deadline, large enough that the check costs nothing on a normal query.
_PROGRESS_EVERY_N_OPS = 10_000


def get_connection() -> sqlite3.Connection:
    """
    Open the database in true read-only mode via SQLite's URI syntax.

    Uses Path.as_uri() rather than manual string formatting: on Windows,
    a raw f"file:{path}" with backslashes produces a malformed URI that
    SQLite silently fails to open ("unable to open database file"),
    even though the file exists. as_uri() handles the OS-specific
    conversion correctly (e.g. file:///D:/Users/.../railpulse_ai.db).
    """
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
    deadline = time.monotonic() + MAX_QUERY_SECONDS
    # A non-zero return from the handler makes SQLite abort the running
    # statement with "interrupted", which lands in the OperationalError branch.
    conn.set_progress_handler(lambda: time.monotonic() > deadline, _PROGRESS_EVERY_N_OPS)
    try:
        cursor = conn.execute(sql)
        rows = [dict(row) for row in cursor.fetchmany(MAX_ROWS)]
        logging.info(f"EXECUTED | rows_returned={len(rows)} | sql={sql!r}")
        return rows
    except sqlite3.OperationalError as e:
        # Catches any residual write attempt too: read-only DB raises here
        if time.monotonic() > deadline:
            logging.info(f"TIMEOUT | budget_s={MAX_QUERY_SECONDS} | sql={sql!r}")
            raise ValueError(
                f"Query execution failed: exceeded the {MAX_QUERY_SECONDS:g} s budget - "
                "narrow the question (a station, a day) or add a LIMIT"
            )
        logging.info(f"DB_ERROR | error={e} | sql={sql!r}")
        raise ValueError(f"Query execution failed: {e}")
    finally:
        conn.close()