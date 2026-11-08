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
from app.guardrails import quoted_identifiers, referenced_tables, validate_query

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


def _quoted_column_not_in_query_tables(conn: sqlite3.Connection, sql: str) -> str | None:
    """Return the first double-quoted identifier that no table in the query has.

    validate_query() checks quoted names against one flat list of every column
    in the schema. That stops an invented name, but not a real column asked of
    the wrong table: "train_category" lives in routes, and SQLite reads a
    double-quoted name it cannot resolve as a string literal. So
    `SELECT "train_category", AVG(delay_seconds) FROM liveboard_records
    GROUP BY "train_category"` would run and return one group, labelled
    'train_category', holding the network-wide average. This checks each
    quoted name against the columns of the tables the query actually reads,
    taken from the database itself rather than a hand-written copy.
    """
    columns = set()
    for table in {t.lower() for t in referenced_tables(sql)}:
        # Table names come from the allow-list already enforced by
        # validate_query, so they are safe to put in the PRAGMA.
        columns.update(row[1].lower() for row in conn.execute(f'PRAGMA table_info("{table}")'))
    for identifier in quoted_identifiers(sql):
        if identifier.lower() not in columns:
            return identifier
    return None


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
        stray = _quoted_column_not_in_query_tables(conn, sql)
    except Exception:
        conn.close()
        raise
    if stray is not None:
        conn.close()
        reason = f'"{stray}" is not a column of any table this query reads'
        logging.info(f"BLOCKED | reason={reason} | sql={sql!r}")
        raise ValueError(f"Query blocked by guardrails: {reason}")

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