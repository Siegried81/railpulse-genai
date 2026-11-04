"""
The text-to-SQL + consultant pipeline as one reusable function.

Why this exists: the pipeline used to live inline inside the Streamlit page
(`app/streamlit_app.py`). The JSON API in `api/main.py` needs the exact same
steps, and two copies drift. `answer_question()` is the single path both the UI
and the API call, so a change to the pipeline is made once and seen by both.

The function keeps every safety property of the original inline code:
- exactly one structured LLM call per step, no agent loop (the two
  `call_llm_until` calls are each a single fixed prompt with bounded retry);
- the query is run through `execute_query`, which validates via the guardrails
  and opens the DB read-only;
- the consultant answer is still checked with `ungrounded_figures`, and the
  figures it flags are returned so the caller can caveat the answer;
- `LAST_ANSWERED_BY` is recorded for both the SQL and the answer step.

Outcomes:
- A normal answer returns the dict below with `sql` and `rows` populated.
- A `NO_QUERY:` verdict from the SQL step is a normal result too: `answer`
  holds the explanation, `sql` and `rows` are None. It is not an error.
- The LLM backend being unreachable raises `LLMConnectionError`.
- The query being blocked by the guardrails or exceeding the time budget raises
  `ValueError` (from `execute_query`).

The caller decides how to present those: the Streamlit page renders them, the
API maps the two exceptions to HTTP 503 and 422.
"""

# llm_client and db are referenced through the module (llm_client.call_llm_until,
# db.execute_query) rather than imported by name, so a test can monkeypatch them
# at their canonical location and the engine picks up the patch.
from app import llm_client, db
from app.llm_client import LLMConnectionError
from app.prompts import (
    CONSULTANT_MAX_TOKENS,
    CONSULTANT_SYSTEM_PROMPT,
    SQL_MAX_TOKENS,
    TEXT_TO_SQL_SYSTEM_PROMPT,
)
from app.sql_utils import extract_sql, build_consultant_input, ungrounded_figures


def answer_question(question: str) -> dict:
    """Run the full pipeline for one question and return a structured result.

    Returns a dict with:
    - answer: the text shown to the user. For a NO_QUERY verdict it is the
      explanation of why the question cannot be answered; otherwise it is the
      consultant's raw answer, without the ungrounded-figures caveat appended
      (the caller appends that from `ungrounded` so both surfaces word it the
      same way).
    - sql: the executed SELECT, or None for a NO_QUERY verdict.
    - rows: the raw result rows (delays in seconds), or None for NO_QUERY. The
      caller converts to minutes with sql_utils.rows_in_minutes for display.
    - sql_by: "provider · model" that produced the SQL (LAST_ANSWERED_BY).
    - answer_by: "provider · model" that produced the answer, or None for
      NO_QUERY (the consultant step does not run then).
    - ungrounded: list of figures in the answer not found in the consultant
      input (empty when the answer is grounded).

    Raises LLMConnectionError if the backend can't be reached, and ValueError
    if execute_query blocks the query or it exceeds the time budget. On an
    LLMConnectionError raised after the SQL ran, the exception carries `.sql`
    and `.rows` so the UI can still show what was computed before the answer
    step failed; the API ignores those attributes and just maps the type.
    """
    # --- Step 1: text-to-SQL ---
    # stop=(";",) halts generation right after the query, which is both faster
    # and safer (no stray text after the semicolon that could trip the
    # "multiple statements" guardrail). call_llm_until retries (bypassing the
    # cache) when a free-tier backend leaks a stray meta/safety line instead of
    # valid SQL or NO_QUERY, even at temperature=0.
    try:
        raw_sql = llm_client.call_llm_until(
            TEXT_TO_SQL_SYSTEM_PROMPT,
            question,
            is_valid=lambda s: extract_sql(s).upper().startswith(("SELECT", "NO_QUERY")),
            stop=(";",),
            max_tokens=SQL_MAX_TOKENS,
        )
    except LLMConnectionError as e:
        e.sql = None
        e.rows = None
        raise
    sql = extract_sql(raw_sql)
    sql_by = llm_client.LAST_ANSWERED_BY

    if sql.upper().startswith("NO_QUERY"):
        reason = sql.split(":", 1)[-1].strip()
        return {
            "answer": f"I can't answer that with the data I have access to. {reason}",
            "sql": None,
            "rows": None,
            "sql_by": sql_by,
            "answer_by": None,
            "ungrounded": [],
        }

    # --- Query (validated + read-only inside execute_query; raises ValueError
    # when blocked or over the time budget). `.sql` is attached so the UI can
    # still show the rejected query; the API ignores it and maps to 422. ---
    try:
        rows = db.execute_query(sql)
    except ValueError as e:
        e.sql = sql
        raise

    # --- Step 2: consultant explanation ---
    consultant_input = build_consultant_input(question, sql, rows)
    try:
        answer = llm_client.call_llm_until(
            CONSULTANT_SYSTEM_PROMPT,
            consultant_input,
            is_valid=lambda s: not llm_client.looks_like_leaked_meta(s)
            and not ungrounded_figures(s, consultant_input),
            max_tokens=CONSULTANT_MAX_TOKENS,
        )
    except LLMConnectionError as e:
        e.sql = sql
        e.rows = rows
        raise
    answer_by = llm_client.LAST_ANSWERED_BY

    # call_llm_until returns its last attempt even if still invalid; report the
    # ungrounded figures rather than hide them.
    ungrounded = ungrounded_figures(answer, consultant_input)

    return {
        "answer": answer,
        "sql": sql,
        "rows": rows,
        "sql_by": sql_by,
        "answer_by": answer_by,
        "ungrounded": ungrounded,
    }
