"""
Quick CLI test harness for the RailPulse AI pipeline.

Usage:
    python test_pipeline.py "Which station had the worst average delay this week?"
    python test_pipeline.py             (runs the full smoke-test suite below)
"""

import sys
import time
from app import llm_client
from app.llm_client import call_llm_until, looks_like_leaked_meta, LLMConnectionError
from app.db import execute_query
from app.prompts import (
    CONSULTANT_MAX_TOKENS,
    CONSULTANT_SYSTEM_PROMPT,
    SQL_MAX_TOKENS,
    TEXT_TO_SQL_SYSTEM_PROMPT,
)
from app.sql_utils import extract_sql, build_consultant_input, ungrounded_figures


def ask(question: str) -> None:
    print(f"\n🗣️  Question: {question}\n")

    raw_sql = call_llm_until(
        TEXT_TO_SQL_SYSTEM_PROMPT,
        question,
        is_valid=lambda s: extract_sql(s).upper().startswith(("SELECT", "NO_QUERY")),
        stop=(";",),
        max_tokens=SQL_MAX_TOKENS,
    )
    sql = extract_sql(raw_sql)
    print(f"🤖 SQL written by: {llm_client.LAST_ANSWERED_BY}")

    if sql.upper().startswith("NO_QUERY"):
        print(f"🚫 Out of scope: {sql}\n")
        return

    print(f"🔎 Generated SQL:\n{sql}\n")

    try:
        rows = execute_query(sql)
    except ValueError as e:
        print(f"⛔ Blocked: {e}")
        return

    print(f"📊 Raw results ({len(rows)} rows): {rows[:5]}\n")

    consultant_input = build_consultant_input(question, sql, rows)
    answer = call_llm_until(
        CONSULTANT_SYSTEM_PROMPT,
        consultant_input,
        is_valid=lambda s: not looks_like_leaked_meta(s) and not ungrounded_figures(s, consultant_input),
        max_tokens=CONSULTANT_MAX_TOKENS,
    )
    print(f"💬 RailPulse Consultant says ({llm_client.LAST_ANSWERED_BY}):\n{answer}\n")
    ungrounded = ungrounded_figures(answer, consultant_input)
    if ungrounded:
        print(f"⚠️  UNGROUNDED figures (not in the results): {ungrounded}\n")


SMOKE_TEST_QUESTIONS = [
    # Core delay / ranking questions
    "Which station had the worst average delay this week?",
    "What is the average delay in minutes for Bruxelles-Central today?",
    "Show me the 10 most delayed trains at Anvers-Central.",
    "Which station has the most train traffic?",

    # Aggregates / percentages
    "What percentage of trains were on time overall?",
    "What is the on-time rate per delay severity category?",
    "How many trains were canceled today?",
    "List the top 5 stations by cancellation count.",

    # Time-based patterns
    "What hour of the day has the most delays?",
    "How does average delay compare between weekdays and weekends?",
    "What is the average delay per day of the week?",
    "What is the busiest hour for train departures at Liège-Guillemins?",

    # Joins / categories
    "Which train category has the most delays?",

    # Comparisons
    "Compare average delay between Bruxelles-Central and Anvers-Central.",

    # Known data limitations -- should gracefully decline, not hallucinate
    "Which platform at Bruxelles-Central had the worst average delay this morning?",
    "Which direction has the most delayed trains?",
    "Which stations have wheelchair accessible boarding?",

    # Out-of-scope 
    "What was the weather like at Bruxelles-Central?",
    "How much does a ticket to Bruges cost?",
]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        for i, q in enumerate(SMOKE_TEST_QUESTIONS):
            try:
                ask(q)
            except LLMConnectionError as e:
                print(f"⚠️  Skipped (backend issue): {e}\n")
            # Small pacing delay between questions so free-tier rate limits
            # (commonly ~20 requests/minute) aren't tripped by running the
            # whole smoke-test suite back to back.
            if i < len(SMOKE_TEST_QUESTIONS) - 1:
                time.sleep(2)