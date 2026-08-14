"""
Quick CLI test harness for the RailPulse AI pipeline.

Usage:
    python test_pipeline.py "Which station had the worst average delay this week?"
    python test_pipeline.py             (runs the full smoke-test suite below)
"""

import sys
import time

from app.llm_client import (
    LLMConnectionError,
    auto_fix_misdirected_recommendation,
    auto_fix_superlative_claims,
    call_llm,
    call_llm_until,
    correct_consultant_generalization,
    correct_consultant_numbers,
    correct_no_invented_entities,
    force_no_query_if_hallucinated_alias_columns,
    force_no_query_if_on_time_wrong_definition,
    force_no_query_if_raw_category_delay_count_ranking,
    force_no_query_if_self_referential_case,
    force_no_query_if_station_ranking_unreliable,
    force_no_query_if_unreliable_wheelchair,
    force_reason_mismatch_worst_best_station,
    replace_vague_period_with_dates,
    strip_invented_cause,
    strip_recommendation_after_unavailability_disclaimer,
    strip_ungrounded_time_reference,
    validate_consultant_numbers,
    validate_no_false_generalization,
    validate_no_hallucinated_alias_columns,  
    validate_no_invented_entities,
    validate_no_query_reason_matches_question,
    validate_no_raw_category_delay_count_ranking,
    validate_no_self_referential_case_filter,
    validate_no_unreliable_wheelchair_query,
    validate_on_time_uses_delay_severity,
    validate_station_query_matches_question_stations,
    validate_station_ranking_has_min_sample,
    validate_superlative_sql_direction,  
)

from app.db import execute_query
from app.prompts import TEXT_TO_SQL_SYSTEM_PROMPT, CONSULTANT_SYSTEM_PROMPT
from app.sql_utils import extract_sql

# Small pause between questions when running the full suite, to stay under
# a free-tier provider's RPM cap (each question = 2 calls: SQL gen + consultant).
PACING_SECONDS = 4

# Small pause between the SQL-gen call and the consultant call WITHIN one
# question -- these two calls fire back-to-back with no gap otherwise, which
# can be enough on its own to trip a per-minute rate limit on a free-tier
# model, independent of the pacing between different questions above.
INTRA_QUESTION_PAUSE_SECONDS = 1.5


def ask(question: str) -> None:
    print(f"\n🗣️  Question: {question}\n")

    raw_sql = call_llm_until(
        TEXT_TO_SQL_SYSTEM_PROMPT,
        question,
        validate=lambda text: validate_superlative_sql_direction(question, text)
        and validate_no_self_referential_case_filter(text)
        and validate_no_unreliable_wheelchair_query(text)
        and validate_no_raw_category_delay_count_ranking(text)
        and validate_on_time_uses_delay_severity(question, text)
        and validate_station_ranking_has_min_sample(text)
        and validate_no_query_reason_matches_question(question, text)
        and validate_station_query_matches_question_stations(question, text),
    )
    sql = extract_sql(raw_sql)

    # Deterministic safety net: even after the retry loop above, a weak
    # model can still fail these checks on every attempt (call_llm_until
    # gives up after max_attempts and returns the last invalid result
    # rather than raising). Force a safe NO_QUERY fallback rather than let
    # a tautological or unreliable-column query reach execution.
    sql = force_no_query_if_unreliable_wheelchair(sql)
    sql = force_no_query_if_self_referential_case(sql)
    sql = force_no_query_if_raw_category_delay_count_ranking(sql)
    sql = force_no_query_if_on_time_wrong_definition(question, sql)
    sql = force_no_query_if_station_ranking_unreliable(sql)
    sql = force_reason_mismatch_worst_best_station(question, sql)

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

    time.sleep(INTRA_QUESTION_PAUSE_SECONDS)

    consultant_input = (
        f"Question: {question}\n"
        f"SQL executed: {sql}\n"
        f"Results: {rows[:20]}"
    )
    answer = call_llm_until(
        CONSULTANT_SYSTEM_PROMPT,
        consultant_input,
        validate=lambda text: validate_consultant_numbers(text, rows)
        and validate_no_false_generalization(text, rows)
        and validate_no_invented_entities(text, rows),
    )
    answer = correct_consultant_numbers(answer, rows)
    answer = correct_consultant_generalization(answer, rows)
    answer = correct_no_invented_entities(answer, rows)
    answer = auto_fix_superlative_claims(answer, rows)
    answer = auto_fix_misdirected_recommendation(answer, rows)
    print(f"💬 RailPulse Consultant says:\n{answer}\n")


SMOKE_TEST_QUESTIONS = [
    # Trimmed from the original 18 down to 8 -- 18 questions x 2 calls each
    # (SQL gen + consultant), plus retries, reliably burns through Groq's
    # ~50 req/day free-tier quota before the run even finishes. This
    # smaller set still covers every category below, and specifically
    # keeps the two questions that exercise the most recently added
    # guards (weekday/weekend recommendation direction, train category
    # delay-count vs. rate).

    # Core delay / ranking
    "Which station had the worst average delay this week?",

    # Aggregates / percentages
    "What percentage of trains were on time overall?",

    # Time-based patterns -- exercises auto_fix_misdirected_recommendation
    "How does average delay compare between weekdays and weekends?",

    # Joins / categories -- exercises validate_no_raw_category_delay_count_ranking
    "Which train category has the most delays?",

    # Comparisons
    "Compare average delay between Bruxelles-Central and Anvers-Central.",

    # Known data limitations -- should gracefully decline, not hallucinate
    "Which platform at Bruxelles-Central had the worst average delay this morning?",
    "Which stations have wheelchair accessible boarding?",

    # Out-of-scope
    "What was the weather like at Bruxelles-Central?",
]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        for i, q in enumerate(SMOKE_TEST_QUESTIONS):
            ask(q)
            if i < len(SMOKE_TEST_QUESTIONS) - 1:
                time.sleep(PACING_SECONDS)