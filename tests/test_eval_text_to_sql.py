"""
Tests for scripts/eval_text_to_sql.py: the result comparator, the golden set's
shape, and the scoring of one case. The comparator is tested on wrong answers
above all: an accuracy figure is only worth something if a wrong result fails.
Every query runs on a tiny tmp_path database; no LLM is called.
"""

import re
import sqlite3

import pytest

from app import config, llm_client
from app.guardrails import validate_query
from app.prompts import MIN_STATION_SAMPLE, TEXT_TO_SQL_SYSTEM_PROMPT
from eval_text_to_sql import (
    evaluate_case,
    load_golden,
    main,
    render_report,
    results_match,
    summarise,
)


# --- results_match ---------------------------------------------------------------


def test_extra_columns_and_other_aliases_are_accepted():
    ref = [{"s": "Namur", "avg": 2.5}]
    cand = [{"station": "Namur", "avg_delay_minutes": 2.5, "sample_size": 562}]
    assert results_match(ref, cand, order_matters=False) == (True, "")


def test_a_rounded_average_still_matches():
    assert results_match([{"a": 2.4358}], [{"b": 2.4}], order_matters=False)[0]


def test_a_different_figure_fails():
    ok, reason = results_match([{"a": 2.4358}], [{"b": 2.5}], order_matters=False)
    assert not ok and "no column matches" in reason


def test_seconds_and_minutes_are_compared_as_the_user_sees_them():
    # The UI converts *_seconds columns to minutes; so does the comparison.
    ref = [{"max_minutes": 96.0}]
    assert results_match(ref, [{"max_delay_seconds": 5760}], order_matters=False)[0]
    assert not results_match(ref, [{"max_delay": 5760}], order_matters=False)[0]


def test_a_missing_or_extra_row_fails():
    ref = [{"h": 1}, {"h": 2}]
    assert results_match(ref, [{"h": 1}], order_matters=False) == (False, "1 rows, expected 2")
    assert not results_match(ref, [{"h": 1}, {"h": 2}, {"h": 3}], order_matters=False)[0]


def test_order_counts_only_when_the_question_asks_for_it():
    ref = [{"s": "A"}, {"s": "B"}]
    cand = [{"s": "B"}, {"s": "A"}]
    assert results_match(ref, cand, order_matters=False)[0]
    assert not results_match(ref, cand, order_matters=True)[0]


def test_columns_that_match_one_by_one_but_are_paired_wrongly_fail():
    # Both columns hold the right values, but each station got the other's delay.
    ref = [{"s": "A", "d": 1.0}, {"s": "B", "d": 9.0}]
    cand = [{"s": "A", "d": 9.0}, {"s": "B", "d": 1.0}]
    ok, reason = results_match(ref, cand, order_matters=False)
    assert not ok and "row by row" in reason


def test_mixed_types_and_nulls_do_not_crash_the_sort():
    ref = [{"x": None}, {"x": "a"}, {"x": 1}]
    assert results_match(ref, list(reversed(ref)), order_matters=False)[0]


# --- the golden set itself ---------------------------------------------------------


def test_golden_set_is_well_formed():
    cases = load_golden()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)) >= 30
    for case in cases:
        assert isinstance(case["in_few_shot"], bool), case["id"]
        if case.get("expected") == "NO_QUERY":
            assert "reference_sql" not in case, case["id"]
        else:
            assert validate_query(case["reference_sql"]) == (True, ""), case["id"]
            assert isinstance(case["order_matters"], bool), case["id"]
    assert sum(c.get("expected") == "NO_QUERY" for c in cases) >= 3
    assert sum(not c["in_few_shot"] for c in cases) > len(cases) / 2


def test_most_delays_means_the_highest_average_everywhere():
    """One meaning for "the most delays" in the prompt and the golden set.

    The few-shot once answered it with a count of delayed rows for the hour and
    an average for the train category; the model applied the count to the
    category and the eval failed on it. A count of delayed rows follows
    traffic, so both places must rank by the average, with the sample guard.
    """
    shots = re.findall(r"^Q: (.*)\nSQL: (.*)$", TEXT_TO_SQL_SYSTEM_PROMPT, flags=re.MULTILINE)
    golden = [(c["question"], c["reference_sql"]) for c in load_golden() if "reference_sql" in c]
    pairs = [(q, sql) for q, sql in shots + golden if re.search(r"\bmost delays\b", q, re.IGNORECASE)]
    assert len(pairs) >= 4  # two few-shot examples, two golden cases
    for question, sql in pairs:
        assert "AVG(" in sql and "delay_seconds > 0" not in sql, question
        assert re.search(rf"HAVING COUNT\(\*\) >= ({MIN_STATION_SAMPLE}\b)", sql), question


# --- scoring one case on a tiny database ------------------------------------------


@pytest.fixture
def tiny_db(tmp_path, monkeypatch):
    path = tmp_path / "tiny.db"
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE liveboard_records ("Stations Name" TEXT, delay_seconds INTEGER)')
    conn.executemany(
        "INSERT INTO liveboard_records VALUES (?, ?)",
        [("Namur", 60), ("Namur", 180), ("Arlon", 600)],
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(config, "DB_PATH", str(path))
    return path


CASE = {
    "id": "avg_namur",
    "question": "Average delay at Namur?",
    "reference_sql": "SELECT AVG(delay_seconds) / 60.0 FROM liveboard_records WHERE \"Stations Name\" = 'Namur'",
    "order_matters": False,
    "in_few_shot": False,
}
NO_QUERY_CASE = {"id": "weather", "question": "Weather?", "expected": "NO_QUERY", "in_few_shot": False}


def _says(sql):
    return lambda question: (sql, "groq · test-model")


def test_a_right_query_written_differently_passes(tiny_db):
    sql = "SELECT \"Stations Name\", AVG(delay_seconds) AS avg_delay_seconds FROM liveboard_records WHERE \"Stations Name\" = 'Namur' GROUP BY 1;"
    r = evaluate_case(CASE, _says(sql))
    assert r["status"] == "pass", r["reason"]
    assert r["by"] == "groq · test-model"


def test_a_query_on_the_wrong_station_fails(tiny_db):
    sql = "SELECT AVG(delay_seconds) / 60.0 FROM liveboard_records WHERE \"Stations Name\" = 'Arlon';"
    assert evaluate_case(CASE, _says(sql))["status"] == "fail"


def test_a_blocked_query_fails_with_the_guardrail_reason(tiny_db):
    r = evaluate_case(CASE, _says("SELECT * FROM (json_each('[1]'));"))
    assert r["status"] == "fail" and "guardrails" in r["reason"]


def test_no_query_is_right_only_where_expected(tiny_db):
    assert evaluate_case(NO_QUERY_CASE, _says("NO_QUERY: no weather data"))["status"] == "pass"
    assert evaluate_case(NO_QUERY_CASE, _says("SELECT 1;"))["status"] == "fail"
    r = evaluate_case(CASE, _says("NO_QUERY: cannot"))
    assert r["status"] == "fail" and "answerable" in r["reason"]


def test_an_llm_failure_is_not_measured_rather_than_wrong(tiny_db):
    # A first live run counted 38 quota errors as wrong SQL and printed 2/40.
    def down(question):
        raise llm_client.LLMRateLimitError("tokens per day (TPD): Limit 200000")

    r = evaluate_case(CASE, down)
    assert r["status"] == "error" and "TPD" in r["reason"]

    summary = summarise([r, evaluate_case(CASE, _says(CASE["reference_sql"]))])
    assert summary["all"] == {"passed": 1, "measured": 1, "errors": 1}


def test_summary_splits_few_shot_cases_and_report_lists_failures(tiny_db):
    results = [
        {**evaluate_case(CASE, _says(CASE["reference_sql"]))},
        {**evaluate_case({**NO_QUERY_CASE, "in_few_shot": True}, _says("SELECT 1;"))},
    ]
    summary = summarise(results)
    assert summary["all"] == {"passed": 1, "measured": 2, "errors": 0}
    assert summary["not_in_few_shot"] == {"passed": 1, "measured": 1, "errors": 0}
    assert summary["in_few_shot"] == {"passed": 0, "measured": 1, "errors": 0}

    report = render_report(results, summary, "groq", ["groq · test-model"], "2026-11-19 19:00")
    assert "| All cases | 1/2 (50%) |" in report
    assert "### weather" in report and "should have answered NO_QUERY" in report


def test_offline_run_fails_when_a_reference_query_is_broken(tiny_db, monkeypatch):
    import eval_text_to_sql

    broken = {**CASE, "id": "broken", "reference_sql": "SELECT AVG(delay_seconds) FROM no_such_table"}
    monkeypatch.setattr(eval_text_to_sql, "load_golden", lambda: [CASE, NO_QUERY_CASE])
    assert main([]) == 0
    monkeypatch.setattr(eval_text_to_sql, "load_golden", lambda: [CASE, broken])
    with pytest.raises(ValueError, match="no_such_table|Unknown table"):
        main([])


def test_live_run_scores_the_engine_sql_step_and_writes_the_report(tiny_db, monkeypatch, tmp_path):
    import eval_text_to_sql

    monkeypatch.setattr(eval_text_to_sql, "load_golden", lambda: [CASE, NO_QUERY_CASE])
    monkeypatch.setattr(eval_text_to_sql, "REPORT_PATH", tmp_path / "eval.md")
    answers = {CASE["question"]: CASE["reference_sql"] + ";", NO_QUERY_CASE["question"]: "SELECT 1;"}
    monkeypatch.setattr(eval_text_to_sql.engine, "generate_sql", lambda q: (answers[q], "groq · test-model"))

    assert main(["--live", "--provider", "groq"]) == 0
    report = (tmp_path / "eval.md").read_text(encoding="utf-8")
    assert "| All cases | 1/2 (50%) |" in report
    assert "SQL written by: groq · test-model" in report
    assert (tmp_path / "eval.json").exists()


def test_a_run_with_errors_is_reported_as_incomplete_and_exits_2(tiny_db, monkeypatch, tmp_path):
    import eval_text_to_sql

    monkeypatch.setattr(eval_text_to_sql, "load_golden", lambda: [CASE, NO_QUERY_CASE])
    monkeypatch.setattr(eval_text_to_sql, "REPORT_PATH", tmp_path / "eval.md")

    def generate(question):
        if question == CASE["question"]:
            raise llm_client.LLMRateLimitError("tokens per day (TPD)")
        return "NO_QUERY: no weather", "groq · test-model"

    monkeypatch.setattr(eval_text_to_sql.engine, "generate_sql", generate)
    assert main(["--live"]) == 2
    report = (tmp_path / "eval.md").read_text(encoding="utf-8")
    assert "Incomplete run: 1 cases got no answer" in report
    assert "| All cases | 1/1 (100%), 1 not measured |" in report
    assert "## Errors (not measured)" in report and "avg_namur: LLM error" in report


def test_a_run_where_nothing_answered_shows_no_percentage(tiny_db):
    def down(question):
        raise llm_client.LLMConnectionError("down")

    summary = summarise([evaluate_case(CASE, down)])
    report = render_report([evaluate_case(CASE, down)], summary, "groq", [], "2026-11-19 22:00")
    assert "| All cases | not measured (1 errors) |" in report
    assert "%" not in report.split("## Failures")[0].split("| Subset")[1]
