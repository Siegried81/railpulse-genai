"""
Tests for app/sql_utils.py: SQL extraction and the consultant input builder.
"""

from app.sql_utils import (
    CONSULTANT_MAX_ROWS,
    build_consultant_input,
    extract_sql,
    rows_in_minutes,
)


def test_extract_sql_strips_fences_and_trailing_text():
    raw = "```sql\nSELECT 1;\nThis query returns one.\n```"
    assert extract_sql(raw) == "SELECT 1;"


def test_extract_sql_skips_preamble_before_select():
    assert extract_sql("Here is the query:\nSELECT 2;") == "SELECT 2;"


def test_extract_sql_keeps_no_query_line():
    raw = "NO_QUERY: direction data is unavailable.\nextra"
    assert extract_sql(raw) == "NO_QUERY: direction data is unavailable."


def test_rows_in_minutes_replaces_delay_seconds():
    rows = [{"Stations Name": "Gouvy", "delay_seconds": 90}]
    assert rows_in_minutes(rows) == [{"Stations Name": "Gouvy", "delay_minutes": 1.5}]


def test_rows_in_minutes_converts_bare_aggregates_and_seconds_aliases():
    rows = [{"AVG(delay_seconds)": 150, "MAX(delay_seconds)": 600, "total_delay_seconds": 3600}]
    assert rows_in_minutes(rows) == [
        {"AVG(delay_minutes)": 2.5, "MAX(delay_minutes)": 10.0, "total_delay_minutes": 60.0}
    ]


def test_rows_in_minutes_leaves_counts_minutes_and_labels_alone():
    rows = [{"COUNT(delay_seconds)": 42, "avg_delay_minutes": 3.2, "Stations Name": "Gouvy"}]
    assert rows_in_minutes(rows) == rows


def test_rows_in_minutes_keeps_same_columns_on_every_row_with_nulls():
    # A NULL must not leave one row with delay_seconds and another with
    # delay_minutes: the table and CSV would get two half-empty columns.
    rows = [{"delay_seconds": None}, {"delay_seconds": 120}]
    assert rows_in_minutes(rows) == [{"delay_minutes": None}, {"delay_minutes": 2.0}]


def test_consultant_input_has_minutes_not_seconds_and_is_capped():
    rows = [{"delay_seconds": 60 * i} for i in range(CONSULTANT_MAX_ROWS + 5)]
    text = build_consultant_input("q", "SELECT 1", rows)
    assert "delay_seconds" not in text
    assert text.count("delay_minutes") == CONSULTANT_MAX_ROWS


# --- ungrounded_figures: cases taken from real consultant answers ---------

from app.sql_utils import ungrounded_figures  # noqa: E402


def _source(rows, question="q", sql="SELECT 1"):
    return build_consultant_input(question, sql, rows)


def test_grounded_answer_with_rounding_and_thousands_passes():
    src = _source([{"Stations Name": "Nivelles", "avg_delay_minutes": 4.4321, "sample_size": 452, "n": 2520}])
    answer = "Nivelles had about 4.4 minutes of delay over 452 trains; 2,520 records in total."
    assert ungrounded_figures(answer, src) == []


def test_clock_times_and_caveat_constants_pass():
    src = _source([{"Hour": 16, "train_count": 97}], question="busiest hour at Liege?")
    answer = "The busiest hour is 16:00 with 97 trains; this is not a confirmed 100% completion rate."
    assert ungrounded_figures(answer, src) == []


def test_derived_percentage_is_flagged():
    src = _source([{"on_time_rate_pct": 73.4}])
    assert ungrounded_figures("On time 73.4%, so about 26.6% were late.", src) == ["26.6"]


def test_derived_figures_in_words_are_flagged():
    src = _source([{"Stations Name": "A", "avg_delay_minutes": 2.91}, {"Stations Name": "B", "avg_delay_minutes": 1.86}])
    answer = "A averages 2.91 versus 1.86 at B, roughly one minute more; one in four trains is late."
    assert ungrounded_figures(answer, src) == ["one minute more", "one in four"]


def test_double_check_is_not_a_multiple():
    src = _source([{"n": 3}])
    assert ungrounded_figures("Please double-check the 3 records.", src) == []


def test_no_break_space_thousands_separator_is_grounded():
    # gpt-oss writes "21\u202f763"; it was read as "21" and "763" and flagged.
    src = _source([{"Train Category": "S", "delay_count": 21763}])
    for sep in ("\u202f", "\u00a0", ","):
        assert ungrounded_figures(f"Category S has 21{sep}763 delays.", src) == []


def test_plain_space_does_not_join_two_numbers():
    src = _source([{"n": 2}, {"n": 300}])
    assert ungrounded_figures("Between 2 and 300 trains.", src) == []
    assert ungrounded_figures("Category S has 21 763 delays.", src) == ["21", "763"]


def test_consultant_input_rounds_sql_computed_minutes():
    rows = [{"Stations Name": "Nivelles", "avg_delay_minutes": 4.426991150442478, "rate": 0.0042137, "n": 452}]
    text = build_consultant_input("q", "SELECT 1", rows)
    assert "4.43" in text and "4.4269" not in text
    assert "0.00421" in text
    assert "452" in text


def test_rounding_leaves_table_values_raw():
    rows = [{"avg_delay_minutes": 4.426991150442478}]
    assert rows_in_minutes(rows) == rows


def test_decimal_comma_is_read_as_a_decimal():
    # A question in French gets an answer in French, and French writes 4,43
    # for 4.43. Read as "4" and "43", a correct figure was flagged and a wrong
    # one ("4,12" when the data says 4.43 and 12) passed as two grounded parts.
    src = _source([{"Stations Name": "Nivelles", "avg_delay_minutes": 4.4321, "train_count": 12}])
    assert ungrounded_figures("Le retard moyen à Nivelles est de 4,43 minutes.", src) == []
    assert ungrounded_figures("Le taux est de 72,2 %.", _source([{"on_time_rate_pct": 72.2}])) == []
    assert ungrounded_figures("Le retard moyen est de 4,12 minutes.", src) == ["4,12"]


def test_three_digit_comma_group_is_still_thousands():
    src = _source([{"n": 2520}])
    assert ungrounded_figures("2,520 records.", src) == []


def test_count_of_a_subset_in_words_is_flagged():
    # From the committed weekly brief: the model counted rows sharing a date
    # itself. "one of the stations" has no second number and stays allowed.
    src = _source([{"n": 5}])
    answer = "Three of the five peak delays occurred on one day."
    assert ungrounded_figures(answer, src) == ["Three of the five"]
    assert ungrounded_figures("Nivelles is one of the stations to watch.", src) == []


def test_upper_case_seconds_column_is_renamed_to_minutes():
    # The match is case-insensitive, so the rename must be too: the value was
    # converted but the name stayed "...SECONDS", a 5-minute delay read as 5 s.
    rows = rows_in_minutes([{"AVG(DELAY_SECONDS)": 300, "Max_Delay_Seconds": 600}])
    assert rows == [{"AVG(DELAY_minutes)": 5.0, "Max_Delay_minutes": 10.0}]
