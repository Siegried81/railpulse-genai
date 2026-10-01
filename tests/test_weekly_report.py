"""
Tests for the figures scripts/generate_weekly_report.py feeds the LLM.

Runs gather_weekly_data() against a tiny throwaway database; the LLM is
never called.
"""

import sqlite3

import pytest

from app import config
from app.prompts import MIN_STATION_SAMPLE, TEXT_TO_SQL_SYSTEM_PROMPT
from generate_weekly_report import build_report_input, gather_weekly_data

COLUMNS = '("Stations Name", vehicle_id, delay_seconds, "Scheduled Date", scheduled_time, "Delay Severity", canceled)'


@pytest.fixture
def report_db(tmp_path, monkeypatch):
    rows = [
        # Train T1 on 2026-08-05: late at three consecutive stops, peak 600 s
        # reached first at Bordet then held at Vilvorde.
        ("Schaerbeek", "T1", 300, "2026-08-05", "05/08/2026 08:00", "Moderate (5-15min)", 0),
        ("Bordet", "T1", 600, "2026-08-05", "05/08/2026 08:10", "Moderate (5-15min)", 0),
        ("Vilvorde", "T1", 600, "2026-08-05", "05/08/2026 08:20", "Moderate (5-15min)", 0),
        # Same vehicle_id on another day is a different run.
        ("Bordet", "T1", 500, "2026-08-06", "06/08/2026 08:10", "Moderate (5-15min)", 0),
        ("Gouvy", "T2", 400, "2026-08-06", "06/08/2026 09:00", "Moderate (5-15min)", 0),
        # Outside the 7-day window: must be ignored.
        ("Gouvy", "T3", 9000, "2026-07-27", "27/07/2026 09:00", "Severe (>15min)", 0),
    ]
    # Enough on-time stops at one station to clear MIN_STATION_SAMPLE.
    rows += [
        ("Anvers-Central", f"B{i}", 0, "2026-08-07", "07/08/2026 10:00", "On Time (<2min)", 0)
        for i in range(MIN_STATION_SAMPLE)
    ]
    path = tmp_path / "report.db"
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE liveboard_records {COLUMNS}")
    conn.executemany("INSERT INTO liveboard_records VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    monkeypatch.setattr(config, "DB_PATH", str(path))


def test_top_anomalies_has_one_row_per_train_run(report_db):
    top = gather_weekly_data()["top_anomalies"]
    runs = [(r["vehicle_id"], r["Scheduled Date"]) for r in top]
    assert len(top) == 5
    assert len(set(runs)) == 5
    # The on-time Anvers-Central runs fill the last two slots with 0 min.
    assert runs[:3] == [("T1", "2026-08-05"), ("T1", "2026-08-06"), ("T2", "2026-08-06")]


def test_top_anomaly_is_shown_where_the_peak_was_first_reached(report_db):
    top = gather_weekly_data()["top_anomalies"]
    assert top[0]["Stations Name"] == "Bordet"
    assert top[0]["delay_minutes"] == 10.0


def test_worst_station_respects_min_sample(report_db):
    # Only Anvers-Central has MIN_STATION_SAMPLE stops; the very late but
    # tiny-sample stations must not be ranked.
    ws = gather_weekly_data()["worst_station"]
    assert ws["Stations Name"] == "Anvers-Central"
    assert ws["sample_size"] == MIN_STATION_SAMPLE


def test_report_input_describes_rows_as_train_runs(report_db):
    text = build_report_input(gather_weekly_data())
    assert "most delayed train runs" in text
    assert "Bordet" in text


def test_text_to_sql_prompt_uses_the_shared_threshold():
    assert f"HAVING COUNT(*) >= {MIN_STATION_SAMPLE}" in TEXT_TO_SQL_SYSTEM_PROMPT
    assert "HAVING COUNT(*) >= 30" not in TEXT_TO_SQL_SYSTEM_PROMPT


def test_this_morning_few_shot_is_anchored_on_the_latest_date():
    from app.prompts import TEXT_TO_SQL_SYSTEM_PROMPT as p

    few_shot = p.split("this morning?")[1].split("\n")[1]
    assert 'MAX("Scheduled Date")' in few_shot
    assert '"Hour" BETWEEN 6 AND 11' in few_shot
