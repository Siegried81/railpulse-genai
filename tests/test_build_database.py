"""
Tests for the dedup rule in scripts/build_database.py.

The dedup key defines what one row of liveboard_records means (one stop
event), so these tests pin it: one row per train per station per date, the
latest poll wins, and day-first export timestamps are read as day-first.
"""

import pandas as pd

from build_database import DEDUP_KEY_BY_TABLE, keep_latest_snapshot


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    group_cols, sort_col = DEDUP_KEY_BY_TABLE["liveboard_records"]
    return keep_latest_snapshot(df, group_cols, sort_col)


def test_dedup_key_is_one_stop_event():
    group_cols, sort_col = DEDUP_KEY_BY_TABLE["liveboard_records"]
    assert group_cols == ["vehicle_id", "station_id", "Scheduled Date"]
    assert sort_col == "pulled_at"


def test_same_train_at_two_stations_keeps_both_stops():
    # Pins the withdrawn (vehicle_id, Scheduled Date) key: it kept only one
    # station per train per day and dropped most real stop events.
    df = pd.DataFrame(
        {
            "vehicle_id": ["T1", "T1"],
            "station_id": ["S_A", "S_B"],
            "Scheduled Date": ["2026-08-05", "2026-08-05"],
            "pulled_at": ["05/08/2026 08:00", "05/08/2026 08:30"],
            "delay_seconds": [60, 120],
        }
    )
    out = _dedup(df)
    assert sorted(out["station_id"]) == ["S_A", "S_B"]


def test_repeated_polls_of_one_stop_keep_latest():
    df = pd.DataFrame(
        {
            "vehicle_id": ["T1"] * 3,
            "station_id": ["S_A"] * 3,
            "Scheduled Date": ["2026-08-05"] * 3,
            "pulled_at": ["05/08/2026 08:20", "05/08/2026 08:00", "05/08/2026 08:10"],
            "delay_seconds": [300, 0, 120],
        }
    )
    out = _dedup(df)
    assert len(out) == 1
    assert out.iloc[0]["delay_seconds"] == 300


def test_pulled_at_is_parsed_day_first():
    # First value "01/08/2026" is ambiguous; a guessed month-first format
    # would turn "31/07/2026" into NaT and keep it as the "latest" row.
    df = pd.DataFrame(
        {
            "vehicle_id": ["T1", "T1"],
            "station_id": ["S_A", "S_A"],
            "Scheduled Date": ["2026-08-01", "2026-08-01"],
            "pulled_at": ["01/08/2026 00:10", "31/07/2026 23:50"],
            "delay_seconds": [120, 60],
        }
    )
    out = _dedup(df)
    assert len(out) == 1
    assert out.iloc[0]["delay_seconds"] == 120
