"""
Tests for the dedup rule in scripts/build_database.py.

The dedup key defines what one row of liveboard_records means (one stop
event), so these tests pin it: one row per train per station per date, the
latest poll wins, and day-first export timestamps are read as day-first.
"""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from app.guardrails import KNOWN_TABLES
from build_database import DEDUP_KEY_BY_TABLE, TABLES, keep_latest_snapshot


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


def test_tables_built_are_exactly_the_guardrails_allow_list():
    """build_database derives its tables from KNOWN_TABLES, and the committed
    database must hold those tables and nothing else, or the allow-list
    would block a real table (or allow a missing one)."""
    assert tuple(TABLES) == KNOWN_TABLES

    db_path = Path(__file__).resolve().parent.parent / "data" / "railpulse_ai.db"
    if not db_path.exists():
        pytest.skip("data/railpulse_ai.db not present")
    conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
    try:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )}
    finally:
        conn.close()
    assert names == set(KNOWN_TABLES)


def test_planned_time_is_rebuilt_from_the_drifting_export_time():
    """08:05 on time and 08:06 with 60 s are one stop planned at 08:05."""
    from build_database import restore_planned_time

    df = pd.DataFrame({
        "scheduled_time": ["04/08/2026 08:05", "04/08/2026 08:06"],
        "delay_seconds": [0, 60],
        "Hour": [8, 8],
        "day_of_week": ["Tuesday", "Tuesday"],
        "day_number": [2, 2],
    })
    out = restore_planned_time(df)
    assert out["scheduled_time"].tolist() == ["04/08/2026 08:05", "04/08/2026 08:05"]


def test_a_late_train_stays_in_its_planned_hour():
    from build_database import restore_planned_time

    df = pd.DataFrame({
        "scheduled_time": ["03/08/2026 00:04"],   # planned 23:58 on Sunday, 6 min late
        "delay_seconds": [360],
        "Hour": [0],
        "day_of_week": ["Monday"],
        "day_number": [1],
    })
    row = restore_planned_time(df).iloc[0]
    assert row["Hour"] == 23
    assert row["day_of_week"] == "Sunday"
    assert row["day_number"] == 7


LIVEBOARD_HEADER = (
    "record_id;station_id;vehicle_id;platform;scheduled_time;delay_seconds;canceled;pulled_at;"
    "stations.station_id;Stations Name;stations.standard_name;stations.latitude;stations.longitude;"
    "stations.wheelchair_boarding;stations.location_type;vehicles.vehicle_id;vehicles.vehicle_type;"
    "vehicles.direction;Hour;day_of_week;day_number;Delay Severity;Scheduled Date"
)


def _stop(record_id, vehicle, delay, severity, pulled="07/08/2026 04:15", station="gs:1"):
    """One liveboard_records CSV line in the export's format (cp1252, ";", decimal comma)."""
    return (
        f"{record_id};{station};{vehicle};;07/08/2026 04:54;{delay};False;{pulled};{station};"
        f"Arlon;Arlon;49,6833;5,8167;;0;{vehicle};Train;;4;Friday;5;{severity};07/08/2026 00:00"
    )


@pytest.fixture
def tiny_csv_dir(tmp_path):
    """The five CSV exports with their real headers and a handful of rows.

    Every table is present because a build now fails its checks when one is
    missing, instead of silently skipping it.
    """
    files = {
        "liveboard_records": [
            LIVEBOARD_HEADER,
            _stop(1, "T1", 0, "On Time (<2min)"),
            _stop(2, "T2", 900, "Moderate (5-15min)"),
            _stop(3, "T3", 960, "Severe (>15min)"),
        ],
        "stations": [
            "station_id;name;stations names;latitude;longitude;wheelchair_boarding;location_type",
            "gs:1;Herzogenrath;Herzogenrath;50,8709;6,0944;;0",
            "gs:2;Arlon;Arlon;49,6833;5,8167;;0",
        ],
        "routes": [
            "agency_id;route_color;route_desc;route_id;route_long_name;route_short_name;"
            "route_text_color;route_type;route_url;train_category",
            "a;;;R1;;;;2;;IC",
        ],
        "trips": ["trip_id;route_id", "T1;R1"],
        "vehicles": ["vehicle_id;vehicle_type;direction", "T1;Train;", "T2;Train;", "T3;Train;"],
    }
    for name, lines in files.items():
        (tmp_path / f"{name}.csv").write_text("\n".join(lines) + "\n", encoding="cp1252")
    return tmp_path


def test_coordinates_with_a_decimal_comma_are_stored_as_numbers(tiny_csv_dir):
    """The export writes 50,8709 for 50.8709. Read as text, "stations.latitude"
    averaged to its integer part and compared as a string, so
    `latitude > 50.85` counted 27,930 stop events instead of 46,727."""
    import build_database

    build_database.build_database(tiny_csv_dir / "out.db", tiny_csv_dir)

    conn = sqlite3.connect(tiny_csv_dir / "out.db")
    try:
        rows = conn.execute(
            "SELECT typeof(latitude), latitude, longitude FROM stations ORDER BY station_id"
        ).fetchall()
        north = conn.execute("SELECT COUNT(*) FROM stations WHERE latitude > 50.85").fetchone()[0]
    finally:
        conn.close()
    assert rows == [("real", 50.8709, 6.0944), ("real", 49.6833, 5.8167)]
    assert north == 1


def test_a_full_build_passes_its_checks_and_leaves_no_temp_file(tiny_csv_dir):
    import build_database

    out = tiny_csv_dir / "out.db"
    build_database.build_database(out, tiny_csv_dir)

    conn = sqlite3.connect(out)
    try:
        assert build_database.check_database(conn) == []
        assert conn.execute("SELECT COUNT(*) FROM liveboard_records").fetchone()[0] == 3
    finally:
        conn.close()
    assert not (tiny_csv_dir / "out.db.tmp").exists()


def test_a_failed_check_leaves_the_previous_database_untouched(tiny_csv_dir):
    import build_database

    out = tiny_csv_dir / "out.db"
    out.write_bytes(b"previous database")
    lines = (tiny_csv_dir / "liveboard_records.csv").read_text(encoding="cp1252").splitlines()
    # 60 s labelled Severe: the export's banding changed, the build must stop.
    lines.append(_stop(4, "T4", 60, "Severe (>15min)"))
    (tiny_csv_dir / "liveboard_records.csv").write_text("\n".join(lines) + "\n", encoding="cp1252")

    with pytest.raises(SystemExit, match="labelled 'Severe"):
        build_database.build_database(out, tiny_csv_dir)
    assert out.read_bytes() == b"previous database"


def test_a_missing_csv_fails_the_build_instead_of_skipping_the_table(tiny_csv_dir):
    import build_database

    (tiny_csv_dir / "trips.csv").unlink()
    with pytest.raises(SystemExit, match="missing tables"):
        build_database.build_database(tiny_csv_dir / "out.db", tiny_csv_dir)
    assert not (tiny_csv_dir / "out.db").exists()


@pytest.mark.parametrize(
    "corruption, expected",
    [
        ('UPDATE liveboard_records SET "Delay Severity" = \'On Time\' WHERE delay_seconds = 0', "unknown Delay Severity"),
        ("UPDATE liveboard_records SET delay_seconds = NULL WHERE delay_seconds = 0", 'NULL "delay_seconds"'),
        ("INSERT INTO liveboard_records SELECT * FROM liveboard_records WHERE delay_seconds = 0", "more than once"),
        ('UPDATE liveboard_records SET "Scheduled Date" = \'07/08/2026\'', "not YYYY-MM-DD"),
        ("ALTER TABLE stations ADD COLUMN weather TEXT", "only in DB ['weather']"),
    ],
)
def test_check_database_reports_each_broken_invariant(tiny_csv_dir, corruption, expected):
    import build_database

    out = tiny_csv_dir / "out.db"
    build_database.build_database(out, tiny_csv_dir)
    conn = sqlite3.connect(out)
    try:
        conn.execute(corruption)
        problems = build_database.check_database(conn)
    finally:
        conn.close()
    assert any(expected in p for p in problems), problems


def test_a_tie_on_pulled_at_keeps_the_later_input_row():
    """Polls of one stop event at the same minute: the later one in the input
    wins, as in the Spark mirror. With the default quicksort, on this mix of
    keys and tied minutes, about a third of the groups kept another row."""
    import numpy as np

    rng = np.random.default_rng(0)
    n = 5000
    df = pd.DataFrame({
        "vehicle_id": rng.integers(0, 800, n).astype(str),
        "station_id": "S",
        "Scheduled Date": "2026-08-07",
        "pulled_at": [f"07/08/2026 04:1{m}" for m in rng.integers(0, 5, n)],
        "delay_seconds": np.arange(n),  # the input position
    })
    latest = df.groupby("vehicle_id")["pulled_at"].transform("max")
    expected = df[df["pulled_at"] == latest].groupby("vehicle_id")["delay_seconds"].max()

    kept = _dedup(df).set_index("vehicle_id")["delay_seconds"]
    assert kept.sort_index().equals(expected.sort_index())
