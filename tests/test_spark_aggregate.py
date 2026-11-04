"""Tests for the PySpark snapshot -> stop-event collapse in spark_aggregate.py.

These pin the same rule the reference pandas pipeline pins (one row per train
per station per date, latest poll wins, day-first timestamps read as day-first)
so the Spark version cannot drift from it.

The whole module skips cleanly when PySpark is not importable, and each Spark
test skips if a SparkSession cannot start (no Java runtime). Nothing here
installs Java or hits the network.
"""

import sys
from pathlib import Path

import pytest

# pyspark is an optional extra (requirements-spark.txt); skip the file whole if
# it is not installed.
pytest.importorskip("pyspark")

# spark_aggregate.py lives in scripts/, next to build_database.py; put scripts/
# on sys.path so `import spark_aggregate` and `import build_database` both work
# regardless of where pytest is invoked from.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import spark_aggregate  # noqa: E402
from spark_aggregate import (  # noqa: E402
    STOP_EVENT_KEY,
    SNAPSHOT_SORT_COL,
    EXPORT_TS_FORMAT,
    keep_latest_snapshot,
)


@pytest.fixture(scope="module")
def spark():
    """A local SparkSession, or a clean skip when one cannot start.

    PySpark can be installed while no Java runtime is present; creating the
    session is what fails in that case, so the guard lives here rather than at
    import time.
    """
    try:
        session = spark_aggregate.build_spark("railpulse-spark-tests")
    except Exception as exc:  # noqa: BLE001 - any JVM/Java failure means "cannot run"
        pytest.skip(f"cannot start a local SparkSession (no Java runtime?): {exc}")
    yield session
    session.stop()


def _collapse(spark, rows):
    """Build a snapshots DataFrame from dict rows and run the real collapse."""
    df = spark.createDataFrame(rows)
    return keep_latest_snapshot(df, STOP_EVENT_KEY, SNAPSHOT_SORT_COL)


def test_key_matches_the_reference_pin():
    assert STOP_EVENT_KEY == ["vehicle_id", "station_id", "Scheduled Date"]
    assert SNAPSHOT_SORT_COL == "pulled_at"


def test_constants_match_build_database():
    """Key, sort column and timestamp format must equal build_database.py's, so
    the Spark collapse and the pandas pipeline can't drift.
    """
    import build_database as bd

    ref_key, ref_sort = bd.DEDUP_KEY_BY_TABLE["liveboard_records"]
    assert STOP_EVENT_KEY == ref_key
    assert SNAPSHOT_SORT_COL == ref_sort
    # build_database uses the pandas strptime spelling; translate it to the
    # Spark/Java pattern and check the two agree.
    spark_fmt = (
        bd.EXPORT_DATETIME_FORMAT.replace("%d", "dd").replace("%m", "MM")
        .replace("%Y", "yyyy").replace("%H", "HH").replace("%M", "mm")
    )
    assert EXPORT_TS_FORMAT == spark_fmt


def test_many_snapshots_of_one_stop_collapse_to_one_event(spark):
    # Five polls of the same train/station/date -> one stop event, carrying the
    # fields of the latest poll (pulled_at 08:40, delay 300).
    rows = [
        {"vehicle_id": "T1", "station_id": "S_A", "Scheduled Date": "2026-08-05",
         "pulled_at": "05/08/2026 08:00", "delay_seconds": 0},
        {"vehicle_id": "T1", "station_id": "S_A", "Scheduled Date": "2026-08-05",
         "pulled_at": "05/08/2026 08:10", "delay_seconds": 60},
        {"vehicle_id": "T1", "station_id": "S_A", "Scheduled Date": "2026-08-05",
         "pulled_at": "05/08/2026 08:40", "delay_seconds": 300},
        {"vehicle_id": "T1", "station_id": "S_A", "Scheduled Date": "2026-08-05",
         "pulled_at": "05/08/2026 08:20", "delay_seconds": 120},
        {"vehicle_id": "T1", "station_id": "S_A", "Scheduled Date": "2026-08-05",
         "pulled_at": "05/08/2026 08:30", "delay_seconds": 180},
    ]
    out = _collapse(spark, rows).collect()
    assert len(out) == 1
    assert out[0]["delay_seconds"] == 300


def test_same_train_at_two_stations_keeps_both_stops(spark):
    # station_id is in the key, so one train at two stations stays two events.
    rows = [
        {"vehicle_id": "T1", "station_id": "S_A", "Scheduled Date": "2026-08-05",
         "pulled_at": "05/08/2026 08:00", "delay_seconds": 60},
        {"vehicle_id": "T1", "station_id": "S_B", "Scheduled Date": "2026-08-05",
         "pulled_at": "05/08/2026 08:30", "delay_seconds": 120},
    ]
    out = _collapse(spark, rows).collect()
    assert sorted(r["station_id"] for r in out) == ["S_A", "S_B"]


def test_pulled_at_is_parsed_day_first(spark):
    # "01/08/2026 00:10" is later than "31/07/2026 23:50" only under a day-first
    # read; a month-first guess would null out "31/07/2026" and keep the wrong row.
    rows = [
        {"vehicle_id": "T1", "station_id": "S_A", "Scheduled Date": "2026-08-01",
         "pulled_at": "01/08/2026 00:10", "delay_seconds": 120},
        {"vehicle_id": "T1", "station_id": "S_A", "Scheduled Date": "2026-08-01",
         "pulled_at": "31/07/2026 23:50", "delay_seconds": 60},
    ]
    out = _collapse(spark, rows).collect()
    assert len(out) == 1
    assert out[0]["delay_seconds"] == 120


def test_unparseable_pulled_at_never_wins(spark):
    # A garbage pulled_at parses to null and sorts last, so the good poll wins
    # even though the garbage row comes later in the input. This is where the
    # Spark collapse differs from the pandas one (NaT sorts last there and
    # keep="last" takes it); the docstring of keep_latest_snapshot says so.
    rows = [
        {"vehicle_id": "T1", "station_id": "S_A", "Scheduled Date": "2026-08-05",
         "pulled_at": "05/08/2026 08:00", "delay_seconds": 60},
        {"vehicle_id": "T1", "station_id": "S_A", "Scheduled Date": "2026-08-05",
         "pulled_at": "not a timestamp", "delay_seconds": 999},
    ]
    out = _collapse(spark, rows).collect()
    assert len(out) == 1
    assert out[0]["pulled_at"] == "05/08/2026 08:00"
    assert out[0]["delay_seconds"] == 60


def test_output_schema_equals_input_schema(spark):
    # The collapse drops rows, never columns: one row per stop event with the
    # same fields as the snapshots.
    rows = [
        {"vehicle_id": "T1", "station_id": "S_A", "Scheduled Date": "2026-08-05",
         "pulled_at": "05/08/2026 08:00", "delay_seconds": 0, "platform": "3"},
    ]
    df = spark.createDataFrame(rows)
    out = keep_latest_snapshot(df, STOP_EVENT_KEY, SNAPSHOT_SORT_COL)
    assert out.columns == df.columns
