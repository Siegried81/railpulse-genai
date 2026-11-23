"""
Build railpulse_ai.db (SQLite) from CSV files exported out of Power BI.

Usage:
    python scripts/build_database.py

Expects CSVs in data/ and writes data/railpulse_ai.db. The new database is
built in a temporary file and checked (check_database) before it replaces the
old one, so a failed or inconsistent build leaves the previous database in
place instead of deleting it first.
"""

import os
import sqlite3
import sys
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# `python scripts/build_database.py` puts only scripts/ on sys.path; add the
# repo root so the table list can come from app.guardrails.
sys.path.insert(0, str(BASE_DIR))

from app.guardrails import KNOWN_COLUMNS, KNOWN_TABLES  # noqa: E402

RAW_CSV_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "data" / "railpulse_ai.db"

# One table per CSV export, named after the file. The names come from
# KNOWN_TABLES so the database and the guardrails' table allow-list can
# never drift apart.
TABLES = {name: f"{name}.csv" for name in KNOWN_TABLES}

# Date columns that need conversion from French DD/MM/YYYY text to
# proper ISO 8601 (YYYY-MM-DD), so SQLite's MIN/MAX/comparisons and date
# functions work correctly instead of comparing strings alphabetically.
DATE_COLUMNS_BY_TABLE = {
    "liveboard_records": ["Scheduled Date"],
}

# liveboard_records is built from continuous GTFS-Realtime polling of station
# liveboards: the same still-upcoming stop gets re-captured on every polling
# cycle until the train leaves that station, so raw rows massively
# over-represent stops that stay "upcoming" longer (980,868 raw rows for
# 99,014 distinct stop events, ~10 captures each). Left as-is, every aggregate
# (AVG delay, on-time %, station rankings, etc.) is skewed toward whichever
# stops happened to get polled most, not toward real-world frequency.
#
# On top of that, the export stores most polls twice: for a given
# (stop event, pulled_at) the usual case is two rows identical in every field
# but record_id (439,681 of the duplicated polls, 439,654 byte-identical; only
# 27 carry a different scheduled_time/delay_seconds). This is why raw-row
# counts per stop event cluster on even numbers. The dedup below collapses both
# effects at once, so neither biases the kept figures.
#
# The unit of measurement is one STOP EVENT: one train at one station on one
# date. The key therefore includes station_id. Keying on (vehicle_id,
# Scheduled Date) alone would keep a single station per train per day (the
# last one polled) and silently drop ~87% of real stop events, so every
# station-level figure would only see the station where each train happened
# to be polled last.
#
# Within a stop event, keep the LATEST snapshot by pulled_at: the final poll
# before the train leaves the station is the most complete delay reading.
DEDUP_KEY_BY_TABLE = {
    "liveboard_records": (["vehicle_id", "station_id", "Scheduled Date"], "pulled_at"),
}


# Timestamp format of the Power BI export (French locale), e.g. "07/08/2026 04:15"
# is 7 August, not 8 July.
EXPORT_DATETIME_FORMAT = "%d/%m/%Y %H:%M"

# "Delay Severity" comes ready-made from the Power BI export; it is not derived
# here. Each label and the delay_seconds range it must cover, so a change in
# the export's banding fails the build instead of passing silently. Measured on
# the shipped data: 900 s (15 min exactly) is "Moderate", hence the bound of
# 901 there and for "Severe".
SEVERITY_BANDS = {
    "On Time (<2min)": (None, 120),     # delay_seconds < 120
    "Minor (2-5min)": (120, 300),       # 120 <= delay_seconds < 300
    "Moderate (5-15min)": (300, 901),   # 300 <= delay_seconds <= 900
    "Severe (>15min)": (901, None),     # delay_seconds > 900
}

# No index is created. Measured on the shipped data: every few-shot query ran
# in 0.54 s or less without one (SQLite builds automatic indexes for the
# train_category JOIN), well inside db.MAX_QUERY_SECONDS. Indexes on trips and
# routes cut that JOIN to 0.25 s but added 9 MB to a database that is
# committed to git and copied into the Docker image.


def restore_planned_time(df: pd.DataFrame) -> pd.DataFrame:
    """Rewrite scheduled_time as the timetable time and re-derive Hour from it.

    In the export, scheduled_time moves with the delay: the same stop reads
    08:05 with 0 s of delay, then 08:06 with 60 s. The timetable time is
    therefore scheduled_time - delay_seconds, and Hour, day_of_week and
    day_number are cut on it. Left on the drifting value, a late train slides
    into the next hour and the delay decides which hour it is counted in.
    """
    shown = pd.to_datetime(df["scheduled_time"], format=EXPORT_DATETIME_FORMAT, errors="coerce")
    planned = shown - pd.to_timedelta(df["delay_seconds"].fillna(0), unit="s")
    out = df.copy()
    out["scheduled_time"] = planned.dt.strftime(EXPORT_DATETIME_FORMAT).where(planned.notna(), df["scheduled_time"])
    if "Hour" in out.columns:
        out["Hour"] = planned.dt.hour.where(planned.notna(), df["Hour"])
    if "day_of_week" in out.columns:
        out["day_of_week"] = planned.dt.day_name().where(planned.notna(), df["day_of_week"])
    if "day_number" in out.columns:
        out["day_number"] = (planned.dt.dayofweek + 1).where(planned.notna(), df["day_number"])
    return out


def keep_latest_snapshot(df: pd.DataFrame, group_cols: list[str], sort_col: str) -> pd.DataFrame:
    """Keep one row per group_cols: the one with the latest sort_col timestamp.

    Separate from build_database() so the dedup rule (which defines what one
    row of liveboard_records means) can be tested on a small frame without
    reading the real CSVs.

    sort_col is parsed with the export's explicit day-first format. Without
    it, pandas guesses the format from the first value: an ambiguous one like
    "01/08/2026" is read month-first, every day above 12 then fails to parse,
    and those NaT rows sort last -- so they would be kept as the "latest"
    snapshot whatever their real time.

    The sort is stable, so when two polls of one stop event share the same
    timestamp the one that comes later in the input wins. The default
    quicksort left that choice to the sort algorithm; the Spark mirror
    (scripts/spark_aggregate.py) uses input order, so both now keep the same
    row on a tie.
    """
    sort_dt = pd.to_datetime(df[sort_col], format=EXPORT_DATETIME_FORMAT, errors="coerce")
    return (
        df.assign(_sort_dt=sort_dt)
        .sort_values("_sort_dt", kind="stable")
        .drop_duplicates(subset=group_cols, keep="last")
        .drop(columns="_sort_dt")
    )


def check_database(conn: sqlite3.Connection) -> list[str]:
    """Return every broken invariant of a freshly built database ([] when sound).

    These are the facts the app's prompts, guardrails and figures rely on:
    one row per stop event, no missing key or delay, the four exact
    "Delay Severity" labels each covering its own delay range, ISO dates, and
    a column list equal to guardrails.KNOWN_COLUMNS, so a renamed or new CSV
    column cannot drift away from the whitelist unnoticed.
    """
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    missing = set(KNOWN_TABLES) - tables
    if missing:
        return [f"missing tables: {sorted(missing)}"]

    problems = []
    columns = {r[1].lower() for t in KNOWN_TABLES for r in conn.execute(f'PRAGMA table_info("{t}")')}
    if columns != KNOWN_COLUMNS:
        problems.append(
            "columns differ from guardrails.KNOWN_COLUMNS: only in DB "
            f"{sorted(columns - KNOWN_COLUMNS)}, only in whitelist {sorted(KNOWN_COLUMNS - columns)}"
        )

    key_cols, _ = DEDUP_KEY_BY_TABLE["liveboard_records"]
    key_sql = ", ".join(f'"{c}"' for c in key_cols)
    (dupes,) = conn.execute(
        f"SELECT COUNT(*) FROM (SELECT 1 FROM liveboard_records GROUP BY {key_sql} HAVING COUNT(*) > 1)"
    ).fetchone()
    if dupes:
        problems.append(f"{dupes} stop events appear more than once on {key_cols}")

    for col in [*key_cols, "delay_seconds", "Delay Severity"]:
        (nulls,) = conn.execute(f'SELECT COUNT(*) FROM liveboard_records WHERE "{col}" IS NULL').fetchone()
        if nulls:
            problems.append(f'{nulls} rows with a NULL "{col}"')

    (bad_dates,) = conn.execute(
        'SELECT COUNT(*) FROM liveboard_records WHERE "Scheduled Date" '
        "NOT GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'"
    ).fetchone()
    if bad_dates:
        problems.append(f'{bad_dates} rows whose "Scheduled Date" is not YYYY-MM-DD')

    labels = {r[0] for r in conn.execute('SELECT DISTINCT "Delay Severity" FROM liveboard_records')}
    unknown = labels - set(SEVERITY_BANDS) - {None}
    if unknown:
        problems.append(f"unknown Delay Severity labels: {sorted(unknown)}")
    for label, (low, high) in SEVERITY_BANDS.items():
        outside = []
        if low is not None:
            outside.append(f"delay_seconds < {low}")
        if high is not None:
            outside.append(f"delay_seconds >= {high}")
        (n,) = conn.execute(
            f'SELECT COUNT(*) FROM liveboard_records WHERE "Delay Severity" = ? AND ({" OR ".join(outside)})',
            (label,),
        ).fetchone()
        if n:
            problems.append(f"{n} rows labelled '{label}' have a delay outside its range")
    return problems


def build_database(db_path: Path | None = None, csv_dir: Path | None = None) -> None:
    """Build the database from the CSV exports, check it, then swap it in.

    The paths default to the module's DB_PATH and RAW_CSV_DIR, read at call
    time rather than bound as default values, so a test that points them at
    tmp_path can never fall through to the real data/ folder.
    """
    db_path = Path(db_path or DB_PATH)
    csv_dir = Path(csv_dir or RAW_CSV_DIR)
    tmp_path = db_path.with_name(db_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    conn = sqlite3.connect(tmp_path)

    for table_name, csv_filename in TABLES.items():
        csv_path = csv_dir / csv_filename

        if not csv_path.exists():
            print(f"⚠️  Skipping '{table_name}': file not found at {csv_path}")
            continue

        # The French-locale export writes decimals with a comma ("50,8709").
        # Without decimal=",", latitude and longitude load as text: AVG()
        # keeps only the integer part and "> 50.85" compares strings. Only
        # those four coordinate columns change type; every other column
        # parses the same either way.
        df = pd.read_csv(csv_path, sep=";", encoding="cp1252", decimal=",")

        for col in DATE_COLUMNS_BY_TABLE.get(table_name, []):
            if col in df.columns:
                parsed = pd.to_datetime(df[col], format=EXPORT_DATETIME_FORMAT, errors="coerce")
                still_missing = parsed.isna()
                if still_missing.any():
                    parsed.loc[still_missing] = pd.to_datetime(
                        df.loc[still_missing, col], format="%d/%m/%Y", errors="coerce"
                    )
                df[col] = parsed.dt.strftime("%Y-%m-%d")
                n_failed = df[col].isna().sum()
                if n_failed:
                    print(f"   ⚠️  {n_failed} rows in '{col}' could not be parsed as a date")

        if table_name == "liveboard_records":
            df = restore_planned_time(df)

        if table_name in DEDUP_KEY_BY_TABLE:
            group_cols, sort_col = DEDUP_KEY_BY_TABLE[table_name]
            if all(c in df.columns for c in group_cols) and sort_col in df.columns:
                before = len(df)
                df = keep_latest_snapshot(df, group_cols, sort_col)
                after = len(df)
                print(
                    f"   🧹 Deduplicated '{table_name}' on {group_cols} (kept latest "
                    f"'{sort_col}' per group): {before:,} -> {after:,} rows"
                )

        df.to_sql(table_name, conn, if_exists="replace", index=False)

        print(f"✅ Loaded '{table_name}': {len(df):,} rows, {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)}")

    conn.commit()

    problems = check_database(conn)
    conn.close()
    if problems:
        raise SystemExit(
            "Build checks failed; the existing database was left untouched:\n- "
            + "\n- ".join(problems)
            + f"\n(the rejected build is at {tmp_path})"
        )
    print("✅ Checks passed: one row per stop event, no NULL key or delay, severity labels match their ranges")

    os.replace(tmp_path, db_path)
    print(f"\nDatabase built at: {db_path}")


if __name__ == "__main__":
    build_database()