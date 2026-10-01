"""
Build railpulse_ai.db (SQLite) from CSV files exported out of Power BI.

Usage:
    python scripts/build_database.py

Expects CSVs in data/ and writes data/railpulse_ai.db
"""

import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_CSV_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "data" / "railpulse_ai.db"

TABLES = {
    "liveboard_records": "liveboard_records.csv",
    "stations": "stations.csv",
    "routes": "routes.csv",
    "trips": "trips.csv",
    "vehicles": "vehicles.csv",
}

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
# 99,014 distinct stop events). Left as-is, every aggregate (AVG delay,
# on-time %, station rankings, etc.) is skewed toward whichever stops
# happened to get polled most, not toward real-world frequency.
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
    """
    sort_dt = pd.to_datetime(df[sort_col], format=EXPORT_DATETIME_FORMAT, errors="coerce")
    return (
        df.assign(_sort_dt=sort_dt)
        .sort_values("_sort_dt")
        .drop_duplicates(subset=group_cols, keep="last")
        .drop(columns="_sort_dt")
    )


def build_database() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Removed existing database at {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    for table_name, csv_filename in TABLES.items():
        csv_path = RAW_CSV_DIR / csv_filename

        if not csv_path.exists():
            print(f"⚠️  Skipping '{table_name}': file not found at {csv_path}")
            continue

        df = pd.read_csv(csv_path, sep=";", encoding="cp1252")

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
    conn.close()
    print(f"\nDatabase built at: {DB_PATH}")


if __name__ == "__main__":
    build_database()