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

# The GTFS-Realtime poller re-captures the same still-upcoming train STOP on
# every poll cycle, so a single (vehicle_id, Scheduled Date, station_id) can
# appear hundreds of times in the raw export (up to ~950x observed).
# We keep only the freshest observation per train/date/station, using
# pulled_at as the recency signal.
#
# CRITICAL: station_id MUST be part of this key. vehicle_id identifies a
# single train JOURNEY (e.g. Munich -> Brussels), not a single stop -- the
# same vehicle_id legitimately appears multiple times for the SAME
# Scheduled Date, once per station it stops at along the route. Without
# station_id in the dedup key, all of a multi-stop train's real, distinct
# stops collapse into a single row (whichever station happened to have the
# latest pulled_at across the WHOLE journey), silently discarding every
# other station that same train legitimately passed through that day.

DEDUP_KEYS_BY_TABLE = {
    "liveboard_records": {
        "subset": ["vehicle_id", "Scheduled Date", "station_id"],
        "recency_col": "pulled_at",
    },
}


def _deduplicate(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    dedup_cfg = DEDUP_KEYS_BY_TABLE.get(table_name)
    if not dedup_cfg:
        return df

    subset = dedup_cfg["subset"]
    recency_col = dedup_cfg["recency_col"]

    if not all(col in df.columns for col in subset + [recency_col]):
        print(f"   ⚠️  Skipping dedup for '{table_name}': expected columns missing")
        return df

    before = len(df)

    # Parse pulled_at to a sortable datetime (best-effort; unparseable
    # values sort first via NaT so they lose ties to real timestamps)
    sort_key = pd.to_datetime(df[recency_col], errors="coerce")
    df = df.assign(_sort_key=sort_key)
    df = df.sort_values("_sort_key", ascending=False)
    df = df.drop_duplicates(subset=subset, keep="first")
    df = df.drop(columns=["_sort_key"])

    after = len(df)
    print(f"   🧹 Deduplicated on {subset} (kept latest '{recency_col}'): "
          f"{before:,} -> {after:,} rows ({before - after:,} duplicates removed)")

    return df


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
                parsed = pd.to_datetime(df[col], format="%d/%m/%Y %H:%M", errors="coerce")
                still_missing = parsed.isna()
                if still_missing.any():
                    parsed.loc[still_missing] = pd.to_datetime(
                        df.loc[still_missing, col], format="%d/%m/%Y", errors="coerce"
                    )
                df[col] = parsed.dt.strftime("%Y-%m-%d")
                n_failed = df[col].isna().sum()
                if n_failed:
                    print(f"   ⚠️  {n_failed} rows in '{col}' could not be parsed as a date")

        df = _deduplicate(df, table_name)

        # trips.trip_id sometimes carries a trailing ":<variant>" suffix
        # (e.g. ":1", ":2") to disambiguate repeated identical services,
        # which liveboard_records.vehicle_id never has. A direct
        # vehicle_id = trip_id join therefore silently misses ~13% of
        # rows. Add a normalized column with that suffix stripped, so
        # queries can join on lr.vehicle_id = t.trip_base_id instead.

        if table_name == "trips" and "trip_id" in df.columns:
            df["trip_base_id"] = df["trip_id"].str.replace(
                r"(:\d{8}):\d+$", r"\1", regex=True
            )

            # trip_base_id collapses multiple trip_id variants (":1", ":2"...)
            # into one value, so it is NOT unique in this table -- joining
            # liveboard_records on it (instead of the original trip_id) would
            # silently multiply rows (one liveboard row matching several
            # trips rows), inflating any downstream AVG()/COUNT(). Sanity
            # check that route_id is consistent across variants of the same
            # base trip then collapse to one row per trip_base_id so the join stays 1:1.
            
            route_id_per_base = df.groupby("trip_base_id")["route_id"].nunique()
            inconsistent = (route_id_per_base > 1).sum()
            if inconsistent:
                print(f"   ⚠️  {inconsistent} trip_base_id value(s) map to more than one "
                      f"route_id -- taking the first seen for each (data may be imprecise)")

            before_trips = len(df)
            df = df.drop_duplicates(subset=["trip_base_id"], keep="first")
            print(f"   🧹 Deduplicated 'trips' on trip_base_id (variants collapsed): "
                  f"{before_trips:,} -> {len(df):,} rows")

        df.to_sql(table_name, conn, if_exists="replace", index=False)

        print(f"✅ Loaded '{table_name}': {len(df):,} rows, {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)}")

    conn.commit()
    conn.close()
    print(f"\nDatabase built at: {DB_PATH}")


if __name__ == "__main__":
    build_database()