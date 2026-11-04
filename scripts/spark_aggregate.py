"""PySpark mirror of the snapshot -> stop-event collapse.

It doesn't replace the pandas pipeline. The reference is
scripts/build_database.py::keep_latest_snapshot, next to this file, which
collapses 980,868 snapshot rows into 99,014 stop events. This script
reproduces that reduction with the Spark DataFrame API, in local mode.

What one STOP EVENT is, and why the key is what it is (mirrored from the
reference, kept here so the file is self-contained):

- A stop event is one train, at one station, on one date:
  key = (vehicle_id, station_id, Scheduled Date). The same still-upcoming stop
  is re-captured on every polling cycle until the train leaves, so raw rows
  over-represent stops that stayed "upcoming" longer. Collapsing to the key
  removes that bias.
- station_id MUST be in the key. Keying on (vehicle_id, Scheduled Date) alone
  keeps a single station per train per day and silently drops ~87% of real
  stop events, so every station-level figure would only see the station where
  each train happened to be polled last.
- Within a group, keep the LATEST snapshot by pulled_at: the final poll before
  the train leaves is the most complete delay reading. All other columns come
  from that winning row, so no field is mixed across snapshots.

pulled_at is parsed with an explicit day-first format ("dd/MM/yyyy HH:mm", the
Power BI export locale). Letting the parser guess turns an ambiguous value like
"01/08/2026" into month-first, every day above 12 fails to parse, and the
rows with a null timestamp would then lose to any row that parsed, whatever
their real time.

Usage:
    python spark_aggregate.py INPUT OUTPUT [--input-format csv|parquet]
                              [--output-format parquet|csv] [--sep ;]
                              [--ts-format "dd/MM/yyyy HH:mm"]

INPUT defaults to CSV with ";" separator and a header, matching the export the
reference pipeline reads. OUTPUT defaults to parquet.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# The reference key and sort column. Kept as a module constant so the test and
# any caller use exactly the pin in build_database.py, not a retyped copy.
STOP_EVENT_KEY: list[str] = ["vehicle_id", "station_id", "Scheduled Date"]
SNAPSHOT_SORT_COL: str = "pulled_at"

# Day-first export timestamp format (French locale), e.g. "07/08/2026 04:15"
# is 7 August, not 8 July.
EXPORT_TS_FORMAT: str = "dd/MM/yyyy HH:mm"


def keep_latest_snapshot(
    df: DataFrame,
    group_cols: Sequence[str],
    sort_col: str,
    ts_format: str = EXPORT_TS_FORMAT,
) -> DataFrame:
    """Keep one row per group_cols: the one with the latest sort_col timestamp.

    Spark mirror of the pandas dedup (sort by parsed timestamp, then
    drop_duplicates(keep="last")). A window partitioned by the key and ordered
    by the parsed timestamp descending picks the latest poll; row_number == 1
    keeps it. A second order key on the original input position (descending)
    pins the tie-break: when two polls share a timestamp, the row that appeared
    later in the input wins, so the result is deterministic rather than left to
    Spark's scan order. Pandas' default sort is not stable, so its
    drop_duplicates(keep="last") does not guarantee that same choice on a tie;
    the two can differ only on the rare stop events that carry two polls at the
    same pulled_at with different readings.

    sort_col is parsed with an explicit day-first format for the reason given
    in the module docstring; an unparseable value becomes null and sorts last
    (nulls_last on a descending order). This is the one place the two differ:
    pandas keeps an unparseable timestamp as the latest poll (NaT sorts last
    and keep='last' takes it), this version never does. On the 99,014 rows
    shipped it makes no difference: every kept row has a dd/MM/yyyy HH:mm
    pulled_at.

    Returns the winning rows with the original columns only (the helper columns
    are dropped), so the output schema equals the input schema -- one row per
    stop event.
    """
    group_cols = list(group_cols)
    parsed_ts = F.to_timestamp(F.col(sort_col), ts_format)

    # A stable per-row id for the tie-break. monotonically_increasing_id() is
    # monotonic with the input order within a partition, which is what pandas'
    # stable sort relies on.
    with_helpers = df.withColumn("_sort_ts", parsed_ts).withColumn(
        "_row_id", F.monotonically_increasing_id()
    )

    window = Window.partitionBy(*group_cols).orderBy(
        F.col("_sort_ts").desc_nulls_last(),
        F.col("_row_id").desc(),
    )

    ranked = with_helpers.withColumn("_rn", F.row_number().over(window))
    return (
        ranked.filter(F.col("_rn") == 1)
        .drop("_sort_ts", "_row_id", "_rn")
    )


def build_spark(app_name: str = "railpulse-snapshot-collapse") -> SparkSession:
    """Open a local SparkSession.

    There is one machine here, so local[*] is all the script needs. It shows
    the same reduction in Spark; it doesn't stand up a cluster.
    """
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def read_snapshots(
    spark: SparkSession, path: str, fmt: str, sep: str
) -> DataFrame:
    """Read the raw snapshot table from CSV or parquet.

    CSV is read with a header and the given separator (";" for the Power BI
    export). inferSchema is left off on purpose: every column is read as text,
    so the day-first timestamp parse stays under this script's control instead
    of Spark guessing a format at read time.
    """
    if fmt == "parquet":
        return spark.read.parquet(path)
    if fmt == "csv":
        return (
            spark.read.option("header", True)
            .option("sep", sep)
            .csv(path)
        )
    raise ValueError(f"unsupported input format: {fmt!r} (use 'csv' or 'parquet')")


def write_events(df: DataFrame, path: str, fmt: str, sep: str) -> None:
    """Write the collapsed stop events out, overwriting any previous run."""
    writer = df.coalesce(1).write.mode("overwrite")
    if fmt == "parquet":
        writer.parquet(path)
    elif fmt == "csv":
        writer.option("header", True).option("sep", sep).csv(path)
    else:
        raise ValueError(f"unsupported output format: {fmt!r} (use 'csv' or 'parquet')")


def collapse_file(
    spark: SparkSession,
    input_path: str,
    output_path: str,
    input_format: str,
    output_format: str,
    sep: str,
    ts_format: str,
) -> tuple[int, int]:
    """Read snapshots, collapse to stop events, write them out.

    Returns (rows_in, rows_out) so a caller (and the CLI) can report the same
    before/after figures the reference pipeline prints.
    """
    snapshots = read_snapshots(spark, input_path, input_format, sep)
    before = snapshots.count()
    events = keep_latest_snapshot(snapshots, STOP_EVENT_KEY, SNAPSHOT_SORT_COL, ts_format)
    events = events.cache()
    after = events.count()
    write_events(events, output_path, output_format, sep)
    return before, after


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="path to the raw snapshot CSV or parquet")
    parser.add_argument("output", help="path to write the collapsed stop events")
    parser.add_argument("--input-format", choices=("csv", "parquet"), default="csv")
    parser.add_argument("--output-format", choices=("csv", "parquet"), default="parquet")
    parser.add_argument("--sep", default=";", help="CSV separator (default ';')")
    parser.add_argument("--ts-format", default=EXPORT_TS_FORMAT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    spark = build_spark()
    try:
        before, after = collapse_file(
            spark,
            args.input,
            args.output,
            args.input_format,
            args.output_format,
            args.sep,
            args.ts_format,
        )
        print(
            f"Collapsed '{args.input}' on {STOP_EVENT_KEY} "
            f"(kept latest '{SNAPSHOT_SORT_COL}' per group): "
            f"{before:,} -> {after:,} rows, written to '{args.output}'"
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
