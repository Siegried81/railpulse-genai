"""
Generate a weekly executive brief (Markdown) from the top delay anomalies
in the database, written by the open-source LLM.

Usage:
    python scripts/generate_weekly_report.py

Writes reports/weekly_brief_<most-recent-date-in-data>.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import execute_query
from app import llm_client
from app.llm_client import call_llm_until, looks_like_leaked_meta
from app.prompts import MIN_STATION_SAMPLE, REPORT_MAX_TOKENS, WEEKLY_REPORT_SYSTEM_PROMPT
from app.sql_utils import ungrounded_figures

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"


def gather_weekly_data() -> dict:
    """Pull the stats the report is built from. All queries anchor on the
    most recent date actually in the data (this is a fixed historical
    snapshot, not a live feed -- see prompts.py for why DATE('now') is
    never used anywhere in this app).
    """
    date_range = execute_query(
        'SELECT DATE(MAX("Scheduled Date"), \'-6 days\') AS week_start, '
        'MAX("Scheduled Date") AS week_end FROM liveboard_records'
    )[0]

    # One row per train run, not per stop: a late train stays late for several
    # consecutive stops, so ranking raw stop events fills the top 5 with the
    # same train seen at neighbouring stations. A run is (vehicle_id,
    # Scheduled Date) because the same vehicle_id recurs on other days. Each
    # run is shown at the stop where its delay peaked; on a tie, the earliest
    # such stop (scheduled_time), i.e. where that delay was first reached.
    top_anomalies = execute_query(
        'SELECT "Stations Name", vehicle_id, delay_seconds / 60.0 AS delay_minutes, '
        '"Scheduled Date" FROM ('
        'SELECT "Stations Name", vehicle_id, delay_seconds, "Scheduled Date", '
        'ROW_NUMBER() OVER (PARTITION BY vehicle_id, "Scheduled Date" '
        "ORDER BY delay_seconds DESC, scheduled_time ASC) AS run_rank "
        'FROM liveboard_records WHERE "Scheduled Date" >= '
        '(SELECT DATE(MAX("Scheduled Date"), \'-6 days\') FROM liveboard_records)'
        ") WHERE run_rank = 1 "
        "ORDER BY delay_seconds DESC LIMIT 5"
    )

    on_time_rate = execute_query(
        "SELECT ROUND(100.0 * SUM(CASE WHEN \"Delay Severity\" = 'On Time (<2min)' "
        "THEN 1 ELSE 0 END) / COUNT(*), 1) AS on_time_rate_pct "
        'FROM liveboard_records WHERE "Scheduled Date" >= '
        '(SELECT DATE(MAX("Scheduled Date"), \'-6 days\') FROM liveboard_records)'
    )[0]["on_time_rate_pct"]

    worst_station = execute_query(
        'SELECT "Stations Name", AVG(delay_seconds) / 60.0 AS avg_delay_minutes, '
        "COUNT(*) AS sample_size "
        'FROM liveboard_records WHERE "Scheduled Date" >= '
        '(SELECT DATE(MAX("Scheduled Date"), \'-6 days\') FROM liveboard_records) '
        f'GROUP BY "Stations Name" HAVING COUNT(*) >= {MIN_STATION_SAMPLE} '
        "ORDER BY avg_delay_minutes DESC LIMIT 1"
    )
    worst_station = worst_station[0] if worst_station else None

    canceled_count = execute_query(
        'SELECT COUNT(*) AS n FROM liveboard_records WHERE canceled = 1 '
        'AND "Scheduled Date" >= (SELECT DATE(MAX("Scheduled Date"), \'-6 days\') '
        "FROM liveboard_records)"
    )[0]["n"]

    return {
        "week_start": date_range["week_start"],
        "week_end": date_range["week_end"],
        "top_anomalies": top_anomalies,
        "on_time_rate_pct": on_time_rate,
        "worst_station": worst_station,
        "canceled_count": canceled_count,
    }


def build_report_input(data: dict) -> str:
    lines = [
        f"Week: {data['week_start']} to {data['week_end']}",
        f"Overall on-time rate this week: {data['on_time_rate_pct']}%",
        f"Cancellations recorded this week: {data['canceled_count']} "
        "(caveat: source feed may not reliably capture real-world cancellations)",
    ]
    if data["worst_station"]:
        ws = data["worst_station"]
        lines.append(
            f"Worst average delay this week: {ws['Stations Name']} at "
            f"{ws['avg_delay_minutes']:.2f} min avg (sample size {ws['sample_size']})"
        )
    lines.append(
        "\nTop 5 most delayed train runs this week "
        "(one row per train run, at the stop where its delay peaked):"
    )
    for row in data["top_anomalies"]:
        lines.append(
            f"- Station: {row['Stations Name']} | Train: {row['vehicle_id']} | "
            f"Delay: {row['delay_minutes']:.1f} min | Date: {row['Scheduled Date']}"
        )
    return "\n".join(lines)


def generate_report() -> Path:
    print("Gathering weekly stats from the database...")
    data = gather_weekly_data()
    report_input = build_report_input(data)
    print(f"\n--- Data fed to the LLM ---\n{report_input}\n---------------------------\n")

    print("Asking the LLM to write the executive brief...")
    is_valid = lambda s: (  # noqa: E731
        not looks_like_leaked_meta(s)
        and s.strip().startswith("#")
        and not ungrounded_figures(s, report_input)
    )
    report_md = call_llm_until(
        WEEKLY_REPORT_SYSTEM_PROMPT,
        report_input,
        is_valid=is_valid,
        max_tokens=REPORT_MAX_TOKENS,
    )
    # call_llm_until() returns its last attempt even when invalid; writing that
    # would overwrite the previous brief with a broken one.
    if not is_valid(report_md):
        raise SystemExit(
            "LLM output is not a valid brief; nothing written. "
            f"Ungrounded figures: {ungrounded_figures(report_md, report_input)}. Got:\n{report_md[:300]}"
        )

    # Appended in code, not asked of the model: which model wrote the brief
    # is a fact the reader needs, and the model cannot be trusted to state it.
    report_md = f"{report_md.rstrip()}\n\n---\n_Written by {llm_client.LAST_ANSWERED_BY} from figures pre-computed in SQL._\n"

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / f"weekly_brief_{data['week_end']}.md"
    out_path.write_text(report_md, encoding="utf-8")
    print(f"\n✅ Report written to: {out_path}")
    return out_path


if __name__ == "__main__":
    generate_report()