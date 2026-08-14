"""
Generate a weekly operations report for RailPulse AI.

Runs a fixed set of aggregate SQL queries (on-time rate, delay severity
breakdown, best/worst stations, busiest stations, cancellations, average
delay by day of week, average delay by train category), asks the LLM for
a short executive-summary narrative grounded in those aggregates, and
appends a fully deterministic per-station detail table straight from SQL
(no LLM involved in that part -- zero hallucination risk on the table).

Writes the result to reports/weekly_report_<YYYYMMDD>.md.

Usage:
    python scripts/generate_weekly_report.py
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import execute_query
from app.llm_client import (
    call_llm_until,
    correct_consultant_generalization,
    correct_consultant_numbers,
    correct_no_invented_entities,
    fix_superlative_claims,
    strip_invented_station_listing,
    validate_consultant_numbers,
    validate_no_false_generalization,
    validate_no_invented_entities,
    validate_no_invented_station_listing,
)
from app.prompts import WEEKLY_REPORT_SYSTEM_PROMPT

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

# Minimum sample size for any per-station/per-category stat, to avoid a
# 1-2 record station dominating a ranking with an unreliable average --
# same rule the Text-to-SQL prompt applies to chat queries.
# Minimum sample size for any per-station stat, to avoid a 1-2 record
# station dominating a ranking with an unreliable average -- same rule the
# Text-to-SQL prompt applies to chat queries.
MIN_SAMPLE_SIZE = 10

# Separate, lower threshold for train categories: there are only ~11 total
# (IC, S, L, BUS, P, TRN, OTC, NJ, EC, T, EXT), unlike hundreds of stations,
# so the same 10-record cutoff can silently erase an entire real category
# from every result (e.g. NJ/Nightjet, which runs rarely by nature) rather
# than just filtering out statistical noise. A lower bar here still guards
# against a single outlier trip skewing an average, without hiding a
# legitimate, low-frequency category entirely.
MIN_CATEGORY_SAMPLE_SIZE = 5

QUERIES = {
    "date_range": (
        'SELECT MIN("Scheduled Date") AS start_date, MAX("Scheduled Date") AS end_date '
        "FROM liveboard_records"
    ),
    "on_time_rate": (
        "SELECT ROUND(100.0 * SUM(CASE WHEN \"Delay Severity\" = 'On Time (<2min)' "
        "THEN 1 ELSE 0 END) / COUNT(*), 1) AS on_time_rate_pct FROM liveboard_records"
    ),
    "severity_breakdown": (
        'SELECT "Delay Severity", COUNT(*) AS total, '
        "ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM liveboard_records), 1) AS pct_of_total "
        'FROM liveboard_records GROUP BY "Delay Severity" ORDER BY pct_of_total DESC'
    ),
    "worst_stations": (
        'SELECT "Stations Name", ROUND(AVG(delay_seconds) / 60.0, 2) AS avg_delay_minutes, '
        f'COUNT(*) AS sample_size FROM liveboard_records GROUP BY "Stations Name" '
        f"HAVING COUNT(*) >= {MIN_SAMPLE_SIZE} ORDER BY avg_delay_minutes DESC LIMIT 5"
    ),
    "best_stations": (
        'SELECT "Stations Name", ROUND(AVG(delay_seconds) / 60.0, 2) AS avg_delay_minutes, '
        f'COUNT(*) AS sample_size FROM liveboard_records GROUP BY "Stations Name" '
        f"HAVING COUNT(*) >= {MIN_SAMPLE_SIZE} ORDER BY avg_delay_minutes ASC LIMIT 5"
    ),
    "busiest_stations": (
        'SELECT "Stations Name", COUNT(*) AS train_volume FROM liveboard_records '
        'GROUP BY "Stations Name" ORDER BY train_volume DESC LIMIT 5'
    ),
    "cancellations": (
        "SELECT COUNT(*) AS total_canceled FROM liveboard_records WHERE canceled = 1"
    ),
    "delay_by_day": (
        "SELECT day_of_week, ROUND(AVG(delay_seconds) / 60.0, 2) AS avg_delay_minutes "
        "FROM liveboard_records GROUP BY day_of_week ORDER BY day_number"
    ),
    "train_category_delays": (
        "SELECT r.train_category, ROUND(AVG(lr.delay_seconds) / 60.0, 2) AS avg_delay_minutes, "
        "COUNT(*) AS sample_size FROM liveboard_records lr "
        "JOIN trips t ON lr.vehicle_id = t.trip_base_id "
        "JOIN routes r ON t.route_id = r.route_id "
        f"GROUP BY r.train_category HAVING COUNT(*) >= {MIN_CATEGORY_SAMPLE_SIZE} "
        "ORDER BY avg_delay_minutes DESC"
    ),
    # Full per-station detail table -- rendered directly as markdown further
    # down, NOT passed through the LLM, so it carries zero hallucination risk.
    "station_detail": (
        'SELECT "Stations Name", COUNT(*) AS sample_size, '
        "AVG(delay_seconds) / 60.0 AS avg_delay_minutes, "
        "ROUND(100.0 * SUM(CASE WHEN \"Delay Severity\" = 'On Time (<2min)' "
        "THEN 1 ELSE 0 END) / COUNT(*), 1) AS on_time_rate_pct "
        'FROM liveboard_records GROUP BY "Stations Name" '
        f"HAVING COUNT(*) >= {MIN_SAMPLE_SIZE} ORDER BY sample_size DESC"
    ),
}


def _run_all_queries() -> dict:
    results = {}
    for name, sql in QUERIES.items():
        try:
            results[name] = execute_query(sql)
        except ValueError as e:
            print(f"⚠️  Query '{name}' failed: {e}")
            results[name] = []
    return results


def _markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "_No data available._\n"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        formatted = []
        for h in headers:
            v = row[h]
            formatted.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines) + "\n"


def generate_report() -> Path:
    print("Running aggregate queries...")
    data = _run_all_queries()

    date_range = data["date_range"][0] if data["date_range"] else {}
    start_date = date_range.get("start_date", "unknown")
    end_date = date_range.get("end_date", "unknown")

    # Payload the LLM sees for the executive summary. Deliberately excludes
    # the full per-station detail table -- that's rendered separately,
    # straight from SQL, with no LLM involved.
    summary_payload = (
        f"Date range: {start_date} to {end_date}\n"
        f"Overall on-time rate: {data['on_time_rate']}\n"
        f"Delay severity breakdown: {data['severity_breakdown']}\n"
        f"Worst 5 stations (min {MIN_SAMPLE_SIZE} records): {data['worst_stations']}\n"
        f"Best 5 stations (min {MIN_SAMPLE_SIZE} records): {data['best_stations']}\n"
        f"Busiest 5 stations by volume: {data['busiest_stations']}\n"
        f"Total canceled trains: {data['cancellations']}\n"
        f"Average delay by day of week: {data['delay_by_day']}\n"
        f"Average delay by train category (min {MIN_CATEGORY_SAMPLE_SIZE} records): "
        f"{data['train_category_delays']}\n"
    )

    # All numeric/string values across every query result, flattened into
    # one list of "rows" so the guard functions can check the summary
    # against every real figure and entity at once.
    all_rows = [
        row
        for key in (
            "on_time_rate",
            "severity_breakdown",
            "worst_stations",
            "best_stations",
            "busiest_stations",
            "cancellations",
            "delay_by_day",
            "train_category_delays",
        )
        for row in data[key]
    ]

    print("Generating executive summary...")
    summary = call_llm_until(
        WEEKLY_REPORT_SYSTEM_PROMPT,
        summary_payload,
        validate=lambda text: validate_consultant_numbers(text, all_rows)
        and validate_no_false_generalization(text, all_rows)
        and validate_no_invented_entities(text, all_rows)
        and validate_no_invented_station_listing(text, all_rows),
    )
    summary = correct_consultant_numbers(summary, all_rows)
    summary = correct_consultant_generalization(summary, all_rows)
    summary = correct_no_invented_entities(summary, all_rows)
    summary = strip_invented_station_listing(summary, all_rows)
    summary = fix_superlative_claims(
        summary, data["delay_by_day"], "avg_delay_minutes", "day_of_week"
    )
    summary = fix_superlative_claims(
        summary, data["train_category_delays"], "avg_delay_minutes", "train_category"
    )

    report_lines = [
        "# RailPulse Weekly Operations Report",
        "",
        f"**Period covered:** {start_date} to {end_date}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Executive Summary",
        "",
        summary,
        "",
        "## Station-by-Station Detail",
        "",
        f"_Stations with fewer than {MIN_SAMPLE_SIZE} records in this period are excluded as "
        "statistically unreliable._",
        "",
        _markdown_table(data["station_detail"]),
    ]
    report_text = "\n".join(report_lines)

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / f"weekly_report_{datetime.now().strftime('%Y%m%d')}.md"
    out_path.write_text(report_text, encoding="utf-8")

    print(f"\n✅ Report written to: {out_path}")
    return out_path


if __name__ == "__main__":
    generate_report()