"""
Quick schema/data sanity-check script.

Usage:
    python scripts/check_values.py

Prints the real distinct values for key categorical columns, so prompts.py
can be grounded in actual data instead of assumed labels.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import execute_query

CHECKS = [
    ('SELECT DISTINCT "Delay Severity" FROM liveboard_records', "Delay Severity values"),
    ('SELECT DISTINCT day_of_week FROM liveboard_records', "day_of_week values"),
    ('SELECT DISTINCT "vehicles.direction" FROM liveboard_records LIMIT 10', "direction values (denormalized in liveboard_records)"),
    ('SELECT DISTINCT direction FROM vehicles LIMIT 10', "direction values (raw vehicles table)"),
    ('SELECT COUNT(*) AS total, COUNT(direction) AS non_null_direction FROM vehicles', "vehicles table: null check"),
    ('SELECT DISTINCT train_category FROM routes LIMIT 20', "train_category values"),
    ('SELECT MIN("Scheduled Date") AS min_date, MAX("Scheduled Date") AS max_date FROM liveboard_records', "date range"),
    ('SELECT DISTINCT "stations.wheelchair_boarding" FROM liveboard_records LIMIT 10', "wheelchair_boarding distinct values"),
    ('SELECT "Stations Name", COUNT(*) AS n FROM liveboard_records GROUP BY "Stations Name" ORDER BY n DESC LIMIT 10', "Top 10 busiest stations by record count"),
    ('SELECT DISTINCT canceled, typeof(canceled) AS sqlite_type FROM liveboard_records LIMIT 10', "canceled column: distinct values and type"),
    ('SELECT COUNT(*) AS n FROM liveboard_records WHERE canceled = 1', "canceled = 1 (integer) match count"),
    ('SELECT COUNT(*) AS n FROM liveboard_records WHERE canceled = \'True\'', "canceled = 'True' (text) match count"),
    ('SELECT COUNT(*) AS n_stations_below_10 FROM (SELECT "Stations Name", COUNT(*) AS c FROM liveboard_records WHERE "Scheduled Date" >= (SELECT DATE(MAX("Scheduled Date"), \'-6 days\') FROM liveboard_records) GROUP BY "Stations Name" HAVING c < 10)', "stations with <10 records this week"),
    ('SELECT "Stations Name", COUNT(*) AS c FROM liveboard_records WHERE "Scheduled Date" >= (SELECT DATE(MAX("Scheduled Date"), \'-6 days\') FROM liveboard_records) GROUP BY "Stations Name" ORDER BY c ASC LIMIT 15', "15 lowest-traffic stations this week"),
    ('SELECT vehicle_id FROM liveboard_records LIMIT 5', "sample vehicle_id format"),
    ('SELECT trip_id, route_id FROM trips LIMIT 5', "sample trip_id format"),
    ('SELECT COUNT(*) AS matches FROM liveboard_records lr JOIN trips t ON lr.vehicle_id = t.trip_id', "current join match count (out of 99,014)"),
    ('SELECT DISTINCT route_color, route_text_color FROM routes LIMIT 20', "route_color / route_text_color values"),
    ]

if __name__ == "__main__":
    for sql, label in CHECKS:
        print(f"\n--- {label} ---")
        try:
            rows = execute_query(sql)
            for row in rows:
                print(row)
        except Exception as e:
            print(f"Error: {e}")