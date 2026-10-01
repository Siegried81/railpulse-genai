"""
Safety layer: intercepts LLM-generated SQL before execution.

Strategy: allow-list, not just block-list.
Only a single, standalone SELECT statement is ever allowed.

Note: this is one layer of defense. The actual database connection
(see db.py) is also opened in true read-only mode as a second,
independent layer -- so even a query that slips past this filter
cannot write to the database.
"""

import re

FORBIDDEN_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
    "CREATE", "REPLACE", "ATTACH", "DETACH", "PRAGMA", "VACUUM",
    "GRANT", "REVOKE", "EXEC", "EXECUTE", "LOAD_EXTENSION",
]

MAX_QUERY_LENGTH = 1000  # generous for this schema; blocks pathological queries

# --- Column whitelist -----------------------------------------------------
# CRITICAL: SQLite has a quirky fallback behavior where a double-quoted
# identifier that doesn't match any real column is silently treated as a
# STRING LITERAL instead of raising an error (e.g. SELECT "weather" FROM t
# just returns the literal text "weather" for every row instead of failing).
# This means a hallucinated column name from the LLM can silently "succeed"
# and return garbage instead of erroring loudly. We defend against this by
# whitelisting every column name that legitimately exists in our schema and
# rejecting any double-quoted identifier that isn't in that list.
KNOWN_COLUMNS = {
    "record_id", "station_id", "vehicle_id", "platform", "scheduled_time",
    "delay_seconds", "canceled", "pulled_at", "stations name",
    "stations.standard_name", "stations.latitude", "stations.longitude",
    "stations.wheelchair_boarding", "stations.location_type",
    "vehicles.vehicle_type", "vehicles.direction", "hour", "day_of_week",
    "day_number", "delay severity", "scheduled date",
    "name", "stations names", "latitude", "longitude", "wheelchair_boarding",
    "location_type", "route_id", "route_short_name", "route_long_name",
    "route_desc", "route_color", "route_text_color", "route_type",
    "route_url", "agency_id", "train_category", "trip_id", "vehicle_type",
    "direction",
}


def validate_query(sql: str) -> tuple[bool, str]:
    """
    Returns (is_safe, reason).
    reason is empty string when is_safe is True.
    """
    if not sql or not sql.strip():
        return False, "Empty query"

    cleaned = sql.strip().rstrip(";")

    if len(cleaned) > MAX_QUERY_LENGTH:
        return False, "Query exceeds maximum allowed length"

    # Must be a single statement: no semicolon-separated stacking
    if ";" in cleaned:
        return False, "Multiple statements are not allowed"

    # Must start with SELECT (case-insensitive)
    if not re.match(r"^\s*SELECT\b", cleaned, re.IGNORECASE):
        return False, "Only SELECT statements are allowed"

    # Block forbidden keywords anywhere in the query (word boundaries)
    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", cleaned, re.IGNORECASE):
            return False, f"Forbidden keyword detected: {keyword}"

    # Block SQL comment injection tricks 
    if "--" in cleaned or "/*" in cleaned:
        return False, "SQL comments are not allowed"

    # Block schema-introspection functions (info disclosure, not destructive
    # but not needed for this app's use case)
    if re.search(r"\bpragma_\w+\s*\(", cleaned, re.IGNORECASE):
        return False, "Schema introspection functions are not allowed"

    quoted_identifiers = re.findall(r'"([^"]+)"', cleaned)
    for identifier in quoted_identifiers:
        if identifier.lower() not in KNOWN_COLUMNS:
            return False, f"Unknown column referenced: \"{identifier}\" (possible hallucination)"

    return True, ""