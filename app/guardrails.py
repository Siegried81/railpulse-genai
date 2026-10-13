"""
Safety layer: intercepts LLM-generated SQL before execution.

Strategy: a mix of allow-lists and block-lists over the SQL text.
- Allow-listed: the statement shape (one standalone SELECT), the tables it
  reads (KNOWN_TABLES) and the double-quoted column names (KNOWN_COLUMNS).
- Block-listed: write/DDL keywords, comments, sqlite_* internals and a few
  dangerous functions (DANGEROUS_FUNCTIONS).
These are regex checks, not a SQL parser: they can reject an odd but harmless
query, and they are not meant to be the only line of defense.

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

# --- Table allow-list -----------------------------------------------------
# The only tables the app is meant to query. Single source of truth:
# scripts/build_database.py builds exactly these tables from <name>.csv.
# Anything else after FROM/JOIN (sqlite_master, a hallucinated table, a
# table-valued function such as json_each(...)) is rejected.
KNOWN_TABLES = ("liveboard_records", "stations", "routes", "trips", "vehicles")

# Functions with no use in a read-only analytics question but real abuse
# potential:
# - randomblob / zeroblob: allocate a blob of any size in ONE step, so
#   randomblob(900000000) exhausts memory before the time budget in db.py
#   can interrupt it (that budget only counts SQLite VM steps);
# - printf / format: same memory risk via a huge width, e.g. printf('%.*c', 1e9, 'x');
# - load_extension: would load native code (also disabled by default in Python);
# - readfile / writefile / edit / fts3_tokenizer: file access or historic
#   memory-disclosure helpers from optional SQLite extensions.
DANGEROUS_FUNCTIONS = (
    "randomblob", "zeroblob", "printf", "format", "load_extension",
    "readfile", "writefile", "edit", "fts3_tokenizer",
)

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

    # Block SQLite's internal objects (sqlite_master, sqlite_schema,
    # sqlite_sequence, sqlite_version(), ...): schema disclosure, not needed
    # by the app. Matches anywhere, so a string literal 'sqlite_x' is
    # rejected too -- an acceptable false positive.
    if re.search(r"\bsqlite_\w*", cleaned, re.IGNORECASE):
        return False, "SQLite internal objects are not allowed"

    # Block dangerous function calls (see DANGEROUS_FUNCTIONS for why).
    for func in DANGEROUS_FUNCTIONS:
        if re.search(rf"\b{func}\s*\(", cleaned, re.IGNORECASE):
            return False, f"Forbidden function detected: {func}"

    # Table allow-list: the name right after each FROM/JOIN must be a known
    # table. A subquery "FROM (SELECT ...)" is skipped here; its own FROM is
    # checked by the same loop. Limit of a regex: only the FIRST table of a
    # comma list "FROM a, b" is checked. That cannot reach anything sensitive
    # because the database holds only KNOWN_TABLES plus sqlite_* objects,
    # which are blocked above.
    for table in re.findall(r"\b(?:FROM|JOIN)\s+[\"`\[]?(\w+)", cleaned, re.IGNORECASE):
        if table.lower() not in KNOWN_TABLES:
            return False, f"Unknown table referenced: {table}"

    quoted_identifiers = re.findall(r'"([^"]+)"', cleaned)
    for identifier in quoted_identifiers:
        if identifier.lower() not in KNOWN_COLUMNS:
            return False, f"Unknown column referenced: \"{identifier}\" (possible hallucination)"

    return True, ""
