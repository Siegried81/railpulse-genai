"""
Shared SQL extraction and result-formatting logic, used by both the CLI test
harness and the Streamlit UI, so there is a single source of truth.
"""

import re


def extract_sql(raw_output: str) -> str:
    text = raw_output.strip()

    text = re.sub(r"```sql", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()

    no_query_match = re.match(r"^\s*NO_QUERY\s*:", text, re.IGNORECASE)
    if no_query_match:
        return text.split("\n")[0].strip()

    semicolon_index = text.find(";")
    if semicolon_index != -1:
        text = text[: semicolon_index + 1]

    select_match = re.search(r"SELECT\b.*", text, re.IGNORECASE | re.DOTALL)
    if select_match:
        text = select_match.group(0)

    return text.strip()


CONSULTANT_MAX_ROWS = 20  # rows shown to the consultant LLM; keeps the prompt short


# Result columns known to hold a delay in seconds: the raw column, a bare
# AVG/MIN/MAX/SUM over it (SQLite names an un-aliased column after its
# expression), or any alias ending in "_seconds" (the Text-to-SQL prompt
# requires that suffix for delays left in seconds). COUNT(delay_seconds) is a
# count, not a duration, so it is deliberately not matched.
_SECONDS_COLUMN = re.compile(r"(?i)(avg|min|max|sum)\(\s*delay_seconds\s*\)|\w*_seconds")


def _minutes_column(name: str) -> str | None:
    """Return the minutes name for a seconds column, or None if it isn't one."""
    if not _SECONDS_COLUMN.fullmatch(name):
        return None
    # Found case-insensitively, like the pattern, so "AVG(DELAY_SECONDS)"
    # becomes "AVG(DELAY_minutes)" and no converted value keeps a name that
    # still says seconds.
    cut = name.lower().rfind("seconds")
    return f"{name[:cut]}minutes{name[cut + len('seconds'):]}"


def rows_in_minutes(rows: list[dict]) -> list[dict]:
    """Convert every delay column in seconds to minutes (2 decimals), renamed.

    Delays are stored in seconds, but people read minutes and the consultant
    is forbidden from computing numbers itself. So the conversion happens
    here, deterministically, instead of trusting the SQL to have divided by 60
    or the LLM to do the arithmetic. Used for the consultant input, the
    on-screen table and the CSV download alike. The column is renamed on every
    row (even where the value is NULL) so all rows keep the same columns;
    other columns are left untouched. A seconds column whose name matches none
    of the patterns above cannot be detected and stays as it is.
    """
    converted = []
    for row in rows:
        new_row = {}
        for key, value in row.items():
            minutes_key = _minutes_column(key)
            if minutes_key is None:
                new_row[key] = value
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                new_row[minutes_key] = round(value / 60, 2)
            else:
                new_row[minutes_key] = value
        converted.append(new_row)
    return converted


# Figures every answer may use without them appearing in its input: the
# "Delay Severity" band limits (2, 5, 15 min) and the 0 / 1 / 100 the
# cancellation caveat and plain counting need ("not a confirmed 100%").
_ALWAYS_ALLOWED = (0.0, 1.0, 2.0, 5.0, 15.0, 100.0)

# Digits, with optional thousands separators and decimals. The separator can
# be a comma ("2,520") or the narrow / plain no-break space some models write
# ("21\u202f763"). A regular space is not accepted: "between 2 and 300" must
# stay two numbers.
# A comma followed by one or two digits is a decimal comma ("4,43" in a
# French answer): a thousands group always has three digits. A comma before
# exactly three digits stays a thousands separator, so "4,437" reads 4437.
_THOUSANDS_SEP = ",\u202f\u00a0"
_DECIMAL_COMMA = re.compile(r"\d+,\d{1,2}(?!\d)")
_NUMBER = re.compile(
    rf"\d{{1,3}}(?:[{_THOUSANDS_SEP}]\d{{3}})+(?:\.\d+)?"
    rf"|{_DECIMAL_COMMA.pattern}"
    r"|\d+(?:\.\d+)?"
)


def _normalise(token: str) -> str:
    """Rewrite a _NUMBER match in plain "1234.5" form."""
    if _DECIMAL_COMMA.fullmatch(token):
        return token.replace(",", ".")
    return re.sub(f"[{_THOUSANDS_SEP}]", "", token)


def _to_float(token: str) -> float:
    """Parse a _NUMBER match: thousands separators dropped, decimal comma read as a point."""
    return float(_normalise(token))

# Clock times ("16:00") are labels for an hour value, not measurements; they
# are removed before checking so "between 6:00 and 12:00" is not flagged.
_CLOCK_TIME = re.compile(r"\b\d{1,2}:\d{2}\b")

_WORD_NUM = r"(?:one|two|three|four|five|six|seven|eight|nine|ten)"
# Derived figures written in words, seen in real answers ("roughly one in
# four records", "three of the five peak delays", "roughly one minute more
# per train"): proportions, subset counts, multiples and differences the
# model computed itself.
_DERIVED_PHRASES = re.compile(
    rf"\b{_WORD_NUM}\s+(?:in|out of|of(?:\s+the)?)\s+{_WORD_NUM}\b"
    r"|\b(?:half|a quarter|a third|two thirds|three quarters|twice|triple|twofold|threefold)\b"
    r"|\bdouble\b(?!-)"
    rf"|\b(?:{_WORD_NUM}|a few)\s+(?:\w+\s+)?(?:more|less|fewer|longer|shorter|higher|lower)\b",
    re.IGNORECASE,
)


def ungrounded_figures(answer: str, source_text: str) -> list[str]:
    """List the figures in an LLM answer that are not in the text it was given.

    The consultant and weekly-report prompts forbid computing new numbers,
    but models still do ("roughly one in four", "about one minute more").
    This is the deterministic check behind that rule: every number in the
    answer must match a number in source_text (the exact prompt input) up to
    the answer's own rounding -- "4.4" matches 4.43, "2,520" matches 2520 --
    and derived phrasings in words are flagged outright. An empty list means
    the answer is grounded.
    """
    allowed = [_to_float(n) for n in _NUMBER.findall(source_text)]
    allowed.extend(_ALWAYS_ALLOWED)

    problems = []
    for token in _NUMBER.findall(_CLOCK_TIME.sub(" ", answer)):
        plain = _normalise(token)
        value = float(plain)
        decimals = len(plain.split(".")[1]) if "." in plain else 0
        tolerance = 0.5 * 10 ** -decimals + 1e-9
        if not any(abs(value - a) <= tolerance for a in allowed):
            problems.append(token)
    problems.extend(m.group(0) for m in _DERIVED_PHRASES.finditer(answer))
    return problems


def build_consultant_input(question: str, sql: str, rows: list[dict]) -> str:
    """Build the user prompt for CONSULTANT_SYSTEM_PROMPT.

    Shared by the Streamlit UI and the CLI harness so both send the model
    exactly the same thing, with delays already in minutes.
    """
    return (
        f"Question: {question}\n"
        f"SQL executed: {sql}\n"
        f"Results: {_rounded(rows_in_minutes(rows[:CONSULTANT_MAX_ROWS]))}"
    )


def _rounded(rows: list[dict]) -> list[dict]:
    """Round float values for the consultant: 2 decimals, or 3 significant
    digits below 1 so a small rate does not collapse to 0.

    A minutes average computed in SQL (AVG(delay_seconds) / 60.0) is not a
    seconds column, so rows_in_minutes leaves it alone, and the model then
    copied it verbatim ("4.426991150442478 minutes"). Only the consultant
    input is rounded; the on-screen table and the CSV keep the raw values.
    """
    def _round(value):
        if not isinstance(value, float):
            return value
        return round(value, 2) if abs(value) >= 1 else float(f"{value:.3g}")

    return [{key: _round(value) for key, value in row.items()} for row in rows]