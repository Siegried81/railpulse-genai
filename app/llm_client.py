"""
Provider-agnostic LLM client.

call_llm(system_prompt, user_prompt) always returns a plain string,
regardless of which backend is configured in config.LLM_PROVIDER.

Robustness:
- LLMConnectionError wraps any backend failure (Ollama down, bad API key,
  network issue) into one exception type callers can catch regardless of
  provider.
- Groq's free-tier API rate-limits aggressively; retries once on HTTP 429
  before giving up.
"""

from functools import lru_cache
import re

import requests
from groq import Groq, APIConnectionError as GroqAPIConnectionError
from groq import APITimeoutError as GroqAPITimeoutError
from groq import APIStatusError as GroqAPIStatusError
from app import config


class LLMConnectionError(Exception):
    """Raised when the configured LLM backend can't be reached or times out."""


_WORST_DELAY_WORDS = {"worst", "highest", "most", "biggest", "greatest"}
_BEST_DELAY_WORDS = {"best", "lowest", "least", "smallest", "most punctual"}


def validate_superlative_sql_direction(question: str, sql: str) -> bool:
    """Guards against the wrong ORDER BY direction on "worst/best delay"
    questions (e.g. "worst delay" sorted ASC instead of DESC). Only checks
    the "delay" framing (higher = worse); rate/percentage superlatives have
    the opposite mapping and are intentionally not checked here.
    """
    if "delay" not in question.lower():
        return True

    order_match = re.search(r"ORDER BY\s+\S+\s+(ASC|DESC)", sql, re.IGNORECASE)
    if not order_match:
        return True  # no explicit direction to check (e.g. LIMIT-less query)

    direction = order_match.group(1).upper()
    question_lower = question.lower()
    wants_worst = any(w in question_lower for w in _WORST_DELAY_WORDS)
    wants_best = any(w in question_lower for w in _BEST_DELAY_WORDS)

    if wants_worst and not wants_best and direction != "DESC":
        return False
    if wants_best and not wants_worst and direction != "ASC":
        return False
    return True


def validate_no_raw_category_delay_count_ranking(sql: str) -> bool:
    """Reject SQL ranking train_category by raw COUNT(*) of delay events
    instead of AVG(delay_seconds). Categories with more scheduled services
    (e.g. S-trains) will dominate a raw count regardless of their actual
    delay rate -- the correct pattern is AVG(delay_seconds), not COUNT(*).
    Scoped to SQL referencing train_category, so it won't misfire on other
    legitimate COUNT(*) GROUP BY patterns (e.g. delays per hour).
    """
    if "train_category" not in sql.lower():
        return True
    has_group_by = bool(re.search(r"GROUP BY\s+\w*\.?train_category", sql, re.IGNORECASE))
    has_raw_count = bool(re.search(r"COUNT\s*\(\s*\*\s*\)", sql, re.IGNORECASE))
    has_avg_delay = bool(re.search(r"AVG\s*\(\s*\w*\.?delay_seconds", sql, re.IGNORECASE))
    return not (has_group_by and has_raw_count and not has_avg_delay)


def force_no_query_if_raw_category_delay_count_ranking(sql: str) -> str:
    """Deterministic fallback for validate_no_raw_category_delay_count_ranking:
    forces a NO_QUERY refusal rather than let a volume-biased "most delays"
    claim reach the user. Call AFTER extract_sql(), on every SQL string.
    """
    if validate_no_raw_category_delay_count_ranking(sql):
        return sql
    return (
        "NO_QUERY: a raw count of delay events per train category would be misleading -- "
        "a category with more scheduled services (e.g. S-trains) will always accumulate "
        "more delay events even at a normal delay rate. Try asking for the average delay "
        "per train category instead."
    )


_ON_TIME_QUESTION_PATTERN = re.compile(r"\bon[- ]?time\b", re.IGNORECASE)


_NO_QUERY_REASON_KEYWORDS = ("direction", "wheelchair", "platform", "weather")


def validate_no_query_reason_matches_question(question: str, sql: str) -> bool:
    """Reject a NO_QUERY refusal whose stated reason doesn't match what the
    question actually asked about -- e.g. refusing a station-delay question
    with a canned "direction-level data is unavailable" message copied
    verbatim from a different few-shot. A weak model can echo the wrong
    few-shot's NO_QUERY response when confused by an adjacent example,
    over-refusing a perfectly answerable question.
    """
    if not sql.upper().startswith("NO_QUERY"):
        return True
    reason = sql.split(":", 1)[-1].lower()
    q = question.lower()
    for keyword in _NO_QUERY_REASON_KEYWORDS:
        if keyword in reason and keyword not in q:
            return False
    return True


def force_reason_mismatch_worst_best_station(question: str, sql: str) -> str:
    """Deterministic fallback for validate_no_query_reason_matches_question,
    scoped to the demonstrated, safe-to-rewrite case: a "worst/best station
    average delay" question wrongly refused with a canned "direction-level"
    or "platform-level data is unavailable" reason (echoed from an
    unrelated few-shot -- either can appear depending on which nearby
    example the model latches onto). Since this question type has a single
    well-known canonical SQL pattern (documented in prompts.py), it's safe
    to substitute directly rather than just retrying and hoping the model
    self-corrects -- retries alone weren't enough to fix this in practice,
    since a weak model can deterministically re-echo the same wrong
    few-shot every time. Other mismatch types (wheelchair/weather) have no
    equally safe rewrite, so those still only rely on the validator's
    retry pressure.
    """
    if validate_no_query_reason_matches_question(question, sql):
        return sql
    reason = sql.split(":", 1)[-1].lower()
    q = question.lower()
    if ("direction" not in reason and "platform" not in reason) or "station" not in q:
        return sql  # no safe rewrite known for this mismatch -- leave as-is
    wants_worst = any(w in q for w in _WORST_DELAY_WORDS)
    wants_best = any(w in q for w in _BEST_DELAY_WORDS)
    if not (wants_worst or wants_best) and "delay" not in q:
        return sql  # not actually a station-delay ranking question
    direction = "ASC" if (wants_best and not wants_worst) else "DESC"
    date_filter = (
        'WHERE "Scheduled Date" >= (SELECT DATE(MAX("Scheduled Date"), \'-6 days\') '
        'FROM liveboard_records) '
        if "this week" in q
        else ""
    )
    return (
        'SELECT "Stations Name", AVG(delay_seconds) / 60.0 AS avg_delay_minutes, '
        f'COUNT(*) AS sample_size FROM liveboard_records {date_filter}'
        f'GROUP BY "Stations Name" HAVING COUNT(*) >= 10 '
        f'ORDER BY avg_delay_minutes {direction} LIMIT 1;'
    )


def validate_station_query_matches_question_stations(question: str, sql: str) -> bool:
    """Reject SQL that filters WHERE "Stations Name" IN/= to specific
    station names NOT mentioned anywhere in the question -- a sign the
    model blended in station names (and often a stray Hour filter too)
    from an unrelated few-shot example. Observed reproducibly with
    smaller local models (e.g. Ollama's 3B): "which station had the worst
    average delay this week?" answered with a query hard-filtered to
    Bruxelles-Central and Anvers-Central and an Hour BETWEEN 6 AND 12
    clause, none of which the question asked for -- silently narrowing a
    "search all stations" question down to two arbitrary ones.

    A legitimate named comparison (e.g. "compare X and Y") is unaffected,
    since the named station(s) DO appear in the question in that case.
    """
    where_match = re.search(r"\bWHERE\b(.*?)(?:\bGROUP BY\b|\bORDER BY\b|;|$)", sql, re.IGNORECASE | re.DOTALL)
    if not where_match:
        return True
    station_clause_match = re.search(
        r'"?Stations?\s*Name"?\s*(?:=|IN)\s*\(?[^)]*\)?', where_match.group(1), re.IGNORECASE
    )
    if not station_clause_match:
        return True
    quoted_names = re.findall(r"'([^']+)'", station_clause_match.group(0))
    if not quoted_names:
        return True
    q_lower = question.lower()
    return any(name.lower() in q_lower for name in quoted_names)


def validate_on_time_uses_delay_severity(question: str, sql: str) -> bool:
    """Reject SQL answering an "on-time" question without referencing the
    canonical "Delay Severity" = 'On Time (<2min)' definition used
    everywhere else in this project (README, weekly report, other
    few-shots).
    """
    if not _ON_TIME_QUESTION_PATTERN.search(question):
        return True
    return "delay severity" in sql.lower()


_CANONICAL_ON_TIME_SQL = (
    "SELECT ROUND(100.0 * SUM(CASE WHEN \"Delay Severity\" = 'On Time (<2min)' "
    "THEN 1 ELSE 0 END) / COUNT(*), 1) AS on_time_rate_pct FROM liveboard_records;"
)


def force_no_query_if_on_time_wrong_definition(question: str, sql: str) -> str:
    """Deterministic fallback for validate_on_time_uses_delay_severity.

    Rather than refuse outright, this substitutes the known-correct
    canonical query when it's safe to do so -- specifically, when the
    wrong SQL has no WHERE clause, meaning the question is asking for the
    simple, unfiltered overall rate (the common case in practice, and the
    one that actually shows up in testing). Blindly substituting when a
    WHERE clause IS present would silently discard whatever filter
    (station, date, category...) the person actually asked about, which
    is worse than refusing -- so those cases still fall back to a refusal
    asking the person to rephrase.
    """
    if validate_on_time_uses_delay_severity(question, sql):
        return sql
    if not re.search(r"\bWHERE\b", sql, re.IGNORECASE):
        return _CANONICAL_ON_TIME_SQL
    return (
        "NO_QUERY: this on-time question could not be answered using the project's "
        "standard definition (Delay Severity = 'On Time (<2min)'). Try rephrasing, "
        "e.g. \"what percentage of records have Delay Severity 'On Time (<2min)'?\"."
    )


def validate_station_ranking_has_min_sample(sql: str) -> bool:
    """Reject SQL that RANKS stations by AVG(delay_seconds) (an ORDER BY
    ... LIMIT superlative search across all stations) without a
    HAVING COUNT(*) >= N minimum-sample-size filter. Without it, a station
    with just 1-2 records (common for small international border stops,
    e.g. Rosenheim (DE)) can dominate a "worst average delay" ranking with
    a wildly unreliable number, despite this exact guard being documented
    in the few-shot library -- a weak model doesn't always include it.

    Scoped to require ORDER BY ... LIMIT specifically, so it does NOT fire
    on a WHERE-filtered comparison of specific named stations (e.g.
    "compare Bruxelles-Central and Anvers-Central") -- the person named
    the stations themselves, so there's no risk of an obscure low-traffic
    station winning a ranking by accident, and requiring a sample-size
    filter there would incorrectly refuse a perfectly answerable question.
    """
    if not re.search(r'"?Stations?\s*Name"?', sql, re.IGNORECASE):
        return True
    has_group_by_station = bool(re.search(r'GROUP BY\s+"?Stations?\s*Name"?', sql, re.IGNORECASE))
    has_avg_delay = bool(re.search(r"AVG\s*\(\s*delay_seconds", sql, re.IGNORECASE))
    has_order_limit = bool(
        re.search(r"ORDER BY\s+\S+\s+(?:ASC|DESC).*?LIMIT\s+\d+", sql, re.IGNORECASE | re.DOTALL)
    )
    has_having_min = bool(re.search(r"HAVING\s+COUNT\s*\(\s*\*\s*\)\s*>=\s*\d+", sql, re.IGNORECASE))
    return not (has_group_by_station and has_avg_delay and has_order_limit and not has_having_min)


def force_no_query_if_station_ranking_unreliable(sql: str) -> str:
    """Deterministic fallback for validate_station_ranking_has_min_sample."""
    if validate_station_ranking_has_min_sample(sql):
        return sql
    return (
        "NO_QUERY: ranking stations by average delay without a minimum sample size "
        "would let a station with only 1-2 records dominate the result with an "
        "unreliable number. Try again, or ask for the ranking over a shorter, "
        "well-sampled period."
    )


def validate_no_self_referential_case_filter(sql: str) -> bool:
    """Reject SUM(CASE WHEN col = 'x' THEN 1 ELSE 0 END) / COUNT(*) combined
    with GROUP BY that same column -- a tautology that always yields 100%
    for the matching group and 0% elsewhere, regardless of the real data.
    """
    group_match = re.search(r'GROUP BY\s+"?([\w .]+?)"?\s*(?:ORDER BY|HAVING|LIMIT|;|$)', sql, re.IGNORECASE)
    case_match = re.search(r'CASE\s+WHEN\s+"?([\w .]+?)"?\s*=', sql, re.IGNORECASE)
    if not group_match or not case_match:
        return True
    group_col = group_match.group(1).strip().strip('"').lower()
    case_col = case_match.group(1).strip().strip('"').lower()
    return group_col != case_col


def validate_no_unreliable_wheelchair_query(sql: str) -> bool:
    """Reject any query against wheelchair_boarding -- unpopulated for
    virtually every station, so it's a data gap, not a real signal.
    """
    return "wheelchair_boarding" not in sql.lower()


def force_no_query_if_unreliable_wheelchair(sql: str) -> str:
    """Deterministic fallback for validate_no_unreliable_wheelchair_query.
    Call AFTER extract_sql(), on every SQL string.
    """
    if validate_no_unreliable_wheelchair_query(sql):
        return sql
    return (
        "NO_QUERY: wheelchair_boarding data is not reliably populated in this "
        "system (it defaults to \"no information\" for virtually all stations), "
        "so accessibility cannot be reliably answered from this data."
    )


def force_no_query_if_self_referential_case(sql: str) -> str:
    """Deterministic fallback for validate_no_self_referential_case_filter.
    Call AFTER extract_sql(), on every SQL string.
    """
    if validate_no_self_referential_case_filter(sql):
        return sql
    return (
        "NO_QUERY: this question, as phrased, would trivially compute 100% for "
        "one category and 0% for all others by construction, not a real finding. "
        "Try asking for a percentage breakdown by category instead (e.g. \"what "
        "percentage of records fall into each delay severity category?\")."
    )


def validate_no_ungrounded_time_reference(answer: str, sql: str) -> bool:
    """Reject an answer that references a specific hour/time window when
    the SQL never actually queried the "Hour" column.
    """
    if re.search(r'"?Hour"?', sql):
        return True  # Hour genuinely part of the query -- time claims are grounded
    return not _TIME_REFERENCE_PATTERN.search(answer)


_TIME_REFERENCE_PATTERN = re.compile(
    r"\b(peak\w*|rush hour|off[- ]peak|during peak|"
    r"\d{1,2}\s*(?:am|pm)\b|\d{1,2}\s*-\s*\d{1,2}\s*(?:h\b|:00|\)))",
    re.IGNORECASE,
)

# No root-cause column exists anywhere in the schema, so any named cause is
# always fabricated regardless of which SQL ran.
_INVENTED_CAUSE_PATTERN = re.compile(
    r"\b(dwell|boarding delay|alighting|signal(?:ling)?|congestion|coupling|"
    r"shunting|track work|rolling stock|mechanical (?:fault|issue)|technical "
    r"fault|staffing shortage|understaffed)\b",
    re.IGNORECASE,
)


def validate_no_invented_cause(answer: str) -> bool:
    """Reject an answer attributing delay to a specific named cause -- the
    schema has no root-cause column, so any such claim is fabricated.
    """
    return not _INVENTED_CAUSE_PATTERN.search(answer)


def strip_invented_cause(answer: str) -> str:
    """Deterministic fallback for validate_no_invented_cause."""
    if validate_no_invented_cause(answer):
        return answer
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    kept = [s for s in sentences if not _INVENTED_CAUSE_PATTERN.search(s)]
    result = " ".join(kept).strip()
    return result or answer.split(".")[0].strip() + "."


def strip_ungrounded_time_reference(answer: str, sql: str) -> str:
    """Deterministic fallback for validate_no_ungrounded_time_reference."""
    if validate_no_ungrounded_time_reference(answer, sql):
        return answer

    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    kept = [s for s in sentences if not _TIME_REFERENCE_PATTERN.search(s)]
    result = " ".join(kept).strip()
    return result or answer.split(".")[0].strip() + "."


_UNAVAILABLE_DISCLAIMER_PATTERN = re.compile(
    r"\b(isn'?t available|is not available|aren'?t available|are not available|"
    r"not (?:tracked|captured|populated) in this data)\b",
    re.IGNORECASE,
)
# _RECOMMENDATION_LEADIN_PATTERN is defined once, further down (near
# validate_recommendation_targets_worse_group), and reused here too.


_VAGUE_PERIOD_PATTERN = re.compile(
    r"\b(during the period|this week|this period|in this period|over the period)\b",
    re.IGNORECASE,
)
_EXPLICIT_DATE_PATTERN = re.compile(
    r"2026-07-27|2026-08-07|27 Jul|7 Aug|Jul(?:y)? 27|Aug(?:ust)? 7", re.IGNORECASE
)


def _looks_like_undated_count_or_sum(sql: str) -> bool:
    """True when SQL is a plain COUNT()/SUM() with no "Scheduled Date"
    filter -- the number is a total across the whole 12-day dataset.
    """
    if not sql:
        return False
    has_agg = bool(re.search(r"\b(COUNT|SUM)\s*\(", sql, re.IGNORECASE))
    has_date_filter = bool(re.search(r'"?Scheduled Date"?', sql, re.IGNORECASE))
    return has_agg and not has_date_filter


def replace_vague_period_with_dates(answer: str, sql: str) -> str:
    """Deterministic fix for undated COUNT/SUM aggregates: replaces vague
    period language ("this week") with the real dates, or -- if no period
    language at all is present -- injects a neutral date parenthetical
    (language-agnostic, no English words) after the first sentence.
    """
    if not _looks_like_undated_count_or_sum(sql):
        return answer
    if _EXPLICIT_DATE_PATTERN.search(answer):
        return answer  # dates already stated explicitly elsewhere in the answer
    if _VAGUE_PERIOD_PATTERN.search(answer):
        return _VAGUE_PERIOD_PATTERN.sub("from 2026-07-27 to 2026-08-07", answer, count=1)

    sentences = re.split(r"(?<=[.!?])\s+", answer.strip(), maxsplit=1)
    first = sentences[0]
    match = re.match(r"^(.*?)([.!?])\s*$", first)
    if match:
        sentences[0] = f"{match.group(1)} (2026-07-27\u20132026-08-07){match.group(2)}"
    else:
        sentences[0] = f"{first} (2026-07-27\u20132026-08-07)"
    return " ".join(sentences)


def strip_recommendation_after_unavailability_disclaimer(answer: str) -> str:
    """Once the answer admits a dimension isn't available, drop any
    recommendation sentence that follows -- it would be guessing at a fix
    for a gap it just disclosed.
    """
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    disclaimer_seen = False
    kept = []
    for s in sentences:
        if disclaimer_seen and _RECOMMENDATION_LEADIN_PATTERN.search(s):
            continue
        kept.append(s)
        if _UNAVAILABLE_DISCLAIMER_PATTERN.search(s):
            disclaimer_seen = True
    return " ".join(kept).strip()


def _strip_reasoning_leak(text: str) -> str:
    """Strips chain-of-thought leaks some free/routed models produce:
    <think>/<reasoning> tags, a numbered preamble before the real answer,
    and a markdown code fence around SQL. Best-effort defensive fallback.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.IGNORECASE | re.DOTALL)

    # "I need to: 1. ... 2. ..." preamble -- keep only text after the last numbered line
    lines = text.strip().split("\n")
    last_numbered_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*\d+[\.\)]\s", line):
            last_numbered_idx = i
    if last_numbered_idx is not None and last_numbered_idx < len(lines) - 1:
        remainder = "\n".join(lines[last_numbered_idx + 1 :]).strip()
        if remainder:
            text = remainder

    # Markdown code fence -- no-op if absent (normal case for Groq/Ollama)
    fence_match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    return text.strip()


_SMART_QUOTE_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'",  # curly single quotes -> straight
    "\u201c": '"', "\u201d": '"',  # curly double quotes -> straight
    "\u2013": "-", "\u2014": "-",  # en/em dash -> hyphen
})


def _normalize_smart_quotes(text: str) -> str:
    """Normalizes curly quotes/dashes to ASCII so downstream regex
    validators (all written against straight quotes) don't silently miss.
    """
    return text.translate(_SMART_QUOTE_MAP)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    stop: tuple[str, ...] | None = None,
    max_tokens: int | None = None,
) -> str:
    """Route the call to whichever provider is configured.

    `stop` must be a tuple (not a list) so the call is hashable for caching.
    """
    result = _strip_reasoning_leak(_call_llm_cached(config.LLM_PROVIDER, system_prompt, user_prompt, stop, max_tokens))
    return _normalize_smart_quotes(result)


def correct_consultant_numbers(answer: str, rows: list[dict]) -> str:
    """Fixes scale/decimal hallucinations in percentage claims (e.g. 0.6
    misreported as "60%") when the query returned exactly one numeric
    value. Skipped when multiple numeric fields exist -- too ambiguous
    which one a percentage claim refers to.
    """
    numeric_fields = [
        v
        for row in rows
        for v in row.values()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    if len(numeric_fields) != 1:
        return answer

    real_value = float(numeric_fields[0])

    def _replace(match: re.Match) -> str:
        num = float(match.group(1))
        if abs(num - real_value) < 0.015:
            return match.group(0)  # already correct, leave as-is
        formatted = f"{real_value:.2f}".rstrip("0").rstrip(".")
        return f"{formatted}%"

    return re.sub(r"(\d+(?:\.\d+)?)\s*%", _replace, answer)


_BLANKET_CLAIM_RE = re.compile(
    r"\ball\s+(?:the\s+)?stations?\b[^.]{0,60}?(?:100%|perfect|same|identical)"
    r"|\bevery\s+station\b[^.]{0,60}?(?:100%|perfect|same|identical)",
    re.IGNORECASE,
)


def validate_no_false_generalization(answer: str, rows: list[dict]) -> bool:
    """Rejects "all/every station shares value Y" claims when the results
    actually contain more than one distinct value for that field.
    """
    if len(rows) < 2:
        return True
    if not _BLANKET_CLAIM_RE.search(answer):
        return True
    numeric_fields = [
        v
        for row in rows
        for v in row.values()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    distinct = {round(float(v), 1) for v in numeric_fields}
    return len(distinct) <= 1


def correct_consultant_generalization(answer: str, rows: list[dict]) -> str:
    """Deterministic fallback for validate_no_false_generalization: appends
    the true min/max range rather than let a false blanket claim stand.
    """
    if validate_no_false_generalization(answer, rows):
        return answer
    numeric_fields = [
        v
        for row in rows
        for v in row.values()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    if not numeric_fields:
        return answer
    lo, hi = min(numeric_fields), max(numeric_fields)
    return (
        f"{answer} (Correction: values actually range from {lo:g} to {hi:g} "
        f"across the {len(rows)} rows shown -- not all identical.)"
    )


_STATION_LISTING_LEADIN_PATTERN = re.compile(
    r"\bstations?\s+(?:were|was|include[ds]?|are|remain(?:ed)?)\s+"
)
_NAME_TOKEN = r"[A-Z][\w'\u00C0-\u017F]*(?:-[A-Z][\w'\u00C0-\u017F]*)*"


def _extract_station_listing_names(answer: str) -> list[str]:
    """Extracts capitalized name(s) following a "station(s) were/was/are/
    include..." lead-in -- the phrasing used to introduce a list of
    specific stations (e.g. "the worst-performing stations were Hillegem
    and Haaltert"). Handles plain "X and Y" (no comma), "X, Y" and
    "X, Y, and Z" (Oxford comma) list shapes.
    """
    names = []
    for lead_match in _STATION_LISTING_LEADIN_PATTERN.finditer(answer):
        remainder = answer[lead_match.end():]
        list_match = re.match(
            rf"({_NAME_TOKEN}(?:(?:\s*,\s*and\s+|\s*,\s*|\s+and\s+){_NAME_TOKEN})*)",
            remainder,
        )
        if list_match:
            for name in re.split(r"\s*,\s*and\s+|\s*,\s*|\s+and\s+", list_match.group(1)):
                name = name.strip()
                if name:
                    names.append(name)
    return names


def validate_no_invented_station_listing(answer: str, rows: list[dict]) -> bool:
    """Rejects a "stations were X and Y" claim naming a station absent from
    the query results. Complements validate_no_invented_entities, which
    only catches the "<Name> station" phrasing (name BEFORE "station") --
    this catches the much more common reverse phrasing, where "stations"
    leads and the names follow with no per-name "station" suffix to anchor
    on at all, which the older check silently misses entirely.
    """
    allowed = {
        str(v).strip().lower()
        for row in rows
        for v in row.values()
        if isinstance(v, str)
    }
    if not allowed:
        return True
    for name in _extract_station_listing_names(answer):
        if name.lower() not in allowed:
            return False
    return True


def strip_invented_station_listing(answer: str, rows: list[dict]) -> str:
    """Deterministic fallback for validate_no_invented_station_listing:
    drops the offending sentence rather than let a fabricated station name
    reach the report.
    """
    if validate_no_invented_station_listing(answer, rows):
        return answer
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    kept = [s for s in sentences if validate_no_invented_station_listing(s, rows)]
    cleaned = " ".join(kept).strip()
    return cleaned or "No additional detail is available beyond the data shown above."


def validate_no_invented_entities(answer: str, rows: list[dict]) -> bool:
    """Rejects an answer naming a station (or other string entity) that
    isn't literally present in the query results -- the number attached
    can be real while the entity itself is fabricated.
    """
    allowed = {
        str(v).strip().lower()
        for row in rows
        for v in row.values()
        if isinstance(v, str)
    }
    if not allowed:
        return True  # no string entities in the results to check against

    for match in re.finditer(
        r"\b([A-Z][\w\-]*(?:\s+[A-Z][\w\-]*){0,3})\s+station\b", answer
    ):
        name = match.group(1).strip().lower()
        if name in ("the", "this", "that", "each", "every", "no", "any"):
            continue
        if name not in allowed:
            return False
    return True


def correct_no_invented_entities(answer: str, rows: list[dict]) -> str:
    """Deterministic fallback for validate_no_invented_entities: drops just
    the offending sentence(s), keeping the rest of the answer intact.
    """
    if validate_no_invented_entities(answer, rows):
        return answer
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    kept = [s for s in sentences if validate_no_invented_entities(s, rows)]
    cleaned = " ".join(kept).strip()
    return cleaned or "No additional detail is available beyond the data shown above."


_LOW_WORDS = {"lowest", "least", "fewest", "smallest"}
_HIGH_WORDS = {"highest", "most", "greatest", "biggest", "worst"}
# Comparative phrases treated as HIGH-direction triggers alongside the
# single-word list above. Only "more severe" (not bare "severe") -- the
# comparative form inherently implies a superlative claim, whereas bare
# "severe" doesn't (e.g. "a severe delay of 8 minutes" isn't a comparison),
# so adding it alone would risk new false positives.
_HIGH_PHRASES = ("more severe",)


def fix_superlative_claims(answer: str, rows: list[dict], value_field: str, label_field: str) -> str:
    """Catches a false superlative claim (e.g. "Sundays had the LOWEST
    delay" when a different day is actually lower) -- checks the
    comparative claim itself, not just the number or entity. Replaces the
    false sentence with a corrected one naming the real winner, rather
    than just dropping it -- dropping alone can leave a vague, useless
    remainder (e.g. a second sentence saying "this category has the
    highest delay" with no category actually named anywhere). Also fixes
    the reverse: a superlative sentence that names NEITHER a real entity
    NOR a number at all (a non-answer, not even a wrong one) gets the
    correct entity+value inserted instead of being left as empty prose.

    Label matching uses word boundaries, not substring containment --
    short/single-letter labels (e.g. train_category 'S') otherwise
    produce false "mentioned" matches inside unrelated words (e.g. 'S' is
    "found" inside "this", "has", "highest"; 'IC' inside "indicating"),
    which previously masked real hallucinations from detection.
    """
    if not rows:
        return answer

    values = {
        row[label_field]: row[value_field]
        for row in rows
        if value_field in row and label_field in row
    }
    if not values:
        return answer

    true_min, true_max = min(values.values()), max(values.values())
    true_min_label = next(lbl for lbl, v in values.items() if v == true_min)
    true_max_label = next(lbl for lbl, v in values.items() if v == true_max)
    labels_lower = {str(lbl).lower() for lbl in values}

    all_triggers = (_LOW_WORDS | _HIGH_WORDS) | set(_HIGH_PHRASES)
    superlative_re = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in sorted(all_triggers, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )
    number_re = re.compile(r"\d+\.\d+")

    def _mentioned_label(text_lower: str) -> str | None:
        for lbl in labels_lower:
            if re.search(rf"\b{re.escape(lbl)}\b", text_lower):
                return lbl
        return None

    sentences = re.split(r"(?<=[.!?])\s+", answer)
    kept = []
    correction_inserted = {"low": False, "high": False}
    for sentence in sentences:
        sentence_lower = sentence.lower()
        mentioned_label = _mentioned_label(sentence_lower)
        superlative_match = superlative_re.search(sentence)
        number_match = number_re.search(sentence)
        corrected = False

        if superlative_match:
            word = superlative_match.group().lower()
            direction_key = "low" if word in _LOW_WORDS else "high"
            true_label, true_value = (
                (true_min_label, true_min) if word in _LOW_WORDS else (true_max_label, true_max)
            )
            is_false_claim = (
                mentioned_label is not None
                and number_match
                and abs(round(float(number_match.group()), 2) - round(float(true_value), 2)) > 0.02
            )
            is_non_answer = mentioned_label is None and number_match is None

            if is_false_claim or is_non_answer:
                # Only insert the correction once per direction (high/low)
                # -- a second false/vague sentence about the SAME
                # direction would otherwise duplicate the identical
                # correction rather than just being silently dropped.
                if not correction_inserted[direction_key]:
                    kept.append(f"{true_label} actually has the {word} value, at {round(true_value, 2):g}.")
                    correction_inserted[direction_key] = True
                corrected = True
            # else: partial info (entity but no number, or vice versa) --
            # ambiguous to safely rewrite, leave untouched.

        if not corrected:
            kept.append(sentence)

    cleaned = " ".join(kept).strip()
    # Falling back to the original `answer` here would reinstate the exact
    # false claim just dropped, if it was the ONLY sentence. Use a safe
    # generic message instead.
    return cleaned or "No reliable summary could be generated from this data."


_COUNT_LIKE_FIELD_NAMES = {
    "sample_size", "count", "n", "total", "volume", "records",
    "train_volume", "train_count", "canceled_count", "delay_count",
}


def _detect_label_value_fields(rows: list[dict]) -> tuple[str, str] | None:
    """Shared auto-detection for the common single-metric query shape (one
    string label column + one numeric metric column), used by both
    auto_fix_superlative_claims and auto_fix_misdirected_recommendation.

    Count-like columns (sample_size, count, n, volume...) are excluded from
    the numeric-field candidates before checking for ambiguity -- these are
    context/transparency columns, not the metric being claimed about, and
    including them made the ambiguity check bail out on any query that also
    reports a sample size alongside its main metric (a query shape that's
    common and desirable, not actually ambiguous). Returns None when still
    genuinely ambiguous (e.g. two real metric columns).
    """
    if len(rows) < 2:
        return None
    string_fields = {k for row in rows for k, v in row.items() if isinstance(v, str)}
    numeric_fields_all = {
        k
        for row in rows
        for k, v in row.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    numeric_fields = {k for k in numeric_fields_all if k.lower() not in _COUNT_LIKE_FIELD_NAMES}
    if len(string_fields) != 1 or len(numeric_fields) != 1:
        return None
    return next(iter(string_fields)), next(iter(numeric_fields))


def auto_fix_superlative_claims(answer: str, rows: list[dict]) -> str:
    """Auto-detects the label/value column pair for the common single-metric
    shape and applies fix_superlative_claims. Skipped when the shape is
    still ambiguous after excluding count-like columns (see
    _detect_label_value_fields).
    """
    detected = _detect_label_value_fields(rows)
    if detected is None:
        return answer
    label_field, value_field = detected
    return fix_superlative_claims(answer, rows, value_field, label_field)


_RECOMMENDATION_TARGET_PATTERN = re.compile(
    r"\b(?:review|reviewing|investigate|investigating|address|addressing|"
    r"improve|improving|adjust|adjusting|reallocat\w*|focus on|focusing on|"
    r"prioriti[sz]e|prioriti[sz]ing)\s+"
    r"([\w\-]+(?:\s+[\w\-]+){0,2}?)\s+"
    r"(?:operations?|performance|service|schedule|scheduling|efficiency|"
    r"staff(?:ing)?|hours?)\b",
    re.IGNORECASE,
)
_RECOMMENDATION_LEADIN_PATTERN = re.compile(
    r"^\s*(?:[\w\s]{0,30}recommendation\s*:\s*)?(recommend|consider|suggest)\b",
    re.IGNORECASE,
)

# Fields where a LOWER number is worse (opposite of the "delay" default).
_LOWER_IS_WORSE_FIELD_HINTS = ("rate", "percent", "pct", "on_time", "punctual")


def validate_recommendation_targets_worse_group(
    answer: str, rows: list[dict], value_field: str, label_field: str
) -> bool:
    """Rejects a recommendation ("review X operations", "consider flagging Y
    for staff reallocation"...) aimed at the BETTER of two compared groups
    instead of the worse one -- e.g. recommending action on Anvers-Central
    when Bruxelles-Central actually has the higher delay.
    Direction: higher is worse for delay fields, lower is worse for
    rate/percentage fields (see _LOWER_IS_WORSE_FIELD_HINTS).

    Two complementary checks, since real phrasing varies too much for one
    pattern: (1) the narrow verb+operations-word pattern
    (_RECOMMENDATION_TARGET_PATTERN), and (2) any Consider/Recommend/
    Suggest-led sentence that mentions exactly one of the compared group
    labels -- broader, but only fires on unambiguous single-label mentions
    to limit false positives. Best-effort, not a semantic guarantee: a
    recommendation phrased without any of these lead-in words, or one that
    praises the better group rather than flagging the worse one, can still
    slip through.
    """
    if len(rows) < 2:
        return True
    values = {
        str(row[label_field]).strip().lower(): row[value_field]
        for row in rows
        if value_field in row and label_field in row
    }
    if len(values) < 2:
        return True

    lower_is_worse = any(h in value_field.lower() for h in _LOWER_IS_WORSE_FIELD_HINTS)
    worse_label = min(values, key=values.get) if lower_is_worse else max(values, key=values.get)

    for match in _RECOMMENDATION_TARGET_PATTERN.finditer(answer):
        mentioned = match.group(1).strip().lower()
        target_label = next(
            (lbl for lbl in values if lbl in mentioned or mentioned in lbl), None
        )
        if target_label is not None and target_label != worse_label:
            return False

    for sentence in re.split(r"(?<=[.!?])\s+", answer.strip()):
        if not _RECOMMENDATION_LEADIN_PATTERN.search(sentence):
            continue
        sentence_lower = sentence.lower()
        mentioned_labels = [lbl for lbl in values if lbl in sentence_lower]
        if len(mentioned_labels) == 1 and mentioned_labels[0] != worse_label:
            return False

    return True


def auto_fix_misdirected_recommendation(answer: str, rows: list[dict]) -> str:
    """Auto-detects the label/value column pair and applies
    validate_recommendation_targets_worse_group, dropping the offending
    sentence rather than rewriting it.
    """
    detected = _detect_label_value_fields(rows)
    if detected is None:
        return answer
    label_field, value_field = detected

    if validate_recommendation_targets_worse_group(answer, rows, value_field, label_field):
        return answer

    values = {
        str(row[label_field]).strip().lower(): row[value_field]
        for row in rows
        if value_field in row and label_field in row
    }
    lower_is_worse = any(h in value_field.lower() for h in _LOWER_IS_WORSE_FIELD_HINTS)
    worse_label = min(values, key=values.get) if lower_is_worse else max(values, key=values.get)

    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    kept = []
    for sentence in sentences:
        drop = False

        match = _RECOMMENDATION_TARGET_PATTERN.search(sentence)
        if match:
            mentioned = match.group(1).strip().lower()
            target_label = next(
                (lbl for lbl in values if lbl in mentioned or mentioned in lbl), None
            )
            if target_label is not None and target_label != worse_label:
                drop = True

        if not drop and _RECOMMENDATION_LEADIN_PATTERN.search(sentence):
            sentence_lower = sentence.lower()
            mentioned_labels = [lbl for lbl in values if lbl in sentence_lower]
            if len(mentioned_labels) == 1 and mentioned_labels[0] != worse_label:
                drop = True

        if not drop:
            kept.append(sentence)
        # else: drop -- recommendation pointed at the wrong group

    cleaned = " ".join(kept).strip()
    # Falling back to the original `answer` here would reinstate the exact
    # misdirected recommendation just dropped, if it was the ONLY sentence.
    # Use a safe generic message instead.
    return cleaned or "No reliable recommendation could be generated from this data."


def validate_consultant_numbers(answer: str, rows: list[dict]) -> bool:
    """Guards against a restated number not matching the real query result
    (e.g. a decimal-shift hallucination). Checks every decimal or "%"
    number in the answer against the real numeric values in the rows.
    Plain integers (counts, years) aren't checked -- too many false
    rejections on legitimately summarized values.
    """
    real_values = set()
    for row in rows:
        for v in row.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                for ndigits in (0, 1, 2, 3, 4):
                    real_values.add(round(float(v), ndigits))

    if not real_values:
        return True  # nothing numeric to check against (e.g. text-only result)

    answer_decimals = re.findall(r"\d+\.\d+", answer)
    answer_percents = re.findall(r"(\d+(?:\.\d+)?)\s*%", answer)

    for num_str in set(answer_decimals) | set(answer_percents):
        num = round(float(num_str), 4)
        if not any(abs(num - rv) < 0.015 for rv in real_values):
            return False
    return True


def call_llm_until(
    system_prompt: str,
    user_prompt: str,
    stop: tuple[str, ...] | None = None,
    max_tokens: int | None = None,
    validate: callable = None,
    max_attempts: int = 3,
) -> str:
    """Like call_llm, but retries (bypassing the cache) if the response
    fails an optional `validate(text) -> bool` check. Rate-limit (429)
    errors are already retried inside the provider call; this is for
    content quality, not transport errors.
    """
    last_result = ""
    for attempt in range(max_attempts):
        nonce = "" if attempt == 0 else f"\n\n<!-- retry:{attempt} -->"
        last_result = _normalize_smart_quotes(_strip_reasoning_leak(
            _call_llm_cached(config.LLM_PROVIDER, system_prompt, user_prompt + nonce, stop, max_tokens)
        ))
        if validate is None or validate(last_result):
            return last_result
    return last_result


@lru_cache(maxsize=256)
def _call_llm_cached(
    provider: str,
    system_prompt: str,
    user_prompt: str,
    stop: tuple[str, ...] | None,
    max_tokens: int | None,
) -> str:
    if provider == "ollama":
        return _call_ollama(system_prompt, user_prompt, stop, max_tokens)
    elif provider == "groq":
        return _call_groq(system_prompt, user_prompt, stop, max_tokens)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def _call_ollama(
    system_prompt: str, user_prompt: str, stop: tuple[str, ...] | None, max_tokens: int | None
) -> str:
    options = {"temperature": 0}
    if max_tokens:
        options["num_predict"] = max_tokens
    if stop:
        options["stop"] = list(stop)

    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": config.OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": options,
                "keep_alive": "30m",  # keep the model resident between requests
            },
            timeout=300,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise LLMConnectionError(
            f"Can't reach Ollama at {config.OLLAMA_BASE_URL}. Make sure it's running "
            f"(`ollama serve`) and that the model is pulled (`ollama pull {config.OLLAMA_MODEL}`)."
        ) from e
    except requests.exceptions.Timeout as e:
        raise LLMConnectionError(
            "Ollama timed out after 300s. The model may still be loading, or this "
            "machine may be too slow for it -- try a smaller model."
        ) from e
    except requests.exceptions.RequestException as e:
        raise LLMConnectionError(f"Ollama request failed: {e}") from e

    return response.json()["message"]["content"].strip()


_groq_client: Groq | None = None


def _get_groq_client() -> Groq:
    """Lazily builds a singleton Groq SDK client (max_retries=2 handles
    429/5xx backoff internally).
    """
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=config.GROQ_API_KEY, max_retries=2)
    return _groq_client


def _call_groq(
    system_prompt: str, user_prompt: str, stop: tuple[str, ...] | None, max_tokens: int | None
) -> str:
    """Call Groq's free-tier hosted inference API via the official SDK."""
    client = _get_groq_client()

    kwargs: dict = {}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if stop:
        kwargs["stop"] = list(stop)

    try:
        response = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            **kwargs,
        )
    except GroqAPIConnectionError as e:
        raise LLMConnectionError("Can't reach Groq's API. Check your internet connection.") from e
    except GroqAPITimeoutError as e:
        raise LLMConnectionError("Groq API timed out.") from e
    except GroqAPIStatusError as e:
        if e.status_code == 401:
            raise LLMConnectionError("Groq API key is missing or invalid (check GROQ_API_KEY in .env).") from e
        if e.status_code == 429:
            raise LLMConnectionError(
                "Groq free-tier rate limit hit, even after the SDK's built-in retries. "
                "Wait a moment, or check limits at console.groq.com/settings/limits."
            ) from e
        if e.status_code == 413:
            raise LLMConnectionError(
                "This request is too large for Groq's free-tier tokens-per-minute budget "
                "(the prompt itself, or a very large result set being summarized, pushed over "
                "the limit). Wait a few seconds for the TPM window to reset and try again, or "
                "ask a narrower question."
            ) from e
        raise LLMConnectionError(f"Groq API error: {e}") from e

    return response.choices[0].message.content.strip()