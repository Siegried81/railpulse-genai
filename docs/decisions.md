# Decision log

One dated entry per decision that changes what a number means or how the
system behaves. Append; never rewrite past entries.

---

## 2026-09-30 — Dedup liveboard_records per stop event, not per train-day

**What.** `scripts/build_database.py` now keeps the latest `pulled_at` snapshot
per `(vehicle_id, station_id, Scheduled Date)` instead of per
`(vehicle_id, Scheduled Date)`. `pulled_at` is now parsed with the export's
explicit day-first format (`%d/%m/%Y %H:%M`).

**Why.** One row of `liveboard_records` is meant to be one stop event. The old
key kept one station per train per day (the last one polled): 12,596 rows out
of 99,014 real stop events, so every station-level figure saw ~13% of stops,
biased toward wherever each train was last polled. Separately, `pulled_at` was
parsed without a format; pandas guessed month-first from the first value,
313,429 of 980,868 timestamps became NaT, and NaT sorted last so those rows
were kept as the "latest" snapshot (3,020 stop events picked a different
reading).

**What changed in the numbers** (before → after, same queries):

| Figure | Before | After |
|---|---|---|
| Rows in `liveboard_records` | 12,596 | 99,014 |
| On-time rate, all data | 67.2% | 72.2% |
| On-time rate, last 7 days | 79.9% | 73.4% |
| Average delay, all data | 1.72 min | 2.01 min |
| Stations with ≥ 30 records, last 7 days | 61 | 447 |
| Worst avg-delay station, last 7 days | Brussels Airport-Zaventem (3.79 min, n=47) | Haren (4.93 min, n=70) |
| Bruxelles-Central avg delay | 2.63 min (n=366) | 2.91 min (n=2,239) |

`reports/weekly_brief_2026-08-07.md` was generated on the old data and its
figures are superseded.

**Consequence.** A train now appears once per station it served, so "top N
most delayed" lists can repeat the same train at consecutive stations.

**Revisit if.** A loop or terminus-and-back service stops twice at the same
station on the same date (both stops would collapse into one row); or the
`HAVING COUNT(*) >= 30` sample guard, calibrated on the old row counts, turns
out too permissive at ~8x more rows per station.

---

## 2026-09-30 — Minimum station sample for average-delay rankings: 30 → 100

**What.** Every "worst/best average delay" ranking requires
`HAVING COUNT(*) >= MIN_STATION_SAMPLE` (100). The constant lives in
`app/prompts.py` and is used by the Text-to-SQL prompt (rule + few-shot), the
weekly report and `scripts/check_values.py`, instead of a literal 30 in each.

**Why.** After the per-stop dedup, 447 stations cleared 30 stops in the last
7 days, and the top of the ranking was noise: Haren ranked worst at 4.93 min
average on 70 stops, with a standard error of 1.82 min. Per-station delay
standard deviations on that window are 3.27 min (median) to 6.93 min (90th
percentile), so 100 stops give a standard error of about 0.3–0.7 min.

| Threshold | Stations eligible (7 days) | Worst station | Its standard error |
|---|---|---|---|
| 30 | 447 | Haren, 4.93 min, n=70 | 1.82 min |
| 100 | 165 | Nivelles, 4.43 min, n=452 | 0.44 min |

**Revisit if.** The data window changes length (the threshold counts stops,
so a longer window makes it looser), or a question needs small stations
ranked explicitly — then report the sample size and standard error instead of
hiding them behind a threshold.

---

## 2026-09-30 — Weekly report "top anomalies" ranks train runs, not stop events

**What.** `scripts/generate_weekly_report.py` lists the 5 most delayed train
runs, one row per `(vehicle_id, Scheduled Date)`, each shown at the stop where
its delay peaked (earliest such stop on a tie).

**Why.** With one row per stop event, a late train stays late across
consecutive stops and filled the top 5 with duplicates (2 trains × 2 stations
on the current data). `vehicle_id` alone is not a run: 2,826 of 3,617 ids
recur on several dates.

**Revisit if.** A run crosses midnight: its stops then split across two
`Scheduled Date` values and it can appear twice.

---

## 2026-09-30 — Delays shown in minutes in the UI table and CSV

**What.** The Streamlit result table and CSV download go through
`sql_utils.rows_in_minutes()`, like the consultant input. It converts, and
renames to `*minutes`: `delay_seconds`, bare `AVG/MIN/MAX/SUM(delay_seconds)`,
and any alias ending in `_seconds`. The Text-to-SQL prompt now requires the
`_seconds` suffix for any delay left in seconds.

**Why.** The table and CSV showed raw seconds next to an answer in minutes.

**Limit.** A seconds column with any other name (e.g. an alias like
`avg_delay` without a unit) cannot be detected and is shown unchanged.

---

## 2026-09-30 — LLM provider "auto": DeepSeek first, Groq with key rotation

**What.** `LLM_PROVIDER=auto` tries `deepseek` (`deepseek-flash`) then `groq`
(`openai/gpt-oss-120b`), rotating through `GROQ_API_KEY`, `GROQ_API_KEY_2`, …
on a 429 or a rejected key. The provider and model that actually answered are
recorded on every call: `llm_client.LAST_ANSWERED_BY`, an `LLM_ANSWERED` /
`LLM_FALLBACK` line in `query_audit.log`, a caption under each UI answer, and
a footer on the weekly brief.

**Why.** Two different models can now answer the same question, so "which
model said this" has to be on the record rather than assumed. Groq's free tier
is 8K tokens/minute per key and one Text-to-SQL call is ~2.3K prompt tokens;
the 5 keys were checked to have independent quotas, so rotation adds capacity.

**Measured while setting it up.**
- `gpt-oss-120b` with `stop=";"` returned an empty answer: the ";" inside its
  own reasoning ended the generation. For gpt-oss models `stop` is not sent
  and `reasoning_effort` is `low`; `extract_sql()` already cuts at ";".
- `deepseek-flash` used 906 then 3,591 reasoning tokens for the same weekly
  brief at temperature 0; a 700 cap published a brief cut after "Investig",
  and a 2,000 cap still cut it at random. Caps are now 2,000 (SQL,
  consultant) and 8,000 (brief); Groq counts its quota on tokens used, not on
  the cap. A cut-off answer (`finish_reason == "length"`) is now an error, so
  `auto` falls back instead of publishing it.
- Live smoke test, 19 questions: 19/19 handled by DeepSeek, no fallback,
  4 correctly declined as out of scope.

**Revisit if.** DeepSeek's cost matters (it is a paid API, unlike the
"free-to-run" pitch in the README intro), a model is renamed again (a wrong
`GROQ_MODEL` now fails fast with a 404 naming the setting, and is not retried
on other keys), or the consultant keeps deriving numbers not in the results
("roughly one in four") despite the prompt rule.

---

## 2026-09-30 — Figures in LLM answers are checked against their input

**What.** `sql_utils.ungrounded_figures(answer, source_text)` lists every
number in an answer that is not in the exact text the model was given (up to
the answer's own rounding: "4.4" matches 4.43, "2,520" matches 2520), plus
derived figures written in words ("one in four", "one minute more", "half",
"twice"). Clock times and a few fixed constants (severity bands 2/5/15,
0/1/100) are allowed. It is part of `is_valid` for the consultant (UI and
`test_pipeline.py`) and the weekly brief, so a failing answer is regenerated;
if it still fails, the UI shows it with a warning listing the figures, and
the brief is not written. The consultant and brief prompts now name these
derived forms explicitly.

**Why.** The prompts already forbade computing new numbers, and the live smoke
test still produced "roughly one in four" and "roughly one minute more".
A prompt rule is guidance; the check makes it enforced.

**Result (live, DeepSeek).** 19/19 questions, 0 answers flagged, 0 retries
needed: the reworded prompt alone removed the derived figures, and the check
stays as the safety net. The brief now says "89 minutes or more" instead of
"above 89".

**Limit.** Counting words ("three of the five") are not checked, and a
number that happens to appear anywhere in the input (e.g. inside a train ID)
counts as grounded.

---

## 2026-09-30 — "This morning" means the latest date, hours 6–11

**What.** The few-shot for "this morning" now filters on
`MAX("Scheduled Date")` and uses `"Hour" BETWEEN 6 AND 11`.

**Why.** It averaged every morning of the whole dataset while the answer said
"this morning", and `"Hour" 12` (12:00–12:59) is afternoon. Live result:
2.16 min on 2026-08-07 06:00–11:59, where the old query gave 2.88 min over all
dates, 06:00–12:59.

---

## 2026-10-10 — A query has a time budget, not only a row cap

**What.** `app/db.py::execute_query` installs a SQLite progress handler that
interrupts the statement after `MAX_QUERY_SECONDS` (5 s). The error names the
budget and says what to do (narrow the question, add a LIMIT).

**Why.** `fetchmany(MAX_ROWS)` bounds what comes back, not the work SQLite does
to produce it: a cartesian join or an unindexed aggregate over the 980,868
polling rows runs to completion before the first row is fetched, and the
read-only connection does nothing against that. A Text-to-SQL model writes
such a query sooner or later. Measured in the suite: a four-way self-join on
the fixture table is stopped within the budget; a normal `COUNT(*)` is
untouched.

**Revisit if** a legitimate question needs more than 5 s on the real
database — raise the constant, or pre-aggregate that question into a view.
