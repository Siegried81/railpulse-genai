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

---

## 2026-10-14 — scheduled_time and Hour are the timetable time, not the drifting one

**What.** `scripts/build_database.py` now rewrites `scheduled_time` as
`scheduled_time - delay_seconds` and re-derives `Hour`, `day_of_week` and
`day_number` from it, before the dedup.

**Why.** In the Power BI export `scheduled_time` moves with the delay: one stop
reads 08:05 with 0 s, then 08:06 with 60 s. `Hour` was cut on that moving value,
so a late train slid into the next hour and its own delay decided which hour it
was counted in. Measured on the 99,014 stop events: 3,205 (3.2%) had an `Hour`
different from their timetable hour.

**What changed in the numbers** (before → after, rebuilt DB):

| Figure | Before | After |
|---|---|---|
| Stop events | 99,014 | 99,014 (same keys, same delays) |
| Rows whose `Hour` differs from the timetable hour | 3,205 | 0 |
| On-time rate (< 2 min), all data | 72.24% | 72.24% |
| Delayed stops in the worst hour (15h) | 4,710 | 4,718 |

Every hour-based answer ("this morning", "busiest hour") can move by the
stops that changed hour; station, date and delay figures do not change.

**Revisit if.** A future export carries the timetable time in its own column:
use it instead of subtracting the delay.

---

## 2026-11-02 — auto chain leads with free Groq, then DeepSeek, OpenRouter, Ollama

**What.** `config.AUTO_PROVIDER_ORDER` changed from `("deepseek", "groq")` to
`("groq", "deepseek", "openrouter", "ollama")`. `.env` stays on
`LLM_PROVIDER=auto`.

**Why.** `auto` tried DeepSeek first, so every call spent DeepSeek credit while
the Groq free tier sat unused. Groq now answers first; DeepSeek is the first
fallback, then the OpenRouter free tier, then a local Ollama. A provider with
no key or no local server raises `LLMConnectionError` and is skipped, so adding
OpenRouter and Ollama costs nothing when they are not configured.

**What changed in the numbers.** No data figure moves. Which model answers
changes: `LAST_ANSWERED_BY` now reads `groq · openai/gpt-oss-120b` on the normal
path instead of DeepSeek. Tests: 84 → 85 passed (added
`test_auto_provider_order_default`; the 2 pre-existing `test_streamlit_app`
failures are the `AppTest` socket block under this interpreter, green in the
Windows `.venv`).

**Known edge.** With OpenRouter and Ollama in the chain but unconfigured, a run
where Groq and DeepSeek are both rate-limited (429) ends as a plain
`LLMConnectionError` rather than `LLMRateLimitError`, because the unreachable
OpenRouter/Ollama raise a non-rate-limit error last. So `call_llm_until()` does
not do its 5 s rate-limit backoff in that case. Acceptable while those two are
last-resort fallbacks.

**Revisit if.** DeepSeek credit is exhausted for good (drop it from the chain),
or an OpenRouter key / local Ollama is added and should be reflected in the
`.env` comments and README.

---

## 2026-11-04 — JSON API over the pipeline, sharing one `answer_question`

**What.** Added a FastAPI app in `api/main.py` (`GET /api/health`,
`GET /api/config`, `POST /ask`) and extracted the text-to-SQL + consultant
pipeline out of the Streamlit page into `app/engine.py::answer_question`. The
Streamlit page now calls that function instead of running the steps inline, so
the UI and the API share one path. Additive: no pipeline behaviour changed.
`fastapi==0.162.0` and `uvicorn==0.54.0` added to `requirements.txt`.
Launch: `python -m uvicorn api.main:app --host 0.0.0.0 --port 8003`.

**Why.** The API needed the same steps the UI runs. Two copies of a pipeline
with safety checks drift; one function cannot. Every guardrail is kept by
construction (one LLM call per step, read-only DB + `validate_query`,
`ungrounded_figures` on the answer, `LAST_ANSWERED_BY` recorded). The API maps
`LLMConnectionError` → 503 and the guardrail/timeout `ValueError` → 422; a
NO_QUERY verdict is a normal 200 with `no_query: true` and null `sql`/`rows`.

**What changed in the numbers.** No data figure moves. One visible UI change:
the three stage-specific spinners ("Translating...", "Querying...",
"Preparing...") became one ("Working on your answer..."), because the three
steps are now a single function call. The answer, SQL, table, captions,
warnings and error messages are unchanged.

**Behaviour kept exactly.** An `LLMConnectionError` raised after the SQL ran
carries `.sql` and `.rows`, and the guardrail `ValueError` carries `.sql`, so
the Streamlit history still shows the query (and rows) for those error cases as
it did inline. The API ignores those attributes.

**Tests.** 87 → 102 passed. Added `tests/test_engine.py` (6) and
`tests/test_api.py` (9, one parametrized over two inputs). All LLM calls mocked at `requests.post`, the DB is a
tiny temp SQLite file, no real network.

**Revisit if.** A client needs the ungrounded caveat already folded into
`answer` (it is returned raw plus the `ungrounded` list), or the API grows
write/stateful endpoints (then it needs the auth the read-only endpoints don't).

---

## 2026-11-04 — tests allow loopback sockets so TestClient can run

**What.** `tests/conftest.py::no_network` now allows connections to
`127.0.0.1` / `::1` / `localhost` and blocks every other host, instead of
blocking all `socket.connect`.

**Why.** FastAPI's `TestClient` runs the app in-process and needs a self-pipe
socket (on Windows it connects over loopback); the blanket block raised inside
the client before any test code ran. Every LLM provider is an external host, so
the "an unmocked provider call fails loudly instead of spending quota" guarantee
is unchanged — loopback reaches no provider.

**What changed in the numbers.** Nothing measured. Test-only change.

**Revisit if.** A test ever needs to assert that a loopback call is blocked, or
a provider is ever addressed over loopback (a local mock server) and should be
caught instead of allowed.

---

## 2026-11-04 — note on the local `.venv`

**What.** The `.venv` in this checkout is Python 3.14.7 (not the 3.11.9 that a
local, untracked setup note describes) and shipped without `pytest`, `fastapi`, `httpx` or
`pydantic`. Installed `pytest`, `fastapi`, `httpx`, `pydantic` into it to run
the suite and the new API tests. Only `fastapi` and `uvicorn` (already present)
are app runtime deps and are pinned in `requirements.txt`; `pytest`/`httpx` are
test-only and, like the pre-existing `pytest`, are not listed there.

httpx is pinned in requirements-dev.txt since 36b6e4a.

**Revisit if.** The reference environment is rebuilt: re-pin `requirements.txt`
to whatever Python/venv is then authoritative, and reconcile that local
"3.11.9" note with reality.

Follow-up (2026-11-04). The setup note named above isn't in this repository. What the repo does pin is Python 3.11: the Dockerfile (`python:3.11-slim`) and CI (`.github/workflows/ci.yml:26`). The local venv is 3.14.7.

---

## 2026-11-04 — optional API key + rate limit on the HTTP API

**What.** `api/main.py` now protects `POST /ask` and `GET /api/config` with two
opt-in checks, added for a planned Render deploy: an API key (header
`X-API-Key` checked against `RAILPULSE_API_KEY`) and a per-client-IP rate limit
(`RAILPULSE_RATE_LIMIT_PER_MINUTE`). `GET /api/health` stays open for the
platform health check. Both config values live in `app/config.py`.

**Why.** Each `/ask` is two LLM calls, so an open public endpoint lets anyone
who finds the URL burn the project's Groq/DeepSeek quota. The key blocks
anonymous traffic; the rate limit caps the damage from a leaked key or an
abusive caller. The read-only DB + `validate_query` already cover the data
side, so cost is the real risk.

**Both off by default** (`RAILPULSE_API_KEY` unset = open, limit 0 = none), so
local dev and the test suite behave exactly as before; you set the two env vars
on Render to turn protection on.

**No new dependency.** The rate limit is a small in-memory sliding window in
`api/main.py` (stdlib only), not `slowapi` — enough for one uvicorn worker.

**What changed in the numbers.** Test suite 148 → 153 passed (+5: health open
with a key set, open without a key, 401 without/with a wrong key, 200 with the
key, 429 past the cap). No existing test changed.

**Revisit if.** The API runs behind more than one worker or instance: the
in-memory window is per-process, so move it to a shared store (Redis) and
reach for `slowapi`. Or if write/stateful endpoints are added: they need auth
too, and probably per-key (not just per-IP) limits. Behind Render's proxy the
limit keys on the proxy IP (`request.client.host`), not the real client, so it
acts as a near-global cap rather than per-caller — key on `X-Forwarded-For`
only with a known trusted-proxy hop count. A wrong-key 401 short-circuits
before the limiter, so invalid-key attempts aren't throttled; fine for cost
(a 401 makes no LLM call) but it allows brute-force.

Follow-up (2026-11-04). `_rate_hits` kept one key per client IP forever. Now every 1,000 accepted calls it drops the clients whose last hit is older than the window; the 429 rule and the "rejected hits aren't recorded" rule are unchanged. Revisit if a deploy sees more than a few thousand distinct IPs a minute: then 1,000 calls between sweeps is too many keys to hold, so sweep on a timer instead. Same day, `require_api_key` compares the key as utf-8 bytes: `hmac.compare_digest` raises on a non-ASCII str, and Starlette decodes headers as latin-1, so a raw `X-API-Key: é` was a 500 instead of a 401.

---

## 2026-11-04 — Offline red-team suite, plus an opt-in live probe

**What.** `tests/test_red_team.py` (46 tests, commit 0e404e1) attacks the
guardrails directly: SQL injection through `validate_query` and the read-only
connection as a second layer, prompt-leak detection with
`looks_like_leaked_meta`, the `ungrounded_figures` check, and an end-to-end
engine probe with a mocked model. `scripts/red_team.py` runs the same ideas
against the real LLM and DB; it needs both `--run-live` and
`RAILPULSE_LIVE_REDTEAM=1`, otherwise it prints how to run and exits. No
product code changed.

**Why.** The suite already proved the happy path. Nothing proved the
deterministic defenses hold when the input is hostile. The live probe exists
because model output varies: it's a thing you read, not a gate, and it spends
quota, so it's gated twice and not collected by pytest.

**What changed in the numbers.** No data figure moves. Tests 102 → 148.

**Known limit.** `looks_like_leaked_meta` doesn't catch a long verbatim dump of
the text-to-SQL system prompt; a test pins that boundary. The real defenses are
the prompt-level instruction and the fact that the consultant call never sees
the text-to-SQL prompt, only its own system prompt plus the question, SQL and
rows.

**Revisit if.** The heuristic is strengthened (the pinned test will move), or a
live probe ever shows an invented figure getting past `ungrounded_figures`.

---

## 2026-11-04 — PySpark mirror of the snapshot-to-event collapse

**What.** `scripts/spark_aggregate.py` (commit 90f6133) reproduces
`build_database.py::keep_latest_snapshot` with the Spark DataFrame API, in
local mode: one row per `(vehicle_id, station_id, Scheduled Date)`, latest
`pulled_at` wins, day-first timestamp parse. `pyspark` is an optional extra in
`requirements-spark.txt` and needs Java; `tests/test_spark_aggregate.py` skips
when either is missing. The pandas pipeline is untouched and stays the
reference.

**Why.** Show the same 980,868 → 99,014 reduction in Spark, next to the
pandas one, without making the app depend on a JVM.

**What changed in the numbers.** No data figure moves; the shipped database is
still built by pandas. One rule differs: pandas keeps an unparseable
`pulled_at` as the latest poll (NaT sorts last, `keep="last"` takes it); the
Spark version never does (`desc_nulls_last`). On the 99,014 rows shipped it
makes no difference, every kept row has a `dd/MM/yyyy HH:mm` `pulled_at`.
The script was first described as able to run on a cluster; `build_spark`
hard-codes `local[*]`, so the docstrings now say local mode. Spark pins the
tie-break on two polls with the same timestamp (later input row wins); pandas
doesn't guarantee it.

**Revisit if.** An export ever ships rows with a bad `pulled_at`: decide which
rule is right and make the other side match. Or the collapse really has to run
on a cluster: then `build_spark` takes a master URL and the CSV read needs a
schema.


## 2026-11-06: round the consultant input, accept no-break-space thousands

**What.** Two fixes in `app/sql_utils.py`, found while retaking the README
screenshots.

1. `build_consultant_input()` now rounds every float it hands the consultant:
   2 decimals, or 3 significant digits below 1. A minutes average computed in
   SQL isn't a seconds column, so `rows_in_minutes()` never rounded it, and
   `gpt-oss-120b` answered "4.426991150442478 minutes". It now says 4.43.
2. `ungrounded_figures()` reads `21\u202f763` (narrow no-break space) and
   `21\u00a0763` as one number. gpt-oss writes thousands that way, so
   "21 763 delays" was split into 21 and 763 and a correct answer got the
   "figures not found" warning. A plain space still splits, so "between 2 and
   300" stays two numbers.

**What changed in the numbers.** The consultant sees rounded floats, so an
answer can no longer quote more precision than 2 decimals (or 3 significant
digits). The figures themselves don't change. The table and the CSV still show
the raw values. The grounding check now passes answers it wrongly flagged
before; it doesn't pass anything it used to catch (a derived 27.8 % on a
72.2 % on-time rate is still flagged, checked live on 2026-11-06).

**Revisit if.** A question needs more than 2 decimals on a value above 1, or a
model starts using a plain space as a thousands separator.


## 2026-11-08: audit fixes (coordinates, wrong-table columns, grounding check)

**What.** Five fixes, each pinned by a test.

1. `scripts/build_database.py` reads the CSVs with `decimal=","`. The export
   writes `50,8709`, so `latitude`, `longitude`, `stations.latitude` and
   `stations.longitude` were stored as text. `AVG("stations.latitude")` gave
   50.25 (integer parts only) and `"stations.latitude" > 50.85` compared
   strings: 27,930 stop events instead of 46,727. A rebuild into a scratch
   file gives REAL columns, 50.80 and 46,727, and every other value in all
   five tables is identical (same row counts, 0 rows differ outside the
   coordinates, on-time rate still 72.2%). The committed
   `data/railpulse_ai.db` has not been rebuilt yet.
2. `app/db.py` checks each double-quoted name against the columns of the
   tables the query reads, taken from the database with `PRAGMA table_info`.
   `KNOWN_COLUMNS` is one flat list, so `SELECT "train_category", ... FROM
   liveboard_records GROUP BY "train_category"` passed it, and SQLite read
   the unknown name as a string: one row labelled `train_category` with the
   network average (2.01 min over 99,014 rows). It is now blocked. All 15
   few-shot queries in the prompt still run.
3. `ungrounded_figures()` reads a comma followed by one or two digits as a
   decimal comma. French answers write `4,43`; it was split into 4 and 43, so
   the correct figure was flagged and a wrong `4,12` passed as 4 and 12. A
   comma before exactly three digits is still a thousands separator.
4. `ungrounded_figures()` flags subset counts in words ("three of the five").
   The committed weekly brief says "Three of the five peak delays occurred
   on 2026-08-05", a count the model made itself, and it passed.
5. `rows_in_minutes()` renames upper-case seconds columns too. The value of
   `AVG(DELAY_SECONDS)` was converted but the column became
   `minutesAVG(DELAY_SECONDS)`, so 5.0 minutes read as seconds.

**What changed in the numbers.** After a rebuild, coordinate columns are
numbers and comparisons on them change result. Nothing else in the data
moves. The grounding check now passes correct French decimals and catches
two things it missed. The query guard blocks a class of query it let through;
it blocks nothing that used to give a right answer.

**Revisit if.** A legitimate query uses a comma join (`FROM a, b`): only the
first table is read, so a quoted column of `b` is blocked. Or a model writes
three-decimal French figures ("4,437"), which still read as thousands.
