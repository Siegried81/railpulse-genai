# RailPulse AI: Intelligent Transit Insights

I built an on-call Railway Operations Assistant that lets station managers ask natural-language
questions about Belgian rail delays and get SQL-backed answers with tactical recommendations —
powered by DeepSeek's API by default, with Groq's free tier as automatic fallback and a fully
local, free Ollama option.

## Overview

- **Type:** Learning challenge (solo)
- **Duration:** 5 days
- **Stack:** Python, SQLite, Streamlit, LLM via DeepSeek (paid API), Groq or OpenRouter (free tiers), or Ollama (local)
- **Data source:** exported from my RailPulse Power BI dashboard (Sprint 3), itself fed by my Azure ingestion pipeline (Sprint 2)

## Features

- **Text-to-SQL**: I translate natural language questions into SQL and execute them against the database.
- **RailPulse Consultant**: I reframe results as short, tactical operational recommendations, not raw data dumps.
- **Safety guardrails**: I only allow `SELECT` statements; destructive keywords (`DROP`, `DELETE`, `UPDATE`, etc.) are blocked before execution, and a column whitelist catches hallucinated column names.
- **Automatic unit conversion**: delays are stored in seconds; I convert them to minutes for all human-facing output.
- **Provider-agnostic LLM client**: I can switch between Ollama (local), Groq, OpenRouter (hosted free tiers) or DeepSeek with a single environment variable — no code changes — or use `auto` (DeepSeek, then Groq with key rotation). Includes caching, stop-sequence tuning, truncation detection, and clean error messages if a backend is unreachable.
- **Chat UI**: sidebar with one-click example questions, SQL and CSV export on every answer, and an SNCB-inspired navy/grey theme.
- **Automated weekly executive brief**: `scripts/generate_weekly_report.py` pulls the week's top delay anomalies, on-time rate, and worst-performing station straight from the database, then has the LLM write a grounded Markdown report to `reports/`. Every figure in the report — and in every consultant answer — is checked against the numbers the LLM was given; an answer with an invented or derived figure is regenerated, and flagged if it still fails.

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd railpulse-genai-challenge
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
```

### 2. Build the database

Place your exported CSVs in `data/`, then run:

```bash
python scripts/build_database.py
```

This creates `data/railpulse_ai.db`.

### 3. Configure your LLM provider

Copy `.env.example` to `.env` and choose a provider:

**Option A — Ollama (fully local, no signup)**
```bash
ollama pull llama3.2:3b
```
```
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:3b
```

**Option B — OpenRouter (hosted free tier, recommended if Groq's signup is unavailable)**

Sign in with Google/GitHub at [openrouter.ai/keys](https://openrouter.ai/keys):
```
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openrouter/free
```
`openrouter/free` is OpenRouter's own auto-router — free model IDs on their platform rotate and get delisted frequently, so pinning a specific one risks a 404 later.

(`LLM_PROVIDER=anthropic` also exists in `app/config.py`, used for fast prototyping only — it is not the open-source path.)

**Option C — Groq (hosted free tier)**
```
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
GROQ_API_KEY_2=optional_second_key
GROQ_MODEL=openai/gpt-oss-120b
```
Each Groq key has its own tokens-per-minute quota (8K on the free tier, and one Text-to-SQL call
is ~2.3K prompt tokens), so extra keys `GROQ_API_KEY_2`, `_3`, … are rotated through on a rate limit.

**Option D — Auto (DeepSeek first, Groq as fallback)**
```
LLM_PROVIDER=auto
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_MODEL=deepseek-flash
GROQ_API_KEY=your_key_here
```
DeepSeek (paid API) answers first; if it fails, is rate-limited, or its answer is cut off, Groq
answers instead. The UI caption under each answer, the weekly brief footer and `query_audit.log`
record which provider and model actually answered.

### 4. Run the app

```bash
python -m streamlit run app/streamlit_app.py
```
(Using `python -m streamlit` rather than the bare `streamlit` command avoids a `ModuleNotFoundError: No module named 'app'` some environments hit.)

## Project Structure

```
├── app/
│   ├── config.py           # provider + DB configuration
│   ├── llm_client.py       # provider-agnostic call_llm() / call_llm_until()
│   ├── db.py               # read-only SQLite connection + safe execution
│   ├── guardrails.py       # SQL safety validation
│   ├── prompts.py          # Text-to-SQL, Consultant and weekly-report prompts
│   ├── sql_utils.py        # SQL extraction + consultant input (delays in minutes)
│   └── streamlit_app.py    # chat UI
├── .streamlit/
│   └── config.toml         # theme colors
├── scripts/
│   ├── build_database.py         # CSV -> SQLite, with polling dedup
│   ├── check_values.py           # prints real column values used to ground the prompts
│   └── generate_weekly_report.py # weekly executive brief
├── tests/                  # offline pytest suite (all LLM calls mocked)
├── test_pipeline.py        # live smoke test against the configured LLM
├── data/
│   └── railpulse_ai.db
└── reports/                # weekly executive brief (nice-to-have)
```

### Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```
The suite runs offline: every provider call is mocked and socket connections are blocked.
`test_pipeline.py` is different — it calls the real LLM and spends free-tier quota.

## Data Schema

I load 5 tables into SQLite from my Power BI export:

**`liveboard_records`** (main table, ~99K rows after dedup) — one row per train stop event
`record_id`, `station_id`, `vehicle_id`, `platform`, `scheduled_time`, `delay_seconds`,
`canceled`, `pulled_at`, `Stations Name`, `stations.standard_name`, `stations.latitude`,
`stations.longitude`, `stations.wheelchair_boarding`, `stations.location_type`,
`vehicles.vehicle_type`, `vehicles.direction`, `Hour`, `day_of_week`, `day_number`,
`Delay Severity`, `Scheduled Date`

**`stations`** — station reference data
`station_id`, `name`, `stations names`, `latitude`, `longitude`, `wheelchair_boarding`, `location_type`

**`routes`** — GTFS static route reference
`route_id`, `route_short_name`, `route_long_name`, `route_desc`, `route_color`,
`route_text_color`, `route_type`, `route_url`, `agency_id`, `train_category`

**`trips`** — GTFS static trip reference
`trip_id`, `route_id`

**`vehicles`** — vehicle reference
`vehicle_id`, `vehicle_type`, `direction`

> `liveboard_records` is already denormalized with station and vehicle info joined in,
> so most questions don't require any JOIN.

## Key Challenges

- **Massive duplicate polling records.** `liveboard_records` is built from continuous GTFS-Realtime
  polling: the same still-upcoming stop gets re-captured on every poll until the train leaves the
  station. This meant 980,868 raw rows for only 99,014 distinct stop events, badly skewing every
  aggregate — a stop polled more often had outsized weight in on-time %, station rankings, and
  delay averages. Fixed in `scripts/build_database.py`: keep only the latest snapshot per stop event
  `(vehicle_id, station_id, Scheduled Date)`, using `pulled_at` to pick the most complete reading.
  A first version keyed on `(vehicle_id, Scheduled Date)` only, which kept a single station per
  train per day and silently dropped ~87% of real stops from every station-level figure.
- **Free-tier LLM instability.** Groq's signup was broken class-wide, so I added OpenRouter as a
  second hosted free-tier option (`app/llm_client.py`, same provider-agnostic pattern). OpenRouter's
  own auto-router (`openrouter/free`) occasionally picked a reasoning model that dumped its internal
  chain-of-thought into the answer instead of a clean response, sometimes eating its whole token
  budget before ever emitting valid SQL. Fixed by passing `reasoning: {exclude: true}` in the API
  call, plus a defensive text-cleanup fallback.
- **Windows `ModuleNotFoundError`.** `streamlit run app/streamlit_app.py` intermittently failed to
  resolve the `app` package on Windows. Fixed by invoking `python -m streamlit run ...` instead.

## Known Limitations

- **"Today" / "this week" / "recently"** are interpreted as the most recent date actually present in
  the dataset (`MAX("Scheduled Date")`), not the real calendar date — this is a fixed historical
  snapshot (2026-07-27 to 2026-08-07), not a live feed, so the true current date would always return
  zero rows.
- **Platform-level data** is not available (empty at ingestion); the assistant answers at station level instead when asked about platforms.
- **Direction-level data** is empty for all records; questions about direction are declined rather than answered with guessed data.
- **Cancellation data** (`canceled`) is 0/False for effectively all records — the source GTFS-Realtime feed doesn't reliably report cancellations, so a "0 canceled trains" answer reflects a feed limitation, not necessarily a perfect on-time record. The assistant caveats this explicitly.
- **Wheelchair accessibility data** is unpopulated for virtually all stations in this feed.

## Author

Siegried Camus — BeCode AI & Data Science bootcamp