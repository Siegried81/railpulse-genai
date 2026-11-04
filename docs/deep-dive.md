# Deploy deep dive

The README has the short version. This is how I got RailPulse online, and two things I looked into on the way.

## Render

Render reads the repo, sees the `Dockerfile`, and picks Docker on its own. That image pins Streamlit to port 8501 (`EXPOSE 8501`, and the Streamlit default). Render doesn't care what you `EXPOSE` — it hands the app a port in `$PORT` and waits for something to listen there. So the image builds, nothing answers on `$PORT`, and the health check fails.

The fix is to not use Docker here. I set the language to **Python** and wrote the start command myself:

```
python -m streamlit run app/streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

Build command is `pip install -r requirements.txt`. Nothing else.

Keys go in Render's **Environment** tab, not the repo — `.env` never leaves my machine. Same names as local: `LLM_PROVIDER=groq`, `GROQ_API_KEY`, `GROQ_API_KEY_2…`, `GROQ_MODEL`. Streamlit reads them the same way whether they come from `.env` or the dashboard.

One free-tier catch: the instance sleeps after about 15 minutes idle, and the first request after that takes ~50s to wake it. Render also runs on Python 3.14 here (the wheels come down as `cp314`); everything installed clean.

## Streamlit Community Cloud

Same repo, same `main`. The difference is `LLM_PROVIDER=auto` there, so the client tries Groq first, then DeepSeek, OpenRouter, Ollama, and keeps whichever answers. With a Groq key set that's Groq → `gpt-oss-120b`, same model as Render. Secrets go in the app's own Secrets box, not the repo.

## What RAILPULSE_API_KEY actually does here

Nothing, on this deploy.

The Streamlit app calls `app.engine.answer_question()` directly — a Python call, in-process. It never makes an HTTP request to `/ask`, and it never reads `RAILPULSE_API_KEY`. The start command only runs Streamlit.

`RAILPULSE_API_KEY` guards the separate JSON API in `api/` — the one that answers `/ask` and `/api/config` over HTTP and checks an `X-API-Key` header. That API isn't what's running here. I checked: every path on the live site (`/api/health`, `/api/config`) returns the Streamlit HTML page, not JSON. So the key protects an endpoint that isn't exposed.

Which means the Streamlit UI is open to anyone with the link, and every question spends my Groq quota. Closing that off would be a Streamlit-level password, not this key.

## Why the same average prints different decimals

On Render I get `4.426991150442478 minutes`; locally I've seen `4.42`. Same code, same number — a different model formatting it.

The chain: SQL returns `AVG(delay_seconds)`, `sql_utils.rows_in_minutes()` divides by 60 and passes the full-precision value to the consultant LLM, and the LLM writes the sentence. Nothing rounds it. So `265.6194…` seconds becomes `4.426991150442478` minutes, and the model decides how to show it. Groq's `gpt-oss-120b` prints it whole; another model rounds to `4.42`. On `LLM_PROVIDER=auto` it depends on which provider answered.

The figure is still grounded. `ungrounded_figures()` checks every number in the answer against what the model was given, and the full-precision value is exactly what it was given, so it passes. It's a display inconsistency, not a wrong number. If I want both sides identical, the clean fix is to round inside `rows_in_minutes()` so the model only ever sees two decimals — then the guardrail sees `4.43` too and everything lines up.
