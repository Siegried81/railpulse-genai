"""JSON API over the RailPulse AI text-to-SQL + consultant pipeline.

Why it exists: the Streamlit chat app is the interactive surface; this gives the
same pipeline a plain HTTP surface so other services (or the other repos' React
UIs, which expect a `GET /api/health` and a `POST /ask`) can ask a question and
get back the answer, the SQL, the rows and who answered. It is additive: it does
not change how the Streamlit app behaves.

Both surfaces call the one function `app.engine.answer_question`, so they cannot
drift. Every safety property of the pipeline is kept here by construction, since
nothing in this file re-implements it: one structured LLM call per step and no
agent loop, the read-only DB plus `validate_query` guardrails inside
`execute_query`, the `ungrounded_figures` check on the answer (returned as
`ungrounded`), and `LAST_ANSWERED_BY` recorded as `sql_by` / `answer_by`.

Outcome mapping:
- a normal answer, and a NO_QUERY verdict (the model declining because the data
  can't answer the question), are both a 200 with an explanation; NO_QUERY is
  flagged with `no_query: true` and null `sql`/`rows`, not treated as an error;
- the LLM backend being unreachable -> 503;
- the query being blocked by the guardrails or exceeding the time budget
  (a ValueError from execute_query) -> 422.

Run with:
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8003
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import config
from app.engine import answer_question
from app.llm_client import LLMConnectionError

app = FastAPI(title="RailPulse AI API", version="1.0")


def _active_model() -> str:
    """Model name for the configured provider (never a key).

    With "auto" it is the fallback order, matching the badge the Streamlit
    sidebar shows.
    """
    return {
        "auto": " → ".join(config.AUTO_PROVIDER_ORDER),
        "deepseek": config.DEEPSEEK_MODEL,
        "ollama": config.OLLAMA_MODEL,
        "groq": config.GROQ_MODEL,
        "openrouter": config.OPENROUTER_MODEL,
        "anthropic": config.ANTHROPIC_MODEL,
    }.get(config.LLM_PROVIDER, "unknown")


class AskRequest(BaseModel):
    """Body of POST /ask: just the question.

    The 500-character cap keeps a pathological prompt from being sent to the
    model; the Streamlit chat box has no formal cap but a real question is far
    shorter. min_length 1 rejects an empty body.
    """

    question: str = Field(min_length=1, max_length=500)


@app.get("/api/health")
def health() -> dict:
    """Liveness probe; touches no model and no database, so config can't fail it."""
    return {"status": "ok"}


@app.get("/api/config")
def get_config() -> dict:
    """Provider and model the API is running on, for a client badge. No keys."""
    return {"llm_provider": config.LLM_PROVIDER, "llm_model": _active_model()}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    """Answer one question through the shared pipeline.

    Returns the same fields `answer_question` produces, plus `no_query` so a
    client can tell a declined question (sql/rows null, a normal 200) from an
    answered one without inspecting the text. The answer is returned raw with
    its `ungrounded` figures listed separately, exactly as the Streamlit page
    receives it, so a client can surface the same "check against the raw data"
    caveat.
    """
    try:
        result = answer_question(req.question)
    except LLMConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        # Guardrail block or the query exceeding its time budget.
        raise HTTPException(status_code=422, detail=str(e)) from e

    return {
        "question": req.question,
        "no_query": result["sql"] is None,
        "answer": result["answer"],
        "sql": result["sql"],
        "rows": result["rows"],
        "sql_by": result["sql_by"],
        "answer_by": result["answer_by"],
        "ungrounded": result["ungrounded"],
    }
