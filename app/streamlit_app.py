"""
RailPulse AI -- Streamlit chat interface.

Run with:
    streamlit run app/streamlit_app.py
"""

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# `streamlit run app/streamlit_app.py` puts only app/ on sys.path (Streamlit
# Cloud starts it this way), so add the repo root to make the `app` package
# importable without `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm_client import call_llm, call_llm_until, looks_like_leaked_meta, LLMConnectionError
from app.db import execute_query
from app.prompts import (
    CONSULTANT_MAX_TOKENS,
    CONSULTANT_SYSTEM_PROMPT,
    SQL_MAX_TOKENS,
    TEXT_TO_SQL_SYSTEM_PROMPT,
)
from app.sql_utils import extract_sql, build_consultant_input, rows_in_minutes, ungrounded_figures
from app import config, llm_client

st.set_page_config(page_title="RailPulse AI", page_icon="🚆", layout="wide")

# Simple geometric train icon (original design, not the SNCB logo) in two
# color variants: navy-on-light for the white main area, white-on-navy for
# the sidebar (its cutout windows use the sidebar's own navy as "glass").
_TRAIN_ICON_NAVY = """<svg width="{size}" height="{size}" viewBox="0 0 100 90" style="vertical-align:middle">
<rect x="8" y="14" width="84" height="48" rx="12" fill="#0C3B8C"/>
<rect x="16" y="26" width="20" height="18" rx="3" fill="#FFFFFF"/>
<rect x="44" y="26" width="20" height="18" rx="3" fill="#FFFFFF"/>
<rect x="72" y="26" width="12" height="18" rx="3" fill="#FFFFFF"/>
<rect x="2" y="62" width="96" height="6" rx="3" fill="#3D63B0"/>
<circle cx="22" cy="74" r="8" fill="#0C3B8C"/>
<circle cx="78" cy="74" r="8" fill="#0C3B8C"/>
</svg>"""

_TRAIN_ICON_WHITE = """<svg width="{size}" height="{size}" viewBox="0 0 100 90" style="vertical-align:middle">
<rect x="8" y="14" width="84" height="48" rx="12" fill="#FFFFFF"/>
<rect x="16" y="26" width="20" height="18" rx="3" fill="#0C3B8C"/>
<rect x="44" y="26" width="20" height="18" rx="3" fill="#0C3B8C"/>
<rect x="72" y="26" width="12" height="18" rx="3" fill="#0C3B8C"/>
<rect x="2" y="62" width="96" height="6" rx="3" fill="#A9C2EA"/>
<circle cx="22" cy="74" r="8" fill="#FFFFFF"/>
<circle cx="78" cy="74" r="8" fill="#FFFFFF"/>
</svg>"""


def _train_title(size: int, icon_variant: str, text_size: str, text_color: str = "inherit") -> str:
    icon = icon_variant.format(size=size)
    return (
        f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.2rem;">'
        f'{icon}<span style="font-size:{text_size};font-weight:700;color:{text_color};">'
        f"RailPulse AI</span></div>"
    )

def _answered_by(sql_by: str | None, answer_by: str | None) -> str | None:
    """Caption naming the model(s) that actually answered.

    With LLM_PROVIDER="auto" the SQL and the explanation can come from two
    different providers, so both are shown when they differ.
    """
    if not sql_by and not answer_by:
        return None
    if sql_by == answer_by or not sql_by or not answer_by:
        return f"Answered by {sql_by or answer_by}"
    return f"SQL by {sql_by} · answer by {answer_by}"


def _render_sql_and_data(sql: str | None, rows: list[dict] | None, file_index: int, key_suffix: str) -> None:
    """Show the SQL and the result table, each with a download button.

    Shared by the chat history and the fresh answer so both render the same
    way. The table and the CSV go through rows_in_minutes(), the same
    conversion the consultant sees, so a person never reads a delay in
    seconds next to an answer given in minutes. The raw rows kept in
    session_state are not modified.
    """
    if sql:
        with st.expander("🔎 SQL query used"):
            st.code(sql, language="sql")
            st.download_button(
                "⬇️ Download SQL",
                data=sql,
                file_name=f"railpulse_query_{file_index}.sql",
                mime="text/plain",
                key=f"sql_dl_{key_suffix}",
            )
    if rows:
        df = pd.DataFrame(rows_in_minutes(rows))
        with st.expander(f"📊 Raw data ({len(df)} rows)"):
            if list(df.columns) != list(rows[0].keys()):
                st.caption("Delay columns converted from seconds to minutes.")
            st.dataframe(df, use_container_width=True)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                "⬇️ Download CSV",
                data=csv_buffer.getvalue(),
                file_name=f"railpulse_results_{file_index}.csv",
                mime="text/csv",
                key=f"csv_dl_{key_suffix}",
            )


EXAMPLE_QUESTIONS = [
    "Which station had the worst average delay this week?",
    "What percentage of trains were on time overall?",
    "Show me the 10 most delayed trains at Bruxelles-Central.",
    "How does average delay compare between weekdays and weekends?",
    "Which train category has the most delays?",
    "What is the busiest hour for train departures at Liège-Guillemins?",
    "List the top 5 stations by cancellation count.",
]

# --------------------------------------------------------------------------
# Light custom styling -- SNCB-inspired navy/grey accents on top of the
# primaryColor/secondaryBackgroundColor set in .streamlit/config.toml
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        [data-testid="stSidebar"] {
            background-color: #0C3B8C;
            border-right: 1px solid rgba(0,0,0,0.1);
        }
        [data-testid="stSidebar"] * { color: #FFFFFF !important; }
        [data-testid="stSidebar"] .stButton button {
            background-color: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.25);
            color: #FFFFFF;
            text-align: left;
        }
        [data-testid="stSidebar"] .stButton button:hover {
            background-color: rgba(255,255,255,0.2);
            border-color: #FFFFFF;
        }
        .rp-badge {
            display: inline-block;
            padding: 0.15rem 0.6rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.4);
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .rp-welcome {
            padding: 1.5rem;
            border-radius: 0.5rem;
            background: #EEF1F6;
            border-left: 4px solid #0C3B8C;
        }
        h1, h2, h3 { color: #14213D; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "queued_question" not in st.session_state:
    st.session_state.queued_question = None

_MODEL_BY_PROVIDER = {
    "auto": " → ".join(config.AUTO_PROVIDER_ORDER),
    "deepseek": config.DEEPSEEK_MODEL,
    "ollama": config.OLLAMA_MODEL,
    "groq": config.GROQ_MODEL,
    "openrouter": config.OPENROUTER_MODEL,
    "anthropic": config.ANTHROPIC_MODEL,
}
active_model = _MODEL_BY_PROVIDER.get(config.LLM_PROVIDER, "unknown")

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f'<span class="rp-badge">● {config.LLM_PROVIDER} · {active_model}</span>', unsafe_allow_html=True)
    st.markdown(_train_title(32, _TRAIN_ICON_WHITE, "1.6rem", "#FFFFFF"), unsafe_allow_html=True)
    st.caption("On-call Railway Operations Assistant")

    st.divider()

    st.subheader("💡 Try asking")
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if st.button(q, key=f"example_{i}", use_container_width=True):
            st.session_state.queued_question = q
            st.rerun()

    st.divider()

    with st.expander("ℹ️ What can I ask?"):
        st.markdown(
            """
            Ask about Belgian rail delays, cancellations, on-time rates, busiest
            stations, train categories, or accessibility -- station-level detail,
            not real-time.

            **Known limitations:**
            - Platform-level data is unavailable -- questions about a specific
              platform are answered at the station level instead.
            - Direction-level data is unavailable.
            - Wheelchair accessibility data is not meaningfully populated in the
              source feed.
            - Data covers **2026-07-27 to 2026-08-07** only (fixed historical
              snapshot, not real-time).
            """
        )

    st.divider()

    n_questions = sum(1 for m in st.session_state.messages if m["role"] == "user")
    st.caption(f"📊 {n_questions} question(s) asked this session")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.markdown(_train_title(44, _TRAIN_ICON_NAVY, "2.25rem", "#14213D"), unsafe_allow_html=True)
st.caption(
    f"On-call Railway Operations Assistant · running on **{config.LLM_PROVIDER}** ({active_model})"
)

if not st.session_state.messages:
    st.markdown(
        """
        <div class="rp-welcome">
        👋 <b>Welcome.</b> Ask me anything about Belgian rail delays, cancellations,
        on-time rates, busiest stations, or train categories -- or pick a question
        from the sidebar to get started.
        </div>
        """,
        unsafe_allow_html=True,
    )

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("answered_by"):
            st.caption(msg["answered_by"])
        _render_sql_and_data(msg.get("sql"), msg.get("rows"), file_index=idx, key_suffix=str(idx))

# --------------------------------------------------------------------------
# Input handling (chat box OR a sidebar example question)
# --------------------------------------------------------------------------
question = st.chat_input("Ask about delays, cancellations, on-time rate...")

if st.session_state.queued_question:
    question = st.session_state.queued_question
    st.session_state.queued_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # --- Step 1: text-to-SQL ---
        try:
            with st.spinner("Translating your question into SQL..."):
                # stop=(";",) halts generation right after the query, which is
                # both faster (no rambling) and safer (no stray text after the
                # semicolon that could trip the "multiple statements" guardrail).
                # call_llm_until retries (bypassing cache) if the free-tier
                # backend returns something that isn't valid SQL or NO_QUERY --
                # auto-routed free models occasionally leak a stray meta/safety
                # line instead of the actual query even at temperature=0.
                raw_sql = call_llm_until(
                    TEXT_TO_SQL_SYSTEM_PROMPT,
                    question,
                    is_valid=lambda s: extract_sql(s).upper().startswith(("SELECT", "NO_QUERY")),
                    stop=(";",),
                    max_tokens=SQL_MAX_TOKENS,
                )
                sql = extract_sql(raw_sql)
                sql_by = llm_client.LAST_ANSWERED_BY
        except LLMConnectionError as e:
            answer = f"🔌 **Can't reach the LLM backend.** {e}"
            st.error(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sql": None, "rows": None}
            )
            st.stop()

        if sql.upper().startswith("NO_QUERY"):
            reason = sql.split(":", 1)[-1].strip()
            answer = f"I can't answer that with the data I have access to. {reason}"
            st.warning(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sql": None, "rows": None}
            )
        else:
            try:
                with st.spinner("Querying the database..."):
                    rows = execute_query(sql)
            except ValueError as e:
                answer = f"⛔ That query was blocked by safety guardrails: {e}"
                st.error(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sql": sql, "rows": None}
                )
            else:
                # --- Step 2: consultant explanation ---
                try:
                    with st.spinner("Preparing your answer..."):
                        consultant_input = build_consultant_input(question, sql, rows)
                        answer = call_llm_until(
                            CONSULTANT_SYSTEM_PROMPT,
                            consultant_input,
                            is_valid=lambda s: not looks_like_leaked_meta(s)
                            and not ungrounded_figures(s, consultant_input),
                            max_tokens=CONSULTANT_MAX_TOKENS,
                        )
                        # call_llm_until() returns its last attempt even if
                        # still invalid; say so rather than hide it.
                        ungrounded = ungrounded_figures(answer, consultant_input)
                except LLMConnectionError as e:
                    answer = f"🔌 **Can't reach the LLM backend.** {e}"
                    st.error(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sql": sql, "rows": rows}
                    )
                    st.stop()

                if ungrounded:
                    answer += (
                        "\n\n⚠️ *Check against the raw data: this answer contains figures not found "
                        f"in the query results ({', '.join(ungrounded)}).*"
                    )
                st.markdown(answer)
                answered_by = _answered_by(sql_by, llm_client.LAST_ANSWERED_BY)
                if answered_by:
                    st.caption(answered_by)

                n = len(st.session_state.messages)
                _render_sql_and_data(sql, rows, file_index=n, key_suffix=f"new_{n}")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sql": sql, "rows": rows, "answered_by": answered_by}
                )