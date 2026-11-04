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

from app.llm_client import LLMConnectionError
from app.engine import answer_question
from app.sql_utils import rows_in_minutes
from app import config

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
        # The whole pipeline lives in app.engine.answer_question so the UI and
        # the JSON API share one path and cannot drift. The branches below only
        # decide how each outcome is shown; the behaviour (SQL, query, answer,
        # grounding check, NO_QUERY handling) is identical to before.
        try:
            with st.spinner("Working on your answer..."):
                result = answer_question(question)
        except LLMConnectionError as e:
            # Raised after the SQL step carries the SQL and rows it computed
            # before the answer step failed, so they still show in the history.
            answer = f"🔌 **Can't reach the LLM backend.** {e}"
            st.error(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer,
                 "sql": getattr(e, "sql", None), "rows": getattr(e, "rows", None)}
            )
            st.stop()
        except ValueError as e:
            answer = f"⛔ That query was blocked by safety guardrails: {e}"
            st.error(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sql": getattr(e, "sql", None), "rows": None}
            )
        else:
            answer = result["answer"]
            if result["sql"] is None:
                # NO_QUERY verdict: a normal outcome, shown as a notice.
                st.warning(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sql": None, "rows": None}
                )
            else:
                if result["ungrounded"]:
                    answer += (
                        "\n\n⚠️ *Check against the raw data: this answer contains figures not found "
                        f"in the query results ({', '.join(result['ungrounded'])}).*"
                    )
                st.markdown(answer)
                answered_by = _answered_by(result["sql_by"], result["answer_by"])
                if answered_by:
                    st.caption(answered_by)

                n = len(st.session_state.messages)
                _render_sql_and_data(result["sql"], result["rows"], file_index=n, key_suffix=f"new_{n}")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sql": result["sql"],
                     "rows": result["rows"], "answered_by": answered_by}
                )