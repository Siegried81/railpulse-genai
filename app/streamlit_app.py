"""
RailPulse AI -- Streamlit chat interface.

Run with:
    streamlit run app/streamlit_app.py
"""

import io

import pandas as pd
import streamlit as st

from app.llm_client import (
    call_llm,
    call_llm_until,
    validate_consultant_numbers,
    validate_no_false_generalization,
    validate_no_invented_entities,
    validate_superlative_sql_direction,
    validate_no_self_referential_case_filter,
    validate_no_unreliable_wheelchair_query,
    validate_no_raw_category_delay_count_ranking,
    validate_on_time_uses_delay_severity,
    validate_station_ranking_has_min_sample,
    validate_no_query_reason_matches_question,
    force_reason_mismatch_worst_best_station,
    validate_station_query_matches_question_stations,
    force_no_query_if_unreliable_wheelchair,
    force_no_query_if_self_referential_case,
    force_no_query_if_raw_category_delay_count_ranking,
    force_no_query_if_on_time_wrong_definition,
    force_no_query_if_station_ranking_unreliable,
    correct_consultant_numbers,
    correct_consultant_generalization,
    correct_no_invented_entities,
    auto_fix_superlative_claims,
    auto_fix_misdirected_recommendation,
    strip_ungrounded_time_reference,
    strip_invented_cause,
    strip_recommendation_after_unavailability_disclaimer,
    replace_vague_period_with_dates,
    LLMConnectionError,
)
from app.db import execute_query
from app.prompts import TEXT_TO_SQL_SYSTEM_PROMPT, CONSULTANT_SYSTEM_PROMPT
from app.sql_utils import extract_sql
from app import config

st.set_page_config(page_title="RailPulse AI", page_icon="🚆", layout="wide")

EXAMPLE_QUESTIONS = [
    "Which station had the worst average delay this week?",
    "What percentage of trains were on time overall?",
    "How does average delay compare between weekdays and weekends?",
    "Which train category has the most delays?",
    "What is the busiest hour for train departures at Liège-Guillemins?",]

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
    "ollama": config.OLLAMA_MODEL,
    "groq": config.GROQ_MODEL,
}
active_model = _MODEL_BY_PROVIDER.get(config.LLM_PROVIDER, "unknown")


@st.cache_data(ttl=3600)
def _get_data_date_range() -> tuple[str, str] | None:
    """Fetch the actual min/max Scheduled Date from the data, so the date
    range shown in the UI can never drift out of sync with what's really in
    the database. Cached for an hour since this is a fixed historical
    snapshot that doesn't change during a session. Returns None if the
    query fails for any reason, so the UI can degrade gracefully.
    """
    try:
        rows = execute_query(
            'SELECT MIN("Scheduled Date") AS min_date, MAX("Scheduled Date") AS max_date '
            "FROM liveboard_records;"
        )
        if rows and rows[0].get("min_date") and rows[0].get("max_date"):
            return rows[0]["min_date"], rows[0]["max_date"]
    except Exception as e:
        st.error(f"DEBUG date range fetch failed: {e}")
    return None


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f'<span class="rp-badge">● {config.LLM_PROVIDER} · {active_model}</span>', unsafe_allow_html=True)
    st.title("🚆 RailPulse AI")
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
            Ask about Belgian rail delays, on-time rates, busiest
            stations, train categories, or accessibility -- station-level detail,
            not real-time.

            **Known limitations:**
            - Platform-level data is unavailable -- questions about a specific
              platform are answered at the station level instead.
            - Direction-level data is unavailable.
            - Wheelchair accessibility data is not meaningfully populated in the
              source feed.
            - Cancellation data shows zero canceled trains in this dataset --
              the "canceled" flag is always 0 at the source, so cancellation
              questions won't return meaningful results.
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
st.title("🚆 RailPulse AI")
st.caption(
    f"On-call Railway Operations Assistant · running on **{config.LLM_PROVIDER}** ({active_model})"
)

_date_range = _get_data_date_range()
if _date_range:
    st.markdown(
        f'<span class="rp-badge" style="color:#14213D !important; '
        f'background:#EEF1F6; border-color:#0C3B8C;">'
        f"📅 Data covers {_date_range[0]} to {_date_range[1]} (fixed historical snapshot)</span>",
        unsafe_allow_html=True,
    )

if not st.session_state.messages:
    st.markdown(
        """
        <div class="rp-welcome">
        👋 <b>Welcome.</b> Ask me anything about Belgian rail delays,
        on-time rates, busiest stations, or train categories -- or pick a question
        from the sidebar to get started.
        </div>
        """,
        unsafe_allow_html=True,
    )

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sql"):
            with st.expander("🔎 SQL query used"):
                st.code(msg["sql"], language="sql")
                st.download_button(
                    "⬇️ Download SQL",
                    data=msg["sql"],
                    file_name=f"railpulse_query_{idx}.sql",
                    mime="text/plain",
                    key=f"sql_dl_{idx}",
                )
        if msg.get("rows"):
            df = pd.DataFrame(msg["rows"])
            with st.expander(f"📊 Raw data ({len(df)} rows)"):
                st.dataframe(df, use_container_width=True)
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    "⬇️ Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"railpulse_results_{idx}.csv",
                    mime="text/csv",
                    key=f"csv_dl_{idx}",
                )

# --------------------------------------------------------------------------
# Input handling (chat box OR a sidebar example question)
# --------------------------------------------------------------------------
question = st.chat_input("Ask about delays, on-time rate, busiest stations...")

if st.session_state.queued_question:
    question = st.session_state.queued_question
    st.session_state.queued_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        sql = None
        rows = None
        answer = None
        # outcome tracks which branch fired so the display code after the
        # status widget can reproduce the exact same messages/keys as before,
        # instead of nesting everything inside the status block (where
        # st.error/st.warning would end up hidden once the status collapses).
        outcome = None  # "llm_error" | "no_query" | "blocked" | "llm_error_step2" | "ok"

        with st.status("Working on your question...", expanded=False) as status:
            # --- Step 1: text-to-SQL ---
            status.update(label="Translating your question into SQL...")
            try:
                # stop=(";",) halts generation right after the query, which is
                # both faster (no rambling) and safer (no stray text after the
                # semicolon that could trip the "multiple statements" guardrail)
                raw_sql = call_llm_until(
                    TEXT_TO_SQL_SYSTEM_PROMPT,
                    question,
                    stop=(";",),
                    max_tokens=300,
                    validate=lambda text: validate_superlative_sql_direction(question, text)
                    and validate_no_self_referential_case_filter(text)
                    and validate_no_unreliable_wheelchair_query(text)
                    and validate_no_raw_category_delay_count_ranking(text)
                    and validate_on_time_uses_delay_severity(question, text)
                    and validate_station_ranking_has_min_sample(text)
                    and validate_no_query_reason_matches_question(question, text)
                    and validate_station_query_matches_question_stations(question, text),
                )
                sql = extract_sql(raw_sql)
                # Deterministic safety net: call_llm_until gives up after
                # max_attempts and returns the last invalid result rather
                # than raising, so a weak model can still fail these checks
                # on every retry. Force a safe NO_QUERY fallback rather than
                # let a tautological or unreliable-column query execute.
                sql = force_no_query_if_unreliable_wheelchair(sql)
                sql = force_no_query_if_self_referential_case(sql)
                sql = force_no_query_if_raw_category_delay_count_ranking(sql)
                sql = force_no_query_if_on_time_wrong_definition(question, sql)
                sql = force_no_query_if_station_ranking_unreliable(sql)
                sql = force_reason_mismatch_worst_best_station(question, sql)
            except LLMConnectionError as e:
                answer = f"🔌 **Can't reach the LLM backend.** {e}"
                outcome = "llm_error"
                status.update(label="Connection error", state="error")

            if outcome is None and sql.upper().startswith("NO_QUERY"):
                reason = sql.split(":", 1)[-1].strip()
                answer = f"I can't answer that with the data I have access to. {reason}"
                outcome = "no_query"
                sql = None
                status.update(label="No matching data for this question", state="complete")

            # --- Step 2: run the query ---
            if outcome is None:
                status.update(label="Querying the database...")
                try:
                    rows = execute_query(sql)
                except ValueError as e:
                    answer = f"⛔ That query was blocked by safety guardrails: {e}"
                    outcome = "blocked"
                    status.update(label="Query blocked", state="error")

            # --- Step 3: consultant explanation ---
            if outcome is None:
                status.update(label="Preparing your answer...")
                try:
                    consultant_input = (
                        f"Question: {question}\n"
                        f"SQL executed: {sql}\n"
                        f"Results: {rows[:20]}"
                    )
                    answer = call_llm_until(
                        CONSULTANT_SYSTEM_PROMPT,
                        consultant_input,
                        max_tokens=150,
                        validate=lambda text: validate_consultant_numbers(text, rows)
                        and validate_no_false_generalization(text, rows)
                        and validate_no_invented_entities(text, rows),
                    )
                    answer = correct_consultant_numbers(answer, rows)
                    answer = correct_consultant_generalization(answer, rows)
                    answer = correct_no_invented_entities(answer, rows)
                    answer = auto_fix_superlative_claims(answer, rows)
                    answer = auto_fix_misdirected_recommendation(answer, rows)
                    answer = strip_ungrounded_time_reference(answer, sql)
                    answer = strip_invented_cause(answer)
                    answer = strip_recommendation_after_unavailability_disclaimer(answer)
                    answer = replace_vague_period_with_dates(answer, sql)
                    outcome = "ok"
                    status.update(label="Done", state="complete")
                except LLMConnectionError as e:
                    answer = f"🔌 **Can't reach the LLM backend.** {e}"
                    outcome = "llm_error_step2"
                    status.update(label="Connection error", state="error")

        # --- Display, outside the status widget so errors/warnings stay
        # visible in the chat instead of being tucked inside a collapsed
        # status container. Each branch mirrors the original code exactly. ---
        if outcome == "llm_error":
            st.error(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sql": None, "rows": None}
            )
            st.stop()

        elif outcome == "no_query":
            st.warning(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sql": None, "rows": None}
            )

        elif outcome == "blocked":
            st.error(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sql": sql, "rows": None}
            )

        elif outcome == "llm_error_step2":
            st.error(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sql": sql, "rows": rows}
            )
            st.stop()

        elif outcome == "ok":
            st.markdown(answer)

            with st.expander("🔎 SQL query used"):
                st.code(sql, language="sql")
                st.download_button(
                    "⬇️ Download SQL",
                    data=sql,
                    file_name=f"railpulse_query_{len(st.session_state.messages)}.sql",
                    mime="text/plain",
                    key=f"sql_dl_new_{len(st.session_state.messages)}",
                )

            if rows:
                df = pd.DataFrame(rows)
                with st.expander(f"📊 Raw data ({len(df)} rows)"):
                    st.dataframe(df, use_container_width=True)
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "⬇️ Download CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"railpulse_results_{len(st.session_state.messages)}.csv",
                        mime="text/csv",
                        key=f"csv_dl_new_{len(st.session_state.messages)}",
                    )

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sql": sql, "rows": rows}
            )