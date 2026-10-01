"""
End-to-end check of the Streamlit UI with Streamlit's AppTest.

The LLM and the database are mocked, so this checks what a person actually
sees: the result table shows delays in minutes, never raw seconds.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import db, llm_client

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


@pytest.fixture
def mocked_backend(monkeypatch):
    def fake_llm(system_prompt, user_prompt, is_valid, **kwargs):
        if "SQL generation engine" in system_prompt:
            llm_client.LAST_ANSWERED_BY = "deepseek · deepseek-flash"
            return "SELECT vehicle_id, delay_seconds FROM liveboard_records LIMIT 2;"
        llm_client.LAST_ANSWERED_BY = "groq · openai/gpt-oss-120b"
        return "Train T1 was 10 minutes late at Bordet; worth a look."

    def fake_query(sql):
        return [{"vehicle_id": "T1", "delay_seconds": 600}, {"vehicle_id": "T2", "delay_seconds": 90}]

    monkeypatch.setattr(llm_client, "call_llm_until", fake_llm)
    monkeypatch.setattr(db, "execute_query", fake_query)


def test_result_table_shows_minutes_not_seconds(mocked_backend):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.chat_input[0].set_value("Most delayed trains?").run()

    assert not at.exception
    df = at.dataframe[0].value
    assert list(df.columns) == ["vehicle_id", "delay_minutes"]
    assert list(df["delay_minutes"]) == [10.0, 1.5]
    assert any("converted from seconds to minutes" in c.value for c in at.caption)


def test_answer_names_the_models_that_actually_answered(mocked_backend):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.chat_input[0].set_value("Most delayed trains?").run()

    captions = [c.value for c in at.caption]
    assert "SQL by deepseek · deepseek-flash · answer by groq · openai/gpt-oss-120b" in captions


def test_app_starts_without_repo_root_on_path(tmp_path):
    """`streamlit run app/streamlit_app.py` only puts app/ on sys.path (this is
    how Streamlit Cloud starts it), so the script must find the `app` package
    by itself. Run in a fresh interpreter, outside the repo and without
    PYTHONPATH, so pytest's own `pythonpath = .` cannot hide the problem.
    Loading the page makes no LLM or database call, so no mock is needed."""
    script = (
        "from streamlit.testing.v1 import AppTest\n"
        f"at = AppTest.from_file({APP_PATH!r}, default_timeout=30)\n"
        "at.run()\n"
        "print('EXC', [e.value for e in at.exception])\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "EXC []" in result.stdout, result.stdout
