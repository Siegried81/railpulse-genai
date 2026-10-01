"""
Tests for app/llm_client.py: rate-limit retry and reasoning-leak stripping.

Every provider call is mocked at requests.post; nothing reaches the network.
"""

import pytest
import requests

from app import config, llm_client
from app.llm_client import LLMConnectionError, LLMRateLimitError, call_llm_until


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} Client Error")

    def json(self):
        return self._payload


OK_PAYLOAD_BY_PROVIDER = {
    "groq": {"choices": [{"message": {"content": "SELECT 1;"}}]},
    "openrouter": {"choices": [{"message": {"content": "SELECT 1;"}}]},
    "anthropic": {"content": [{"text": "SELECT 1;"}]},
}


@pytest.fixture
def fake_post(monkeypatch):
    """Queue of responses returned by successive requests.post calls."""
    queue = []
    calls = []

    def _post(*args, **kwargs):
        calls.append(kwargs)
        return queue.pop(0)

    monkeypatch.setattr(llm_client.requests, "post", _post)
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    return queue, calls


@pytest.mark.parametrize("provider", ["groq", "openrouter", "anthropic"])
def test_429_raises_rate_limit_error(monkeypatch, fake_post, provider):
    monkeypatch.setattr(config, "LLM_PROVIDER", provider)
    queue, _ = fake_post
    queue.append(FakeResponse(429))
    with pytest.raises(LLMRateLimitError):
        llm_client.call_llm("sys", "user")


@pytest.mark.parametrize("provider", ["groq", "openrouter", "anthropic"])
def test_call_llm_until_retries_after_rate_limit(monkeypatch, fake_post, provider):
    monkeypatch.setattr(config, "LLM_PROVIDER", provider)
    queue, calls = fake_post
    queue.extend([FakeResponse(429), FakeResponse(200, OK_PAYLOAD_BY_PROVIDER[provider])])

    result = call_llm_until("sys", "user", is_valid=lambda s: s.startswith("SELECT"))

    assert result == "SELECT 1;"
    assert len(calls) == 2


def test_persistent_rate_limit_is_raised_after_max_attempts(monkeypatch, fake_post):
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    queue, calls = fake_post
    queue.extend([FakeResponse(429)] * 3)

    with pytest.raises(LLMRateLimitError):
        call_llm_until("sys", "user", is_valid=lambda s: True, max_attempts=3)
    assert len(calls) == 3


def test_rate_limit_error_is_still_a_connection_error():
    # Callers (Streamlit UI, CLI harness) only catch LLMConnectionError.
    assert issubclass(LLMRateLimitError, LLMConnectionError)


def test_bad_key_is_not_retried(monkeypatch, fake_post):
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    queue, calls = fake_post
    queue.append(FakeResponse(401))

    with pytest.raises(LLMConnectionError) as exc_info:
        call_llm_until("sys", "user", is_valid=lambda s: True)
    assert not isinstance(exc_info.value, LLMRateLimitError)
    assert len(calls) == 1


def test_strip_removes_leaked_planning_preamble():
    text = "I need to:\n1. Find the station\n2. Compute the average\nBruxelles-Central averages 3.2 minutes."
    assert llm_client._strip_reasoning_leak(text) == "Bruxelles-Central averages 3.2 minutes."


def test_strip_removes_think_tags():
    text = "<think>scratch work</think>SELECT 1;"
    assert llm_client._strip_reasoning_leak(text) == "SELECT 1;"


def test_strip_keeps_numbered_list_in_weekly_report():
    report = (
        "# RailPulse Weekly Operations Brief\n2026-08-01 to 2026-08-07\n\n"
        "## Recommendations\n1. Review Jurbise.\n2. Monitor Charleroi-Central."
    )
    assert llm_client._strip_reasoning_leak(report) == report


def test_strip_keeps_numbered_list_in_consultant_answer():
    answer = "The three most delayed stations are:\n1. Gouvy\n2. Jurbise\n3. Charleroi-Central"
    assert llm_client._strip_reasoning_leak(answer) == answer
