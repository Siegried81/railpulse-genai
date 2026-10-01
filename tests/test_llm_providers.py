"""
Tests for provider selection in app/llm_client.py: the "auto" chain
(DeepSeek, then Groq), Groq key rotation, the gpt-oss payload, and the record
of which provider actually answered. Every HTTP call is mocked.
"""

import pytest

from app import config, llm_client
from app.config import _numbered_keys
from app.llm_client import LLMConnectionError, LLMRateLimitError, call_llm
from tests.test_llm_client import FakeResponse

OK = {"choices": [{"message": {"content": "SELECT 1;"}}]}


@pytest.fixture
def routed_post(monkeypatch):
    """Answer requests.post per host: a list of responses for each of
    'deepseek' and 'groq', consumed in order. Records (host, key, payload)."""
    replies = {"deepseek": [], "groq": []}
    calls = []

    def _post(url, headers=None, json=None, timeout=None):
        host = "deepseek" if "deepseek" in url else "groq"
        calls.append((host, headers["Authorization"].removeprefix("Bearer "), json))
        return replies[host].pop(0)

    monkeypatch.setattr(llm_client.requests, "post", _post)
    return replies, calls


def test_auto_uses_deepseek_first(monkeypatch, routed_post):
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    replies, calls = routed_post
    replies["deepseek"].append(FakeResponse(200, OK))

    assert call_llm("sys", "user") == "SELECT 1;"
    assert [c[0] for c in calls] == ["deepseek"]
    assert llm_client.LAST_ANSWERED_BY == "deepseek · deepseek-flash"


@pytest.mark.parametrize("status", [402, 429, 500])
def test_auto_falls_back_to_groq_when_deepseek_fails(monkeypatch, routed_post, status):
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    replies, calls = routed_post
    replies["deepseek"].append(FakeResponse(status))
    replies["groq"].append(FakeResponse(200, OK))

    assert call_llm("sys", "user") == "SELECT 1;"
    assert [c[0] for c in calls] == ["deepseek", "groq"]
    assert llm_client.LAST_ANSWERED_BY == "groq · openai/gpt-oss-120b"


def test_auto_raises_rate_limit_only_when_every_provider_is_rate_limited(monkeypatch, routed_post):
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    replies, _ = routed_post
    replies["deepseek"].append(FakeResponse(429))
    replies["groq"].append(FakeResponse(429))

    with pytest.raises(LLMRateLimitError):
        call_llm("sys", "user")


def test_auto_raises_connection_error_when_a_failure_is_not_a_rate_limit(monkeypatch, routed_post):
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    replies, _ = routed_post
    replies["deepseek"].append(FakeResponse(402))
    replies["groq"].append(FakeResponse(429))

    with pytest.raises(LLMConnectionError) as exc_info:
        call_llm("sys", "user")
    assert not isinstance(exc_info.value, LLMRateLimitError)
    assert "deepseek" in str(exc_info.value) and "groq" in str(exc_info.value)


def test_groq_rotates_to_next_key_on_rate_limit_and_remembers_it(monkeypatch, routed_post):
    monkeypatch.setattr(config, "GROQ_API_KEYS", ["k1", "k2", "k3"])
    replies, calls = routed_post
    replies["groq"].extend([FakeResponse(429), FakeResponse(200, OK), FakeResponse(200, OK)])

    call_llm("sys", "first")
    call_llm("sys", "second")

    # k1 exhausted -> k2 answers; the next call starts at k2 directly.
    assert [c[1] for c in calls] == ["k1", "k2", "k2"]


def test_groq_skips_an_invalid_key(monkeypatch, routed_post):
    monkeypatch.setattr(config, "GROQ_API_KEYS", ["bad", "good"])
    replies, calls = routed_post
    replies["groq"].extend([FakeResponse(401), FakeResponse(200, OK)])

    assert call_llm("sys", "user") == "SELECT 1;"
    assert [c[1] for c in calls] == ["bad", "good"]


def test_groq_all_keys_rate_limited_raises_rate_limit(monkeypatch, routed_post):
    monkeypatch.setattr(config, "GROQ_API_KEYS", ["k1", "k2"])
    replies, calls = routed_post
    replies["groq"].extend([FakeResponse(429), FakeResponse(429)])

    with pytest.raises(LLMRateLimitError):
        call_llm("sys", "user")
    assert len(calls) == 2


def test_groq_bad_model_is_not_retried_on_other_keys(monkeypatch, routed_post):
    monkeypatch.setattr(config, "GROQ_API_KEYS", ["k1", "k2"])
    replies, calls = routed_post
    replies["groq"].append(FakeResponse(404))

    with pytest.raises(LLMConnectionError, match="GROQ_MODEL"):
        call_llm("sys", "user")
    assert len(calls) == 1


def test_gpt_oss_payload_drops_stop_and_lowers_reasoning(routed_post):
    # A ";" stop ended generation inside gpt-oss's reasoning, leaving an
    # empty answer; the reasoning also eats max_tokens.
    replies, calls = routed_post
    replies["groq"].append(FakeResponse(200, OK))

    call_llm("sys", "user", stop=(";",), max_tokens=300)
    payload = calls[0][2]
    assert "stop" not in payload
    assert payload["reasoning_effort"] == "low"
    assert payload["max_tokens"] == 300


def test_non_reasoning_groq_model_keeps_stop(monkeypatch, routed_post):
    monkeypatch.setattr(config, "GROQ_MODEL", "llama-3.1-8b-instant")
    replies, calls = routed_post
    replies["groq"].append(FakeResponse(200, OK))

    call_llm("sys", "user", stop=(";",))
    assert calls[0][2]["stop"] == [";"]
    assert "reasoning_effort" not in calls[0][2]


def test_deepseek_payload_keeps_stop(monkeypatch, routed_post):
    monkeypatch.setattr(config, "LLM_PROVIDER", "deepseek")
    replies, calls = routed_post
    replies["deepseek"].append(FakeResponse(200, OK))

    call_llm("sys", "user", stop=(";",))
    assert calls[0][2]["stop"] == [";"]
    assert calls[0][2]["model"] == "deepseek-flash"


def test_truncated_answer_is_an_error_not_a_result(routed_post):
    # A brief cut after "Investig" still started with "#" and was published.
    replies, _ = routed_post
    cut = {"choices": [{"message": {"content": "# Brief\n- Investig"}, "finish_reason": "length"}]}
    replies["groq"].append(FakeResponse(200, cut))
    with pytest.raises(LLMConnectionError, match="cut off"):
        call_llm("sys", "user", max_tokens=700)


def test_auto_falls_back_when_deepseek_answer_is_truncated(monkeypatch, routed_post):
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    replies, calls = routed_post
    cut = {"choices": [{"message": {"content": "# Brief"}, "finish_reason": "length"}]}
    replies["deepseek"].append(FakeResponse(200, cut))
    replies["groq"].append(FakeResponse(200, OK))

    assert call_llm("sys", "user") == "SELECT 1;"
    assert llm_client.LAST_ANSWERED_BY.startswith("groq")


def test_empty_content_is_returned_as_empty_string(routed_post):
    replies, _ = routed_post
    replies["groq"].append(FakeResponse(200, {"choices": [{"message": {"content": None}}]}))
    assert call_llm("sys", "user") == ""


def test_numbered_keys_collects_suffixes_and_skips_empties():
    env = {"GROQ_API_KEY": "a", "GROQ_API_KEY_2": "", "GROQ_API_KEY_3": "c", "GROQ_API_KEY_5": "e"}
    # Stops at the first missing suffix (_4), so _5 is not picked up.
    assert _numbered_keys("GROQ_API_KEY", env) == ["a", "c"]
