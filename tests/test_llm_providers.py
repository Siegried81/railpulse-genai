"""
Tests for provider selection in app/llm_client.py: the "auto" chain
(Groq first, then DeepSeek), Groq key rotation, the gpt-oss payload, and the
record of which provider actually answered. Every HTTP call is mocked.
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


def test_auto_provider_order_default():
    # The free Groq tier leads, the SambaNova free tier comes next so a Groq
    # rate limit does not spend paid DeepSeek credit, then DeepSeek, the
    # OpenRouter free tier and a local Ollama. Pinned here because the
    # behaviour tests below run on a subset (only some hosts are mocked).
    assert config.AUTO_PROVIDER_ORDER == ("groq", "sambanova", "deepseek", "openrouter", "ollama")


def test_auto_uses_groq_first(monkeypatch, routed_post):
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    replies, calls = routed_post
    replies["groq"].append(FakeResponse(200, OK))

    assert call_llm("sys", "user") == "SELECT 1;"
    assert [c[0] for c in calls] == ["groq"]
    assert llm_client.LAST_ANSWERED_BY == "groq · openai/gpt-oss-120b"


@pytest.mark.parametrize("status", [402, 429, 500])
def test_auto_falls_back_to_deepseek_when_groq_fails(monkeypatch, routed_post, status):
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    replies, calls = routed_post
    replies["groq"].append(FakeResponse(status))
    replies["deepseek"].append(FakeResponse(200, OK))

    assert call_llm("sys", "user") == "SELECT 1;"
    assert [c[0] for c in calls] == ["groq", "deepseek"]
    assert llm_client.LAST_ANSWERED_BY == "deepseek · deepseek-flash"


def test_auto_raises_rate_limit_only_when_every_provider_is_rate_limited(monkeypatch, routed_post):
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(config, "AUTO_PROVIDER_ORDER", ("groq", "deepseek"))
    replies, _ = routed_post
    replies["deepseek"].append(FakeResponse(429))
    replies["groq"].append(FakeResponse(429))

    with pytest.raises(LLMRateLimitError):
        call_llm("sys", "user")


def test_auto_raises_rate_limit_when_one_provider_is_rate_limited_and_others_cannot_answer(
    monkeypatch, routed_post
):
    # Only Groq is configured in practice: Groq is rate-limited and DeepSeek
    # has no credit. Waiting can free Groq, so this must be a rate limit and
    # call_llm_until must back off, not give up with a plain error.
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(config, "AUTO_PROVIDER_ORDER", ("groq", "deepseek"))
    replies, _ = routed_post
    replies["deepseek"].append(FakeResponse(402))
    replies["groq"].append(FakeResponse(429))

    with pytest.raises(LLMRateLimitError) as exc_info:
        call_llm("sys", "user")
    assert "deepseek" in str(exc_info.value) and "groq" in str(exc_info.value)


def test_auto_raises_connection_error_when_no_failure_is_a_rate_limit(monkeypatch, routed_post):
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(config, "AUTO_PROVIDER_ORDER", ("groq", "deepseek"))
    replies, _ = routed_post
    replies["deepseek"].append(FakeResponse(402))
    replies["groq"].append(FakeResponse(500))

    with pytest.raises(LLMConnectionError) as exc_info:
        call_llm("sys", "user")
    assert not isinstance(exc_info.value, LLMRateLimitError)


def test_auto_retries_after_backoff_when_only_groq_is_rate_limited(monkeypatch, routed_post):
    # End to end through call_llm_until: the 429 + 402 round is retried and
    # the second round answers.
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(config, "AUTO_PROVIDER_ORDER", ("groq", "deepseek"))
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    replies, _ = routed_post
    replies["groq"].extend([FakeResponse(429), FakeResponse(200, OK)])
    replies["deepseek"].append(FakeResponse(402))

    answer = llm_client.call_llm_until("sys", "user", is_valid=lambda s: True)
    assert answer == "SELECT 1;"
    assert answer.answered_by == "groq · openai/gpt-oss-120b"


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


def test_auto_falls_back_when_groq_answer_is_truncated(monkeypatch, routed_post):
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    replies, calls = routed_post
    cut = {"choices": [{"message": {"content": "# Brief"}, "finish_reason": "length"}]}
    replies["groq"].append(FakeResponse(200, cut))
    replies["deepseek"].append(FakeResponse(200, OK))

    assert call_llm("sys", "user") == "SELECT 1;"
    assert llm_client.LAST_ANSWERED_BY.startswith("deepseek")


def test_empty_content_is_returned_as_empty_string(routed_post):
    replies, _ = routed_post
    replies["groq"].append(FakeResponse(200, {"choices": [{"message": {"content": None}}]}))
    assert call_llm("sys", "user") == ""


def test_numbered_keys_collects_suffixes_and_skips_empties():
    env = {"GROQ_API_KEY": "a", "GROQ_API_KEY_2": "", "GROQ_API_KEY_3": "c", "GROQ_API_KEY_5": "e"}
    # Stops at the first missing suffix (_4), so _5 is not picked up.
    assert _numbered_keys("GROQ_API_KEY", env) == ["a", "c"]


# --- Every provider, routed by host ------------------------------------------


@pytest.fixture
def host_post(monkeypatch):
    """Answer requests.post by host substring; records (host, url, headers, payload)."""
    hosts = ("sambanova", "deepseek", "groq", "openrouter", "anthropic", "localhost")
    replies = {h: [] for h in hosts}
    calls = []

    def _post(url, headers=None, json=None, timeout=None):
        host = next(h for h in hosts if h in url)
        calls.append((host, url, headers, json))
        return replies[host].pop(0)

    monkeypatch.setattr(llm_client.requests, "post", _post)
    return replies, calls


def test_sambanova_sends_an_openai_compatible_request(monkeypatch, host_post):
    monkeypatch.setattr(config, "LLM_PROVIDER", "sambanova")
    monkeypatch.setattr(config, "SAMBANOVA_API_KEY", "test-samba")
    replies, calls = host_post
    replies["sambanova"].append(FakeResponse(200, OK))

    answer = call_llm("sys", "user", stop=(";",), max_tokens=300)
    assert answer == "SELECT 1;"
    assert answer.answered_by == "sambanova · test-sambanova-model"
    _, url, headers, payload = calls[0]
    assert url == config.SAMBANOVA_URL
    assert headers["Authorization"] == "Bearer test-samba"
    assert payload["model"] == "test-sambanova-model"
    assert payload["temperature"] == 0
    assert payload["stop"] == [";"] and payload["max_tokens"] == 300


def test_sambanova_without_a_key_makes_no_http_call(monkeypatch, host_post):
    monkeypatch.setattr(config, "LLM_PROVIDER", "sambanova")
    _, calls = host_post
    with pytest.raises(LLMConnectionError, match="SAMBANOVA_API_KEY"):
        call_llm("sys", "user")
    assert calls == []


def test_sambanova_429_is_a_rate_limit(monkeypatch, host_post):
    monkeypatch.setattr(config, "LLM_PROVIDER", "sambanova")
    monkeypatch.setattr(config, "SAMBANOVA_API_KEY", "test-samba")
    replies, _ = host_post
    replies["sambanova"].append(FakeResponse(429))
    with pytest.raises(LLMRateLimitError):
        call_llm("sys", "user")


def test_auto_tries_sambanova_right_after_groq(monkeypatch, host_post):
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(config, "SAMBANOVA_API_KEY", "test-samba")
    replies, calls = host_post
    replies["groq"].append(FakeResponse(429))
    replies["sambanova"].append(FakeResponse(200, OK))

    answer = call_llm("sys", "user")
    assert [c[0] for c in calls] == ["groq", "sambanova"]
    assert answer.answered_by.startswith("sambanova")


# --- Cut-off and malformed answers are errors on every provider ---------------


@pytest.mark.parametrize(
    "provider, host, cut_body",
    [
        ("ollama", "localhost", {"message": {"content": "# Brief"}, "done_reason": "length"}),
        ("anthropic", "anthropic", {"content": [{"text": "# Brief"}], "stop_reason": "max_tokens"}),
        ("openrouter", "openrouter", {"choices": [{"message": {"content": "# Brief"}, "finish_reason": "length"}]}),
        ("sambanova", "sambanova", {"choices": [{"message": {"content": "# Brief"}, "finish_reason": "length"}]}),
    ],
)
def test_cut_off_answer_is_an_error_on_every_provider(monkeypatch, host_post, provider, host, cut_body):
    # Only Groq and DeepSeek checked this before: a brief cut at max_tokens
    # from Ollama, Anthropic or OpenRouter passed as a finished answer.
    monkeypatch.setattr(config, "LLM_PROVIDER", provider)
    monkeypatch.setattr(config, "SAMBANOVA_API_KEY", "test-samba")
    monkeypatch.setattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")
    replies, _ = host_post
    replies[host].append(FakeResponse(200, cut_body))
    with pytest.raises(LLMConnectionError, match="cut off"):
        call_llm("sys", "user", max_tokens=700)


def test_openrouter_never_returns_its_reasoning_as_the_answer(monkeypatch, host_post):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openrouter")
    replies, _ = host_post
    body = {"choices": [{"message": {"content": "", "reasoning": "First I think about 42 trains..."}}]}
    replies["openrouter"].append(FakeResponse(200, body))
    with pytest.raises(LLMConnectionError, match="empty response"):
        call_llm("sys", "user")


def test_malformed_body_is_a_connection_error_and_auto_falls_back(monkeypatch, host_post):
    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    replies, calls = host_post
    replies["groq"].append(FakeResponse(200, {"unexpected": "shape"}))
    replies["deepseek"].append(FakeResponse(200, OK))

    answer = call_llm("sys", "user")
    assert answer == "SELECT 1;"
    assert answer.answered_by.startswith("deepseek")


# --- Who answered, and the audit line per call ---------------------------------


def test_answer_carries_who_answered_even_when_the_global_is_overwritten(routed_post):
    replies, _ = routed_post
    replies["groq"].append(FakeResponse(200, OK))
    answer = call_llm("sys", "user")
    llm_client.LAST_ANSWERED_BY = "another request · another model"
    assert answer.answered_by == "groq · openai/gpt-oss-120b"


def test_each_provider_call_logs_one_structured_line(routed_post, caplog):
    replies, _ = routed_post
    body = {"choices": [{"message": {"content": "SELECT 1;"}}], "usage": {"prompt_tokens": 2301, "completion_tokens": 42}}
    replies["groq"].append(FakeResponse(200, body))

    with caplog.at_level("INFO", logger="app.llm_client"):
        call_llm("sys", "user")
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("LLM_CALL")]
    assert len(lines) == 1
    line = lines[0]
    assert "provider=groq" in line and "model=openai/gpt-oss-120b" in line
    assert f"prompt_sha={llm_client._prompt_hash('sys', 'user')}" in line
    assert "prompt_tokens=2301" in line and "completion_tokens=42" in line
    assert "outcome=ok" in line and "latency_ms=" in line


def test_a_cache_hit_makes_no_provider_call(routed_post, caplog):
    replies, calls = routed_post
    replies["groq"].append(FakeResponse(200, OK))
    with caplog.at_level("INFO", logger="app.llm_client"):
        first = call_llm("sys", "user")
        second = call_llm("sys", "user")
    assert len(calls) == 1
    assert second.answered_by == first.answered_by
    assert sum(r.getMessage().startswith("LLM_CALL") for r in caplog.records) == 1


def test_a_rate_limit_keeps_the_providers_reason(routed_post, caplog):
    # Groq says in the 429 body which limit was hit; it used to be dropped,
    # so a daily quota could not be told apart from a per-minute burst.
    replies, _ = routed_post
    body = {"error": {"message": "Rate limit reached on tokens per day (TPD): Limit 200000"}}
    replies["groq"].append(FakeResponse(429, body))

    with caplog.at_level("INFO", logger="app.llm_client"):
        with pytest.raises(LLMRateLimitError, match="tokens per day"):
            call_llm("sys", "user")
    assert any("tokens per day" in r.getMessage() for r in caplog.records if "LLM_CALL" in r.getMessage())
