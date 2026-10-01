"""
Shared fixtures for the offline test suite.

Two guarantees for every test:
- no network: socket connections raise, so a provider call that is not mocked
  fails loudly instead of spending free-tier quota;
- no stale LLM cache: call_llm() results are lru-cached per process, so the
  cache is cleared between tests to keep mocked responses independent.
"""

import logging
import socket

# app.db calls logging.basicConfig(filename="query_audit.log") at import time.
# basicConfig is a no-op when the root logger already has a handler, so adding
# one here keeps the test run from writing to the real audit log.
logging.getLogger().addHandler(logging.NullHandler())

import pytest  # noqa: E402

from app import llm_client  # noqa: E402


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise RuntimeError("Network access is forbidden in tests; mock requests.post instead.")

    monkeypatch.setattr(socket.socket, "connect", _blocked)


@pytest.fixture(autouse=True)
def clear_llm_cache():
    llm_client._call_llm_cached.cache_clear()
    yield
    llm_client._call_llm_cached.cache_clear()


@pytest.fixture(autouse=True)
def fake_llm_config(monkeypatch):
    """Replace whatever the real .env loaded (real keys, real provider) with
    fixed fake values, so tests never depend on or carry a real key."""
    from app import config

    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(config, "AUTO_PROVIDER_ORDER", ("deepseek", "groq"))
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-deepseek")
    monkeypatch.setattr(config, "DEEPSEEK_BASE_URL", "https://deepseek.test")
    monkeypatch.setattr(config, "DEEPSEEK_MODEL", "deepseek-flash")
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-groq-1")
    monkeypatch.setattr(config, "GROQ_API_KEYS", ["test-groq-1"])
    monkeypatch.setattr(config, "GROQ_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-openrouter")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-anthropic")
    monkeypatch.setattr(llm_client, "_groq_key_index", 0)
    monkeypatch.setattr(llm_client, "LAST_ANSWERED_BY", None)
