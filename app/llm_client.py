"""
Provider-agnostic LLM client.

call_llm(system_prompt, user_prompt) always returns a plain string,
regardless of which backend is configured in config.LLM_PROVIDER.
With LLM_PROVIDER="auto", providers are tried in config.AUTO_PROVIDER_ORDER
(DeepSeek, then Groq with key rotation); LAST_ANSWERED_BY and the audit log
record which one actually answered.

Speed optimizations:
- `stop` sequences let the generation halt as soon as the model finishes
  the useful part of its answer (e.g. right after the closing ";" of a
  SQL query), instead of letting it keep rambling.
- `max_tokens` caps generation length as a safety net.
- Ollama calls set `keep_alive` so the model stays resident in memory
  between requests instead of being reloaded from disk every time.
- Responses are cached in-process (lru_cache): since every provider is
  called with temperature=0, the same (system_prompt, user_prompt) pair
  is deterministic, so repeated questions are instant on a cache hit.

Robustness:
- LLMConnectionError is raised (instead of an unhandled requests
  exception bubbling up as a raw traceback) whenever the configured
  backend can't be reached -- e.g. Ollama isn't running, or a hosted
  API key/network is misconfigured. Callers (the Streamlit UI, the CLI
  test harness) can catch this one exception type regardless of provider.
"""

from functools import lru_cache
import logging
import re
import time

import requests
from app import config

logger = logging.getLogger(__name__)

# "provider · model" of the last call_llm() answer (cache hits included), so
# callers can record which model actually answered -- with "auto", that is
# not necessarily the first provider in the chain.
LAST_ANSWERED_BY: str | None = None


class LLMConnectionError(Exception):
    """Raised when the configured LLM backend can't be reached or times out."""


class LLMRateLimitError(LLMConnectionError):
    """Raised on an HTTP 429 from any provider.

    A subclass rather than a message check, so call_llm_until() can recognise
    a rate limit the same way whatever wording each provider's error uses,
    while existing callers that catch LLMConnectionError keep working.
    """


# First-line openers of the leaked planning preamble ("I need to: 1. ... 2. ...").
# The numbered-list stripping below only runs when the text starts with one of
# these, so a genuine numbered list in an answer (e.g. the weekly report's
# recommendations) is never mistaken for leaked reasoning.
_PLANNING_PREAMBLE = re.compile(
    r"^\s*(i need to|i will|i'll|i should|let me|let's|my plan|steps?\b|first,? i)",
    re.IGNORECASE,
)


def _strip_reasoning_leak(text: str) -> str:
    """Some free/routed models leak their chain-of-thought into the content
    field instead of a separate 'reasoning' field. Strip common patterns:
    explicit <think>/<reasoning> tags, and a numbered "I need to: ..." style
    preamble some instruction-following-weak models produce before the real
    answer. Best-effort only -- the prompt-level instruction and the
    reasoning:exclude API param are the primary fixes, this is a fallback.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.IGNORECASE | re.DOTALL)

    lines = text.strip().split("\n")
    if not _PLANNING_PREAMBLE.match(lines[0]):
        return text.strip()

    last_numbered_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*\d+[\.\)]\s", line):
            last_numbered_idx = i
    if last_numbered_idx is not None and last_numbered_idx < len(lines) - 1:
        remainder = "\n".join(lines[last_numbered_idx + 1 :]).strip()
        if remainder:
            text = remainder

    return text.strip()


def looks_like_leaked_meta(text: str) -> bool:
    """Heuristic check for the 'User Safety: safe' style leak: a short,
    meta-sounding fragment instead of an actual multi-word answer. Used as
    the is_valid check for consultant/free-text calls with call_llm_until.
    """
    stripped = text.strip()
    if len(stripped) < 20:
        return True
    if re.match(r"(?i)^(user\s+safety|safety\s*[:\-]|moderation|policy\s*[:\-])", stripped):
        return True
    return False


def call_llm(
    system_prompt: str,
    user_prompt: str,
    stop: tuple[str, ...] | None = None,
    max_tokens: int | None = None,
    _attempt: int = 0,
) -> str:
    """Route the call to whichever provider is configured.

    `stop` must be a tuple (not a list) so the call is hashable for caching.
    `_attempt` is not sent to the API -- it only changes the cache key, so
    call_llm_until() below can retry a flaky response instead of replaying
    the same cached bad result.
    """
    global LAST_ANSWERED_BY
    text, answered_by = _call_llm_cached(
        config.LLM_PROVIDER, system_prompt, user_prompt, stop, max_tokens, _attempt
    )
    LAST_ANSWERED_BY = answered_by
    logger.info(f"LLM_ANSWERED | by={answered_by} | attempt={_attempt}")
    return _strip_reasoning_leak(text)


def call_llm_until(
    system_prompt: str,
    user_prompt: str,
    is_valid,
    stop: tuple[str, ...] | None = None,
    max_tokens: int | None = None,
    max_attempts: int = 3,
) -> str:
    """Call the LLM, retrying (bypassing the cache each time) while
    is_valid(result) is False. Free-tier auto-routers (e.g. OpenRouter's
    openrouter/free) occasionally land on a flaky backing model even at
    temperature=0 -- e.g. leaking a stray moderation/meta line instead of
    the actual SQL. Retrying with a fresh call recovers from this most of
    the time. Returns the last attempt's output even if never valid, so
    the caller can still surface a clean error instead of hanging forever.

    A 429 rate-limit error gets one short backoff-and-retry per attempt
    (free tiers are commonly capped around 20 requests/minute); any other
    LLMConnectionError (bad key, backend down, etc.) is re-raised right
    away since retrying won't fix it.
    """
    result = ""
    for attempt in range(max_attempts):
        try:
            result = call_llm(system_prompt, user_prompt, stop=stop, max_tokens=max_tokens, _attempt=attempt)
        except LLMRateLimitError:
            if attempt < max_attempts - 1:
                time.sleep(5)
                continue
            raise
        if is_valid(result):
            return result
    return result


def _model_of(provider: str) -> str:
    return {
        "deepseek": config.DEEPSEEK_MODEL,
        "groq": config.GROQ_MODEL,
        "ollama": config.OLLAMA_MODEL,
        "openrouter": config.OPENROUTER_MODEL,
        "anthropic": config.ANTHROPIC_MODEL,
    }.get(provider, "unknown")


def _call_one_provider(
    provider: str, system_prompt: str, user_prompt: str, stop: tuple[str, ...] | None, max_tokens: int | None
) -> str:
    if provider == "deepseek":
        return _call_deepseek(system_prompt, user_prompt, stop, max_tokens)
    elif provider == "ollama":
        return _call_ollama(system_prompt, user_prompt, stop, max_tokens)
    elif provider == "groq":
        return _call_groq(system_prompt, user_prompt, stop, max_tokens)
    elif provider == "openrouter":
        return _call_openrouter(system_prompt, user_prompt, stop, max_tokens)
    elif provider == "anthropic":
        return _call_anthropic(system_prompt, user_prompt, stop, max_tokens)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


@lru_cache(maxsize=256)
def _call_llm_cached(
    provider: str,
    system_prompt: str,
    user_prompt: str,
    stop: tuple[str, ...] | None,
    max_tokens: int | None,
    _attempt: int,
) -> tuple[str, str]:
    """Return (text, "provider · model") for whoever actually answered.

    With provider "auto", config.AUTO_PROVIDER_ORDER is tried in turn and the
    first provider that answers wins. Any LLMConnectionError moves on to the
    next one. If all fail, the error is a rate limit only when every provider
    was rate-limited (so call_llm_until() backs off and retries), otherwise a
    plain connection error listing each provider's failure.
    """
    if provider != "auto":
        text = _call_one_provider(provider, system_prompt, user_prompt, stop, max_tokens)
        return text, f"{provider} · {_model_of(provider)}"

    errors: list[tuple[str, LLMConnectionError]] = []
    for candidate in config.AUTO_PROVIDER_ORDER:
        try:
            text = _call_one_provider(candidate, system_prompt, user_prompt, stop, max_tokens)
        except LLMConnectionError as e:
            logger.info(f"LLM_FALLBACK | provider={candidate} | error={e}")
            errors.append((candidate, e))
            continue
        return text, f"{candidate} · {_model_of(candidate)}"

    summary = " | ".join(f"{name}: {e}" for name, e in errors)
    if all(isinstance(e, LLMRateLimitError) for _, e in errors):
        raise LLMRateLimitError(f"Every provider is rate-limited. {summary}")
    raise LLMConnectionError(f"No LLM provider answered. {summary}")


def _call_ollama(
    system_prompt: str, user_prompt: str, stop: tuple[str, ...] | None, max_tokens: int | None
) -> str:
    options = {"temperature": 0}
    if max_tokens:
        options["num_predict"] = max_tokens
    if stop:
        options["stop"] = list(stop)

    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": config.OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": options,
                "keep_alive": "30m",  # keep the model resident between requests
            },
            timeout=300,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise LLMConnectionError(
            
            f"Can't reach Ollama at {config.OLLAMA_BASE_URL}. Make sure it's running "
            f"(`ollama serve`) and that the model is pulled (`ollama pull {config.OLLAMA_MODEL}`)."
        ) from e
    except requests.exceptions.Timeout as e:
        raise LLMConnectionError(
            "Ollama timed out after 300s. The model may still be loading, or this "
            "machine may be too slow for it -- try a smaller model."
        ) from e
    except requests.exceptions.RequestException as e:
        raise LLMConnectionError(f"Ollama request failed: {e}") from e

    return response.json()["message"]["content"].strip()


def _call_deepseek(
    system_prompt: str, user_prompt: str, stop: tuple[str, ...] | None, max_tokens: int | None
) -> str:
    """Call DeepSeek's hosted API (OpenAI-compatible).

    `stop` is passed through: DeepSeek returns its reasoning in a separate
    reasoning_content field, and a ";" stop was checked not to cut the answer.
    """
    payload = {
        "model": config.DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if stop:
        payload["stop"] = list(stop)

    try:
        response = requests.post(
            f"{config.DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {config.DEEPSEEK_API_KEY}"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise LLMConnectionError("Can't reach DeepSeek's API. Check your internet connection.") from e
    except requests.exceptions.Timeout as e:
        raise LLMConnectionError("DeepSeek API timed out after 60s.") from e
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            raise LLMConnectionError("DeepSeek API key is missing or invalid (check DEEPSEEK_API_KEY in .env).") from e
        if response.status_code == 402:
            raise LLMConnectionError("DeepSeek account balance is insufficient.") from e
        if response.status_code == 429:
            raise LLMRateLimitError("DeepSeek rate limit hit. Wait a moment and try again.") from e
        raise LLMConnectionError(f"DeepSeek API error (model '{config.DEEPSEEK_MODEL}'): {e}") from e

    return _complete_content(response.json(), "DeepSeek", max_tokens)


# Index into config.GROQ_API_KEYS of the key that last answered, so the next
# call starts there instead of re-hitting keys already known to be exhausted.
_groq_key_index = 0


def _call_groq(
    system_prompt: str, user_prompt: str, stop: tuple[str, ...] | None, max_tokens: int | None
) -> str:
    """Call Groq, rotating through config.GROQ_API_KEYS.

    Each free-tier key has its own tokens-per-minute quota (checked: a key
    drained by earlier calls did not reduce the others), and one Text-to-SQL
    call is ~2.3K prompt tokens against an 8K/minute limit. So on a 429 or a
    rejected key, the next key is tried; any other error is raised at once,
    since another key would not fix a bad model name or an outage.
    """
    global _groq_key_index
    keys = config.GROQ_API_KEYS
    if not keys:
        raise LLMConnectionError("No Groq API key configured (set GROQ_API_KEY in .env).")

    last_error: LLMConnectionError | None = None
    for offset in range(len(keys)):
        index = (_groq_key_index + offset) % len(keys)
        try:
            result = _call_groq_with_key(keys[index], system_prompt, user_prompt, stop, max_tokens)
        except (LLMRateLimitError, _LLMInvalidKeyError) as e:
            last_error = e
            continue
        _groq_key_index = index
        return result

    if isinstance(last_error, LLMRateLimitError):
        raise LLMRateLimitError(f"All {len(keys)} Groq keys are rate-limited. Wait a minute and try again.")
    raise LLMConnectionError(f"No working Groq key among {len(keys)}: {last_error}")


class _LLMInvalidKeyError(LLMConnectionError):
    """A provider rejected one specific API key (HTTP 401): try the next one."""


def _call_groq_with_key(
    api_key: str, system_prompt: str, user_prompt: str, stop: tuple[str, ...] | None, max_tokens: int | None
) -> str:
    """Call Groq's free-tier hosted inference API (OpenAI-compatible) with one key.

    gpt-oss models reason before answering, and that reasoning counts against
    max_tokens and against `stop`: with stop=";" the SQL written inside the
    reasoning ended the generation and the answer came back empty. For them,
    `stop` is not sent (extract_sql() already cuts at the first ";") and
    reasoning_effort is set to "low" to keep the reasoning short.
    """
    payload = {
        "model": config.GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }
    is_reasoning_model = config.GROQ_MODEL.startswith("openai/gpt-oss")
    if is_reasoning_model:
        payload["reasoning_effort"] = "low"
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if stop and not is_reasoning_model:
        payload["stop"] = list(stop)

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise LLMConnectionError("Can't reach Groq's API. Check your internet connection.") from e
    except requests.exceptions.Timeout as e:
        raise LLMConnectionError("Groq API timed out after 60s.") from e
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            raise _LLMInvalidKeyError("A Groq API key is invalid (check GROQ_API_KEY* in .env).") from e
        if response.status_code == 404:
            raise LLMConnectionError(
                f"Groq model '{config.GROQ_MODEL}' not found (check GROQ_MODEL in .env)."
            ) from e
        if response.status_code == 429:
            raise LLMRateLimitError("Groq free-tier rate limit hit. Wait a moment and try again.") from e
        raise LLMConnectionError(f"Groq API error: {e}") from e

    return _complete_content(response.json(), "Groq", max_tokens)


def _complete_content(data: dict, provider_name: str, max_tokens: int | None) -> str:
    """Return the answer text of an OpenAI-compatible response, refusing a cut-off one.

    Reasoning models spend part of max_tokens thinking, so an answer can stop
    mid-sentence (finish_reason "length") while still looking valid to a
    shallow check -- a weekly brief was cut after "Investig" yet started with
    "#". A truncated answer is raised as an error instead, so "auto" moves on
    to the next provider rather than publishing half a report.
    """
    choice = data["choices"][0]
    if choice.get("finish_reason") == "length":
        raise LLMConnectionError(
            f"{provider_name} answer was cut off at max_tokens={max_tokens} before it finished."
        )
    return (choice["message"].get("content") or "").strip()


def _call_openrouter(
    system_prompt: str, user_prompt: str, stop: tuple[str, ...] | None, max_tokens: int | None
) -> str:
    """Call OpenRouter's OpenAI-compatible API. Free-tier fallback for Groq.

    Sign in at https://openrouter.ai/keys (Google/GitHub/email) and pick any
    model with a ":free" suffix, e.g. "meta-llama/llama-3.1-8b-instruct:free".
    """
    payload = {
        "model": config.OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        # The openrouter/free auto-router can land on a reasoning model that
        # burns its whole token budget "thinking out loud" in the content
        # field before ever answering -- for a deterministic SQL/short-answer
        # task we never want that. This tells OpenRouter to suppress
        # reasoning tokens from the response for any model that supports it.
        "reasoning": {"exclude": True},
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if stop:
        payload["stop"] = list(stop)

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise LLMConnectionError("Can't reach OpenRouter's API. Check your internet connection.") from e
    except requests.exceptions.Timeout as e:
        raise LLMConnectionError("OpenRouter API timed out after 60s.") from e
    except requests.exceptions.HTTPError as e:
        if response.status_code in (401, 403):
            raise LLMConnectionError(
                "OpenRouter API key is missing or invalid (check OPENROUTER_API_KEY in .env)."
            ) from e
        if response.status_code == 429:
            raise LLMRateLimitError(
                "OpenRouter free-tier rate limit hit. Wait a moment, or switch to another "
                ":free model in OPENROUTER_MODEL."
            ) from e
        raise LLMConnectionError(f"OpenRouter API error: {e}") from e

    data = response.json()
    if "error" in data:
        raise LLMConnectionError(f"OpenRouter API error: {data['error'].get('message', data['error'])}")

    message = data["choices"][0]["message"]
    content = message.get("content") or message.get("reasoning") or ""
    if not content.strip():
        raise LLMConnectionError(
            "OpenRouter returned an empty response (the free model may be overloaded). "
            "Try again, or set OPENROUTER_MODEL to a specific model in openrouter.ai/models "
            "instead of the openrouter/free auto-router."
        )
    return content.strip()



def _call_anthropic(
    system_prompt: str, user_prompt: str, stop: tuple[str, ...] | None, max_tokens: int | None
) -> str:
    """Call Claude via the Anthropic API (used for prototyping only)."""
    payload = {
        "model": config.ANTHROPIC_MODEL,
        "max_tokens": max_tokens or 1000,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if stop:
        payload["stop_sequences"] = list(stop)

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": config.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise LLMConnectionError("Can't reach the Anthropic API. Check your internet connection.") from e
    except requests.exceptions.Timeout as e:
        raise LLMConnectionError("Anthropic API timed out after 60s.") from e
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            raise LLMConnectionError("Anthropic API key is missing or invalid (check ANTHROPIC_API_KEY in .env).") from e
        if response.status_code == 429:
            raise LLMRateLimitError("Anthropic API rate limit hit. Wait a moment and try again.") from e
        raise LLMConnectionError(f"Anthropic API error: {e}") from e

    return response.json()["content"][0]["text"].strip()