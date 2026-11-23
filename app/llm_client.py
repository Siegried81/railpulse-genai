"""
Provider-agnostic LLM client.

call_llm(system_prompt, user_prompt) returns the answer text as an LLMAnswer:
a plain str that also carries `.answered_by` ("provider · model"), whichever
backend is configured in config.LLM_PROVIDER. With LLM_PROVIDER="auto",
providers are tried in config.AUTO_PROVIDER_ORDER (Groq with key rotation,
then SambaNova, DeepSeek, OpenRouter and a local Ollama).

Which model answered travels on the returned string rather than only in a
module global: the API serves requests from a thread pool and Streamlit runs
each session in its own thread, so a global written by one request could be
read by another. LAST_ANSWERED_BY is still set for the single-threaded CLI
scripts.

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
  backend can't be reached, answers with a malformed body, or stops at
  max_tokens before finishing. Callers catch this one exception type
  regardless of provider, and "auto" moves on to the next provider.

Observability: every real provider call writes one LLM_CALL line to the
audit log with provider, model, a hash of the prompt pair, attempt, latency,
token usage and outcome, so a bad answer can be traced to the exact prompt
version and model that produced it. Cache hits make no call and log no
LLM_CALL line; LLM_ANSWERED is logged for every call_llm(), hit or not.
"""

from functools import lru_cache
import hashlib
import logging
import re
import time

import requests
from app import config

logger = logging.getLogger(__name__)

# "provider · model" of the last call_llm() answer (cache hits included), for
# the single-threaded CLI scripts. Code that can run concurrently (the engine,
# so the UI and the API) reads LLMAnswer.answered_by instead.
LAST_ANSWERED_BY: str | None = None


class LLMAnswer(str):
    """The answer text, plus which "provider · model" produced it.

    A str subclass so every caller that treats the answer as text keeps
    working unchanged, while the engine can read who answered from the value
    itself instead of from a module global shared between threads.
    """

    answered_by: str | None = None

    def __new__(cls, text: str, answered_by: str | None):
        obj = super().__new__(cls, text)
        obj.answered_by = answered_by
        return obj


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
) -> LLMAnswer:
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
    return LLMAnswer(_strip_reasoning_leak(text), answered_by)


def call_llm_until(
    system_prompt: str,
    user_prompt: str,
    is_valid,
    stop: tuple[str, ...] | None = None,
    max_tokens: int | None = None,
    max_attempts: int = 3,
) -> LLMAnswer:
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
    result = LLMAnswer("", None)
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
        "sambanova": config.SAMBANOVA_MODEL,
        "anthropic": config.ANTHROPIC_MODEL,
    }.get(provider, "unknown")


def _prompt_hash(system_prompt: str, user_prompt: str) -> str:
    """Short, stable fingerprint of a prompt pair for the audit log.

    The log keeps no prompt text (the user prompt holds the question and the
    rows), but a hash still tells whether two answers came from the same
    prompt version.
    """
    digest = hashlib.sha256(f"{system_prompt}\x00{user_prompt}".encode("utf-8"))
    return digest.hexdigest()[:12]


def _call_one_provider(
    provider: str,
    system_prompt: str,
    user_prompt: str,
    stop: tuple[str, ...] | None,
    max_tokens: int | None,
    attempt: int = 0,
) -> str:
    """Call one provider, log one LLM_CALL line, and return the answer text.

    A response body that does not have the expected shape (a KeyError, an
    IndexError, invalid JSON...) is turned into an LLMConnectionError, so
    "auto" falls back to the next provider instead of crashing on it.
    """
    providers = {
        "deepseek": _call_deepseek,
        "ollama": _call_ollama,
        "groq": _call_groq,
        "openrouter": _call_openrouter,
        "sambanova": _call_sambanova,
        "anthropic": _call_anthropic,
    }
    if provider not in providers:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")

    started = time.monotonic()
    usage: dict = {}
    outcome = "ok"
    try:
        try:
            text, usage = providers[provider](system_prompt, user_prompt, stop, max_tokens)
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise LLMConnectionError(f"{provider} returned a malformed response: {e!r}") from e
    except LLMConnectionError as e:
        outcome = f"{type(e).__name__}: {str(e)[:400]}"
        raise
    finally:
        logger.info(
            f"LLM_CALL | provider={provider} | model={_model_of(provider)}"
            f" | prompt_sha={_prompt_hash(system_prompt, user_prompt)} | attempt={attempt}"
            f" | latency_ms={round((time.monotonic() - started) * 1000)}"
            f" | prompt_tokens={usage.get('prompt_tokens')}"
            f" | completion_tokens={usage.get('completion_tokens')} | outcome={outcome}"
        )
    return text


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
    next one. If all fail and at least one of them was rate-limited, the
    error is a rate limit, so call_llm_until() backs off and retries: waiting
    can free that provider, whereas the others (no key, server down) would
    fail the same way again. Otherwise it is a plain connection error listing
    each provider's failure.
    """
    if provider != "auto":
        text = _call_one_provider(provider, system_prompt, user_prompt, stop, max_tokens, _attempt)
        return text, f"{provider} · {_model_of(provider)}"

    errors: list[tuple[str, LLMConnectionError]] = []
    for candidate in config.AUTO_PROVIDER_ORDER:
        try:
            text = _call_one_provider(candidate, system_prompt, user_prompt, stop, max_tokens, _attempt)
        except LLMConnectionError as e:
            logger.info(f"LLM_FALLBACK | provider={candidate} | error={e}")
            errors.append((candidate, e))
            continue
        return text, f"{candidate} · {_model_of(candidate)}"

    summary = " | ".join(f"{name}: {e}" for name, e in errors)
    if any(isinstance(e, LLMRateLimitError) for _, e in errors):
        raise LLMRateLimitError(f"No provider answered and at least one is rate-limited. {summary}")
    raise LLMConnectionError(f"No LLM provider answered. {summary}")


def _error_detail(response) -> str:
    """The provider's own error message from an HTTP error body, or "".

    A 429 says which limit was hit (per minute or per day, tokens or
    requests) only in its body. Without it a run that stays rate-limited for
    ten minutes cannot be told apart from a burst that clears in one.
    """
    try:
        body = response.json()
    except ValueError:
        return ""
    error = body.get("error") if isinstance(body, dict) else None
    message = error.get("message") if isinstance(error, dict) else error
    return f" Provider said: {str(message)[:300]}" if message else ""


def _openai_usage(data: dict) -> dict:
    """Token counts from an OpenAI-compatible response, {} when absent."""
    usage = data.get("usage") or {}
    return {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens")}


def _call_ollama(
    system_prompt: str, user_prompt: str, stop: tuple[str, ...] | None, max_tokens: int | None
) -> tuple[str, dict]:
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

    data = response.json()
    # Ollama reports a generation stopped by num_predict as done_reason "length".
    if data.get("done_reason") == "length":
        raise LLMConnectionError(f"Ollama answer was cut off at max_tokens={max_tokens} before it finished.")
    usage = {"prompt_tokens": data.get("prompt_eval_count"), "completion_tokens": data.get("eval_count")}
    return data["message"]["content"].strip(), usage


def _call_deepseek(
    system_prompt: str, user_prompt: str, stop: tuple[str, ...] | None, max_tokens: int | None
) -> tuple[str, dict]:
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
            raise LLMRateLimitError(f"DeepSeek rate limit hit. Wait a moment and try again.{_error_detail(response)}") from e
        raise LLMConnectionError(f"DeepSeek API error (model '{config.DEEPSEEK_MODEL}'): {e}") from e

    data = response.json()
    return _complete_content(data, "DeepSeek", max_tokens), _openai_usage(data)


def _call_sambanova(
    system_prompt: str, user_prompt: str, stop: tuple[str, ...] | None, max_tokens: int | None
) -> tuple[str, dict]:
    """Call SambaNova Cloud (OpenAI-compatible chat completions).

    Second in "auto", after Groq: a free cloud tier with its own quota, so a
    Groq rate limit does not have to fall through to the paid DeepSeek
    credit. With no key configured it fails before any HTTP call, so "auto"
    skips it at no cost.
    """
    if not config.SAMBANOVA_API_KEY:
        raise LLMConnectionError("No SambaNova API key configured (set SAMBANOVA_API_KEY in .env).")

    payload = {
        "model": config.SAMBANOVA_MODEL,
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
            config.SAMBANOVA_URL,
            headers={"Authorization": f"Bearer {config.SAMBANOVA_API_KEY}"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise LLMConnectionError("Can't reach SambaNova's API. Check your internet connection.") from e
    except requests.exceptions.Timeout as e:
        raise LLMConnectionError("SambaNova API timed out after 60s.") from e
    except requests.exceptions.HTTPError as e:
        if response.status_code in (401, 403):
            raise LLMConnectionError("SambaNova API key is missing or invalid (check SAMBANOVA_API_KEY in .env).") from e
        if response.status_code == 429:
            raise LLMRateLimitError(f"SambaNova rate limit hit. Wait a moment and try again.{_error_detail(response)}") from e
        raise LLMConnectionError(f"SambaNova API error (model '{config.SAMBANOVA_MODEL}'): {e}") from e

    data = response.json()
    return _complete_content(data, "SambaNova", max_tokens), _openai_usage(data)


# Index into config.GROQ_API_KEYS of the key that last answered, so the next
# call starts there instead of re-hitting keys already known to be exhausted.
_groq_key_index = 0


def _call_groq(
    system_prompt: str, user_prompt: str, stop: tuple[str, ...] | None, max_tokens: int | None
) -> tuple[str, dict]:
    """Call Groq, rotating through config.GROQ_API_KEYS.

    Groq's free tier caps gpt-oss-120b at 8K tokens a minute and 200,000
    tokens a day, and its 429 names the organization the limit applies to: at
    least the daily cap is shared by every key of one Groq account. Extra keys
    therefore add capacity only when they come from different accounts. One
    Text-to-SQL call is about 3.7K prompt tokens (3,732 to 3,747 measured), so
    one organization serves about 50 questions a day. On a 429 or a rejected
    key the next key is tried; any other error is raised at once, since
    another key would not fix a bad model name or an outage.
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
        raise LLMRateLimitError(
            f"All {len(keys)} Groq keys are rate-limited. Wait a minute and try again. Last key: {last_error}"
        )
    raise LLMConnectionError(f"No working Groq key among {len(keys)}: {last_error}")


class _LLMInvalidKeyError(LLMConnectionError):
    """A provider rejected one specific API key (HTTP 401): try the next one."""


def _call_groq_with_key(
    api_key: str, system_prompt: str, user_prompt: str, stop: tuple[str, ...] | None, max_tokens: int | None
) -> tuple[str, dict]:
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
            raise LLMRateLimitError(
                f"Groq free-tier rate limit hit. Wait a moment and try again.{_error_detail(response)}"
            ) from e
        raise LLMConnectionError(f"Groq API error: {e}") from e

    data = response.json()
    return _complete_content(data, "Groq", max_tokens), _openai_usage(data)


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
) -> tuple[str, dict]:
    """Call OpenRouter's OpenAI-compatible API. Free-tier fallback for Groq.

    Sign in at https://openrouter.ai/keys (Google/GitHub/email) and pick any
    model with a ":free" suffix, e.g. "meta-llama/llama-3.1-8b-instruct:free".
    Only the `content` field is used: the `reasoning` field holds the model's
    chain of thought, which is not an answer even when content is empty.
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

    content = _complete_content(data, "OpenRouter", max_tokens)
    if not content:
        raise LLMConnectionError(
            "OpenRouter returned an empty response (the free model may be overloaded). "
            "Try again, or set OPENROUTER_MODEL to a specific model in openrouter.ai/models "
            "instead of the openrouter/free auto-router."
        )
    return content, _openai_usage(data)


def _call_anthropic(
    system_prompt: str, user_prompt: str, stop: tuple[str, ...] | None, max_tokens: int | None
) -> tuple[str, dict]:
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

    data = response.json()
    if data.get("stop_reason") == "max_tokens":
        raise LLMConnectionError(f"Anthropic answer was cut off at max_tokens={payload['max_tokens']} before it finished.")
    usage = data.get("usage") or {}
    return data["content"][0]["text"].strip(), {
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
    }
