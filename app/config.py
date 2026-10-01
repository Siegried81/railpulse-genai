"""
Central configuration for RailPulse AI.

Switch LLM_PROVIDER to change which backend answers Text-to-SQL and
consultant-style questions, without touching any other file.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _numbered_keys(name: str, environ=os.environ) -> list[str]:
    """Collect NAME, NAME_2, NAME_3, ... (stopping at the first gap), skipping empties.

    Lets .env hold several free-tier keys for one provider without a code
    change per key; each key has its own rate-limit quota.
    """
    keys = [environ.get(name, "")]
    i = 2
    while f"{name}_{i}" in environ:
        keys.append(environ[f"{name}_{i}"])
        i += 1
    return [k for k in keys if k]


# --- LLM provider switch ------------------------------------------------
# One of: "auto", "deepseek", "groq", "ollama", "openrouter", "anthropic".
# "auto" tries AUTO_PROVIDER_ORDER in turn and uses the first that answers.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
AUTO_PROVIDER_ORDER = ("deepseek", "groq")

# --- DeepSeek (hosted, OpenAI-compatible) ---------------------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

# --- Anthropic (used for fast prototyping only, not the open-source path)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# --- Ollama (local, free, open-source) -----------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# --- Groq (free-tier hosted open-source models) --------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEYS = _numbered_keys("GROQ_API_KEY")  # GROQ_API_KEY, GROQ_API_KEY_2, ...
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# --- OpenRouter (free-tier hosted, OpenAI-compatible) ---------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

# --- Database -------------------------------------------------------------
DB_PATH = os.getenv("DB_PATH", "data/railpulse_ai.db")