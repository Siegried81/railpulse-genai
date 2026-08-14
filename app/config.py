"""
Central configuration for RailPulse AI.

Switch LLM_PROVIDER to change which backend answers Text-to-SQL and
consultant-style questions, without touching any other file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# LLM provider switch 
# One of: "ollama", "groq"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# Ollama (local, free, open-source) 
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Groq (free-tier hosted open-source models) 
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Database 
DB_PATH = os.getenv("DB_PATH", "data/railpulse_ai.db")