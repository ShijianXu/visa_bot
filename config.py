"""Central configuration loaded from environment / .env file."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── LLM ─────────────────────────────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Knowledge base ───────────────────────────────────────────────────────────
KB_PATH: Path = Path(os.getenv("KB_PATH", "./knowledge_base"))
KB_PATH.mkdir(parents=True, exist_ok=True)

# ── Retrieval ────────────────────────────────────────────────────────────────
MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "6"))
MAX_PAGES_PER_QUERY: int = int(os.getenv("MAX_PAGES_PER_QUERY", "4"))
CACHE_TTL_HOURS: int = int(os.getenv("CACHE_TTL_HOURS", "24"))
