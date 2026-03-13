"""Central configuration loaded from environment / .env file."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── LLM ─────────────────────────────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")

# Groq
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_FALLBACK_MODEL: str = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")

# Google Gemini  (free key at https://aistudio.google.com/apikey)
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_FALLBACK_MODEL: str = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.0-flash-lite")

# ── Knowledge base ───────────────────────────────────────────────────────────
KB_PATH: Path = Path(os.getenv("KB_PATH", "./knowledge_base"))
KB_PATH.mkdir(parents=True, exist_ok=True)

# ── Retrieval ────────────────────────────────────────────────────────────────
MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "8"))
MAX_PAGES_PER_QUERY: int = int(os.getenv("MAX_PAGES_PER_QUERY", "8"))
CACHE_TTL_HOURS: int = int(os.getenv("CACHE_TTL_HOURS", "24"))

# ── Search providers ─────────────────────────────────────────────────────────
# Tavily: purpose-built search for RAG, returns pre-cleaned content.
# Get a free key at https://app.tavily.com  (1 000 searches/month free)
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# ── Re-ranking ───────────────────────────────────────────────────────────────
# Cross-encoder re-ranking via flashrank (no torch required).
# Set to "false" to skip re-ranking (saves ~200 ms per query).
USE_RERANKER: bool = os.getenv("USE_RERANKER", "true").lower() == "true"

# ── Document pre-processing ──────────────────────────────────────────────────
# When True, each scraped page is run through the LLM before storage to strip
# navigation/boilerplate and extract only visa-relevant content.
# Adds ~1-2 s per page but produces much cleaner chunks for RAG.
PREPROCESS_DOCS: bool = os.getenv("PREPROCESS_DOCS", "false").lower() == "true"

# ── LLM-driven query generation ──────────────────────────────────────────────
# When True, the LLM generates targeted search queries before the web search,
# using its knowledge of how each country's visa system works (e.g. VFS Global
# vs direct consulate, typical portal names, local service centres).
# Adds one fast LLM call (~1 s) but significantly improves page coverage.
USE_LLM_QUERY_GEN: bool = os.getenv("USE_LLM_QUERY_GEN", "false").lower() == "true"
