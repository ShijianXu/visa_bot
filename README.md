# Visa Bot

An AI-powered CLI assistant that helps users navigate visa applications. Given your nationality, residence, destination, and travel details, it retrieves information from official government sources and generates a structured, step-by-step application guide.

## Features

- Determines whether a visa is required (visa-free, visa on arrival, e-visa, or embassy visa)
- Identifies the responsible embassy or consulate based on your country of **residence** (not nationality)
- Retrieves information from official government and embassy websites, with curated official portals prepended for 60+ countries
- Pre-processes scraped pages with the LLM to strip boilerplate and keep only visa-relevant content
- Cross-encoder re-ranking (via flashrank) to score retrieved chunks by relevance before generation
- Generates a streamed, structured visa guide with required documents, fees, processing times, and source citations
- Stores retrieved information locally in a vector database (ChromaDB) for fast follow-up questions (RAG)
- Caches results per country pair (configurable TTL) to avoid redundant searches
- Follow-up questions answered from the knowledge base; keyword triggers (`search:`, `appointment`, `vfs`, etc.) run a fresh web search automatically

## Project Structure

```
visa_bot/
├── main.py               # Entry point and environment validation
├── config.py             # Centralized configuration from .env
├── core/
│   ├── agent.py          # VisaAgent: input collection, research pipeline, follow-up loop
│   └── workflow.py       # VisaWorkflow: search → scrape → preprocess → store → RAG → re-rank → guide
├── llm/
│   ├── base.py           # LLMProvider abstract interface
│   ├── factory.py        # Provider factory
│   └── groq_provider.py  # Groq Cloud integration (streaming + retry)
├── retrieval/
│   ├── searcher.py       # Web search (Tavily → DuckDuckGo fallback); query builder; official source stubs
│   └── scraper.py        # Web scraping (Trafilatura → Jina Reader fallback); concurrent fetching
├── knowledge/
│   ├── models.py         # VisaDocument and VisaQuery dataclasses
│   └── store.py          # ChromaDB vector store: upsert, semantic search, TTL eviction, chunking
├── ui/
│   └── display.py        # Rich terminal UI: prompts, streaming, progress spinner, source table
├── knowledge_base/       # Local ChromaDB storage (auto-created)
├── .env.example          # Environment variable template
└── pyproject.toml        # Project dependencies
```

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- A [Groq API key](https://console.groq.com) (free tier available)
- Optional: a [Tavily API key](https://app.tavily.com) for higher-quality search (1 000 free searches/month)

## Setup

**1. Clone the repository**

```bash
git clone <repo-url>
cd visa_bot
```

**2. Install dependencies**

```bash
uv sync
```

**3. Configure environment variables**

```bash
cp .env.example .env
```

Edit `.env` and set your API key(s):

```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here   # optional but recommended
```

**4. Run**

```bash
uv run python main.py
# or
python main.py
```

## Configuration

All settings are read from `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq Cloud API key |
| `LLM_PROVIDER` | `groq` | LLM provider (currently: groq) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model to use |
| `TAVILY_API_KEY` | *(optional)* | Tavily search API key — enables richer results and pre-fetched page content |
| `KB_PATH` | `./knowledge_base` | Path to local ChromaDB vector database |
| `MAX_SEARCH_RESULTS` | `8` | Max web search results per query |
| `MAX_PAGES_PER_QUERY` | `8` | Max pages scraped per query |
| `CACHE_TTL_HOURS` | `24` | How long cached results are considered fresh |
| `USE_RERANKER` | `true` | Enable cross-encoder re-ranking via flashrank (saves ~200 ms if disabled) |
| `PREPROCESS_DOCS` | `true` | Run each scraped page through the LLM to extract only visa-relevant content before storing |

## How It Works

### 1. Input Collection

The assistant prompts for:
- Nationality, country of residence, **city of residence** (used to target the exact consulate)
- Destination country, travel purpose, departure date, duration, entry type, and optional companions

### 2. Web Search

Searches are built to target the consulate in the user's **city of residence**, not the destination country's capital. Both Tavily and DuckDuckGo are tried (Tavily first if a key is set). Curated official visa portal URLs for 60+ destination countries are always prepended to results so they are scraped first. Local-language queries are appended to surface official pages that don't appear in English searches.

### 3. Scraping

Each URL is fetched concurrently (up to 5 workers). Trafilatura is tried first; Jina Reader (`r.jina.ai`) is used as a fallback for JavaScript-heavy pages such as VFS Global and embassy portals. If a page cannot be scraped at all, the search snippet is stored instead (if it is long enough to be useful).

When Tavily is used as the search provider, it returns pre-fetched page content directly — those URLs skip scraping entirely.

### 4. Preprocessing & Storage

Each scraped page is passed through the LLM (`PREPROCESS_DOCS=true`) to strip navigation menus, cookie banners, and unrelated content, keeping only visa-relevant facts. The cleaned text is chunked (≤ 2 500 chars, with 400-char overlap at paragraph boundaries) and upserted into ChromaDB with metadata: source URL, nationality, destination country, page title, and timestamp.

Stale documents for the same country pair (older than `CACHE_TTL_HOURS`) are evicted once before each batch is stored.

### 5. RAG Retrieval & Re-ranking

The knowledge base is queried with a structured visa query string. Up to 8 candidate chunks are retrieved, then scored by a cross-encoder (`ms-marco-MiniLM-L-12-v2` via flashrank) and trimmed to the top 6 most relevant.

### 6. Guide Generation

The top chunks are formatted as numbered sources and injected into a detailed prompt. The LLM streams back a structured guide covering:

- Visa requirement (visa-free / on-arrival / e-visa / embassy visa)
- Application timeline
- Where to apply (consulate address, portal URL)
- Required documents (full list)
- Step-by-step application procedure
- Fees and processing time
- Important notes and official source links

Every specific fact is cited inline as `[Source N]`.

### 7. Follow-up Questions

After the guide, an interactive Q&A loop opens. Any question is answered from the stored knowledge base. Questions that contain keywords like `appointment`, `vfs`, `biometric`, `slot`, or an explicit `search:` prefix trigger a fresh web search, which preprocesses and stores new results before answering.

## Tech Stack

| Component | Library |
|---|---|
| LLM | [Groq](https://groq.com) (Llama 3.3 70B) |
| Web search | [Tavily](https://tavily.com) (primary) · [duckduckgo-search](https://github.com/deedy5/duckduckgo_search) (fallback) |
| Web scraping | [Trafilatura](https://trafilatura.readthedocs.io) (primary) · [Jina Reader](https://jina.ai/reader/) (fallback) |
| Re-ranking | [flashrank](https://github.com/PrithivirajDamodaran/FlashRank) (ms-marco cross-encoder) |
| Vector database | [ChromaDB](https://www.trychroma.com) |
| Terminal UI | [Rich](https://rich.readthedocs.io) |
| HTTP | [Requests](https://requests.readthedocs.io) |
| Package manager | [uv](https://github.com/astral-sh/uv) / [Hatchling](https://hatch.pypa.io) |

## Disclaimer

Visa policies change frequently. Always verify information with the official embassy or consulate website before making travel plans. This tool is intended to assist research, not replace official advice.
