# Visa Bot

An AI-powered CLI assistant that helps users navigate visa applications. Given your nationality, residence, destination, and travel details, it retrieves information from official government sources and generates a structured, step-by-step application guide.

## Features

- Determines whether a visa is required (visa-free, visa on arrival, e-visa, or embassy visa)
- Identifies the responsible embassy or consulate based on your country of residence
- Retrieves information from official government and embassy websites
- Generates a step-by-step application guide with required documents, fees, and processing times
- Stores retrieved information locally in a vector database (ChromaDB) for fast follow-up questions (RAG)
- Minimizes redundant web searches and LLM calls through caching

## Project Structure

```
visa_bot/
├── main.py               # Entry point
├── config.py             # Configuration and settings
├── core/
│   ├── agent.py          # Main VisaAgent orchestrator
│   └── workflow.py       # Conversation and reasoning workflow
├── llm/
│   ├── base.py           # LLM provider interface
│   ├── factory.py        # Provider factory
│   └── groq_provider.py  # Groq LLM integration
├── retrieval/
│   ├── scraper.py        # Web scraping (Trafilatura)
│   └── searcher.py       # DuckDuckGo search integration
├── knowledge/
│   ├── models.py         # Document data models
│   └── store.py          # ChromaDB vector store
├── ui/
│   └── display.py        # Rich terminal UI
├── knowledge_base/       # Local ChromaDB storage (auto-created)
├── .env.example          # Environment variable template
└── pyproject.toml        # Project dependencies
```

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- A [Groq API key](https://console.groq.com) (free tier available)

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

Edit `.env` and set your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

**4. Run**

```bash
uv run python main.py
# or
python main.py
```

## Configuration

All settings are in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq Cloud API key |
| `LLM_PROVIDER` | `groq` | LLM provider (currently: groq) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model to use |
| `KB_PATH` | `./knowledge_base` | Path to local vector database |
| `MAX_SEARCH_RESULTS` | `6` | Max web search results per query |
| `MAX_PAGES_PER_QUERY` | `4` | Max pages scraped per query |
| `CACHE_TTL_HOURS` | `24` | Cache lifetime in hours |

## How It Works

1. **Input collection** — the assistant asks for your nationality, country of residence, destination, travel purpose, and departure date
2. **Web retrieval** — searches official government, embassy, and consulate websites using DuckDuckGo; scrapes pages with Trafilatura
3. **Knowledge storage** — saves extracted content into a local ChromaDB vector store with metadata (source URL, countries, visa type, timestamp)
4. **RAG answering** — follow-up questions are answered by retrieving relevant stored documents, minimizing additional web searches and LLM calls
5. **Structured output** — produces a formatted guide with visa requirement, application timeline, where to apply, step-by-step procedure, required documents, fees, and official links

## Tech Stack

| Component | Library |
|---|---|
| LLM | [Groq](https://groq.com) (Llama 3.3 70B) |
| Web search | [duckduckgo-search](https://github.com/deedy5/duckduckgo_search) |
| Web scraping | [Trafilatura](https://trafilatura.readthedocs.io) |
| Vector database | [ChromaDB](https://www.trychroma.com) |
| Terminal UI | [Rich](https://rich.readthedocs.io) |
| HTTP | [Requests](https://requests.readthedocs.io) |
| Package manager | [uv](https://github.com/astral-sh/uv) / [Hatchling](https://hatch.pypa.io) |

## Disclaimer

Visa policies change frequently. Always verify information with the official embassy or consulate website before making travel plans. This tool is intended to assist research, not replace official advice.
