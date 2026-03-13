"""Web search module – targets official government and embassy sources.

Search priority:
  1. Tavily (if TAVILY_API_KEY is set) – purpose-built for RAG, returns
     pre-cleaned page content so many pages don't need scraping.
  2. DuckDuckGo – free fallback with no rate limits in practice.

In both cases, curated OFFICIAL_SOURCES for the destination are prepended to
the result list so they are always scraped first.

To update official portal URLs, capitals, or local search terms, edit
retrieval/sources.json — no Python changes required.
"""

import json
from pathlib import Path

import config
from ddgs import DDGS

# Domains considered authoritative for visa info
_OFFICIAL_DOMAINS = (
    ".gov", ".gob", ".gouv", ".gc.ca", ".gov.uk", ".gov.au",
    ".admin.ch", ".europa.eu", "embassy", "consulate", "immigration",
    "mfa.", "mofa.", "mfaic.", "travel.state.gov", "itamaraty",
    "vfsglobal", "tlscontact", "blsintl",
)

# Load curated data from sources.json (editable without touching Python code)
_SOURCES_FILE = Path(__file__).parent / "sources.json"
_sources_data: dict = {}
try:
    _sources_data = json.loads(_SOURCES_FILE.read_text(encoding="utf-8"))
except Exception:
    pass

_CAPITALS: dict[str, str] = _sources_data.get("capitals", {})
_DEST_LOCAL_TERMS: dict[str, str] = _sources_data.get("local_terms", {})


def search_topic(
    topic: str,
    nationality: str,
    destination: str,
    residence: str = "",
    city: str = "",
    max_results: int | None = None,
) -> list[dict]:
    """Search for a specific topic within the context of a visa application."""
    max_results = max_results or config.MAX_SEARCH_RESULTS
    res = residence.lower()
    dest = destination.lower()
    location = city.lower() if city else _CAPITALS.get(res, res)

    queries = [
        f"{destination} consulate {location} {topic}".strip(),
        f"{topic} {destination} visa {nationality} {city or residence} official".strip(),
        f"{destination} {topic} {nationality} apply from {res} site:.gov OR site:.gouv OR site:.gob OR site:.gc.ca",
        f"{destination} embassy {location} {topic} {nationality}".strip(),
    ]

    local = _DEST_LOCAL_TERMS.get(dest, "")
    if local and location:
        queries.append(f"{local} {location}")

    return _run_search(queries, max_results, search_depth="basic")


def search_visa_info(
    nationality: str,
    destination: str,
    residence: str = "",
    city: str = "",
    purpose: str = "tourism",
    max_results: int | None = None,
    extra_queries: list[str] | None = None,
) -> list[dict]:
    """Return a ranked list of search results about visa requirements.

    Official government/embassy URLs are sorted to the top.
    Results mentioning the residence country are boosted above generic ones.
    Curated OFFICIAL_SOURCES for the destination are always prepended.
    extra_queries (LLM-generated) are prepended to rule-based queries so
    they are tried first and consume slots before generic fallbacks.
    Each result: {url, title, snippet, content, query, official}
    """
    max_results = max_results or config.MAX_SEARCH_RESULTS
    rule_queries = _build_queries(nationality, destination, residence, city, purpose)
    # LLM-generated queries go first — they're more targeted
    queries = list(extra_queries or []) + rule_queries
    search_hits = _run_search(queries, max_results, search_depth="advanced")

    # Drop results with no mention of the destination in URL/title/snippet —
    # these are noise that will confuse the LLM context.
    dest_lower = destination.lower()
    search_hits = [
        h for h in search_hits
        if dest_lower in (h.get("url", "") + h.get("title", "") + h.get("snippet", "")).lower()
    ]

    # Re-sort: official first, then residence-relevant, then generic.
    # This ensures e.g. "Brazilian consulate in Bern" outranks
    # "how to apply Brazil visa from China".
    if residence:
        search_hits.sort(
            key=lambda r: (r["official"], _residence_score(r, residence, city)),
            reverse=True,
        )

    return search_hits


# ── Helpers ───────────────────────────────────────────────────────────────────

def _residence_score(hit: dict, residence: str, city: str = "") -> int:
    """Return 1 if this result appears to be about applying from the residence country/city."""
    text = " ".join([
        hit.get("url", ""),
        hit.get("title", ""),
        hit.get("snippet", ""),
    ]).lower()
    if city and city.lower() in text:
        return 2
    if residence.lower() in text:
        return 1
    return 0


def _run_search(queries: list[str], max_results: int, search_depth: str = "advanced") -> list[dict]:
    """Try Tavily first; fall back to DuckDuckGo."""
    if config.TAVILY_API_KEY:
        results = _run_tavily_search(queries, max_results, search_depth)
        if results:
            return results
    return _run_ddgs_search(queries, max_results)


def _run_tavily_search(queries: list[str], max_results: int, search_depth: str) -> list[dict]:
    """Search via Tavily API; returns [] on any failure."""
    try:
        from tavily import TavilyClient
    except ImportError:
        return []

    client = TavilyClient(api_key=config.TAVILY_API_KEY)
    results: list[dict] = []
    seen_urls: set[str] = set()

    for query in queries:
        if len(results) >= max_results * 2:
            break
        try:
            response = client.search(
                query=query,
                search_depth=search_depth,
                max_results=5,
                include_raw_content=True,
            )
            for r in response.get("results", []):
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    results.append({
                        "url": url,
                        "title": r.get("title", ""),
                        "snippet": r.get("content", ""),
                        # raw_content is the full cleaned page text — skip scraping
                        "content": r.get("raw_content") or "",
                        "query": query,
                        "official": _is_official(url),
                    })
        except Exception:
            continue

    if not results:
        return []

    results.sort(key=lambda r: r["official"], reverse=True)
    return results[:max_results]


def _run_ddgs_search(queries: list[str], max_results: int) -> list[dict]:
    """Search via DuckDuckGo (free, no key required)."""
    results: list[dict] = []
    seen_urls: set[str] = set()
    errors: list[str] = []

    with DDGS() as ddgs:
        for query in queries:
            if len(results) >= max_results * 2:
                break
            try:
                for hit in ddgs.text(query, max_results=5, safesearch="off", region="wt-wt"):
                    url = hit.get("href", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        results.append(
                            {
                                "url": url,
                                "title": hit.get("title", ""),
                                "snippet": hit.get("body", ""),
                                "content": "",  # DuckDuckGo never returns full content
                                "query": query,
                                "official": _is_official(url),
                            }
                        )
            except Exception as exc:
                errors.append(str(exc))
                continue

    if errors and not results:
        raise RuntimeError(f"All search queries failed. First error: {errors[0]}")

    results.sort(key=lambda r: r["official"], reverse=True)
    return results[:max_results]


def _build_queries(
    nationality: str, destination: str, residence: str, city: str, purpose: str
) -> list[str]:
    nat = nationality.lower()
    dest = destination.lower()
    res = residence.lower()
    # City beats capital: if the user told us they live in Geneva, use Geneva
    location = city.lower() if city else _CAPITALS.get(res, "")

    queries = []

    if location:
        # 1. Find the exact consulate / visa-centre page for this city
        queries += [
            f"{dest} consulate general {location} official website visa",
            f"{dest} embassy {location} visa application {nat} how to apply",
        ]
        # 2. Step-by-step application procedure for this nationality + city
        queries += [
            f"how to apply {dest} {purpose} visa {nat} passport {location} step by step",
            f"{dest} visa {location} required documents checklist {nat} {purpose}",
            f"{dest} visa fee {location} {nat} payment method amount",
            f"{dest} visa appointment booking {location} {nat}",
        ]

    if res and res != location:
        # Country-level fallback (captures e.g. national consulate pages)
        queries += [
            f"{dest} consulate {res} {nat} visa application official",
            f"{dest} visa {nat} apply from {res} {purpose} documents fee",
        ]

    if res:
        # Anchor to residence so results are about applying *from* that country
        queries += [
            f"{dest} {purpose} visa application {nat} living in {res} procedure",
            f"{dest} MFA consulate {res} {nat} visa",
        ]
    else:
        queries += [
            f"{dest} visa {nat} {purpose} application procedure official government",
            f"{dest} immigration {nat} required documents fee official",
        ]

    # Local-language query: official consulate pages are often in the
    # destination country's language and won't appear in English searches.
    local = _DEST_LOCAL_TERMS.get(dest, "")
    if local and location:
        queries.append(f"{local} {location}")
    elif local and res:
        queries.append(f"{local} {res}")

    return queries


def _is_official(url: str) -> bool:
    lower = url.lower()
    return any(token in lower for token in _OFFICIAL_DOMAINS)
