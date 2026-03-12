"""Web search module – targets official government and embassy sources."""

import config
from ddgs import DDGS

# Domains considered authoritative for visa info
_OFFICIAL_DOMAINS = (
    ".gov", ".gob", ".gouv", ".gc.ca", ".gov.uk", ".gov.au",
    ".admin.ch", ".europa.eu", "embassy", "consulate", "immigration",
    "mfa.", "mofa.", "mfaic.", "travel.state.gov",
)


def search_topic(
    topic: str,
    nationality: str,
    destination: str,
    residence: str = "",
    max_results: int | None = None,
) -> list[dict]:
    """Search for a specific topic within the context of a visa application.

    Used when the user wants fresh information on a particular aspect
    (e.g. "appointment booking", "bank statement requirements").
    """
    max_results = max_results or config.MAX_SEARCH_RESULTS
    queries = [
        f"{topic} {destination} visa application {residence}".strip(),
        f"{destination} embassy {residence} {topic}".strip(),
        f"{topic} {nationality} {destination} visa official",
    ]
    return _run_search(queries, max_results)


def search_visa_info(
    nationality: str,
    destination: str,
    residence: str = "",
    purpose: str = "tourism",
    max_results: int | None = None,
) -> list[dict]:
    """Return a ranked list of search results about visa requirements.

    Official government/embassy URLs are sorted to the top.
    Each result: {url, title, snippet, query}
    """
    max_results = max_results or config.MAX_SEARCH_RESULTS
    queries = _build_queries(nationality, destination, residence, purpose)
    return _run_search(queries, max_results)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_search(queries: list[str], max_results: int) -> list[dict]:
    results: list[dict] = []
    seen_urls: set[str] = set()

    with DDGS() as ddgs:
        for query in queries:
            if len(results) >= max_results * 2:
                break
            try:
                for hit in ddgs.text(query, max_results=4, safesearch="off", region="wt-wt"):
                    url = hit.get("href", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        results.append(
                            {
                                "url": url,
                                "title": hit.get("title", ""),
                                "snippet": hit.get("body", ""),
                                "query": query,
                                "official": _is_official(url),
                            }
                        )
            except Exception:
                continue

    results.sort(key=lambda r: r["official"], reverse=True)
    return results[:max_results]



def _build_queries(
    nationality: str, destination: str, residence: str, purpose: str
) -> list[str]:
    nat = nationality.lower()
    dest = destination.lower()
    res = residence.lower()
    queries = []
    if res:
        # Residence-first queries: find the embassy/consulate in the applicant's
        # country of residence, which is the actual application point
        queries += [
            f"{dest} embassy {res} visa application {nat}",
            f"apply {dest} visa from {res} {nat} citizens official",
            f"{dest} consulate {res} {nat} passport visa requirements",
        ]
    queries += [
        f"visa requirements {nat} citizens {dest} official government",
        f"{dest} visa {nat} entry requirements {purpose}",
        f"{dest} immigration {nat} official",
    ]
    return queries


def _is_official(url: str) -> bool:
    lower = url.lower()
    return any(token in lower for token in _OFFICIAL_DOMAINS)
