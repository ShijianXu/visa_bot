"""Web scraping module – fetches pages and extracts clean text."""

import time
from typing import Optional

import requests
import trafilatura

import config

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
_MIN_TEXT_LENGTH = 150
_REQUEST_TIMEOUT = 15  # seconds


def scrape_page(url: str) -> Optional[dict]:
    """Fetch *url* and extract clean text via trafilatura.

    Returns:
        dict with keys (url, text, title) or None on failure.
    """
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()

        text = trafilatura.extract(
            resp.text,
            include_links=False,
            include_images=False,
            include_tables=True,
            no_fallback=False,
            favor_precision=False,
        )

        if not text or len(text.strip()) < _MIN_TEXT_LENGTH:
            return None

        # Try to get a title
        title = _extract_title(resp.text) or url

        return {"url": url, "text": text.strip(), "title": title}

    except Exception:
        return None


def scrape_multiple(
    urls: list[str],
    delay: float = 1.2,
    max_pages: int | None = None,
) -> list[dict]:
    """Scrape *urls* with rate-limiting.

    Args:
        urls:      List of URLs to scrape.
        delay:     Seconds to wait between requests.
        max_pages: Cap on number of pages (defaults to config value).
    """
    limit = max_pages or config.MAX_PAGES_PER_QUERY
    results: list[dict] = []

    for url in urls[:limit]:
        page = scrape_page(url)
        if page:
            results.append(page)
        time.sleep(delay)

    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_title(html: str) -> Optional[str]:
    """Quick regex-free title extraction."""
    start = html.lower().find("<title")
    if start == -1:
        return None
    end_open = html.find(">", start)
    end_close = html.lower().find("</title>", end_open)
    if end_open == -1 or end_close == -1:
        return None
    return html[end_open + 1 : end_close].strip() or None
