"""Web scraping module – fetches pages and extracts clean text.

Extraction cascade for each URL:
  1. Trafilatura  – fast HTML parser, great for static pages.
  2. Jina Reader  – free cloud renderer (r.jina.ai), handles JS-heavy pages
                    such as VFS Global, TLSContact, and embassy portals that
                    Trafilatura cannot process.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
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
_JINA_HEADERS = {
    "User-Agent": _HEADERS["User-Agent"],
    "Accept": "text/markdown",
    # Increase Jina timeout allowance; it needs to render JS
    "X-Timeout": "20",
}
_MIN_TEXT_LENGTH = 150
_REQUEST_TIMEOUT = 15   # seconds for direct HTTP fetch
_JINA_TIMEOUT = 25      # seconds for Jina Reader (renders JS)
_MAX_WORKERS = 5

# trafilatura calls lxml (a C extension) which is not safe to invoke
# from multiple threads simultaneously — guard it with a mutex so HTTP
# requests remain concurrent while extraction is serialized.
_extract_lock = threading.Lock()


def scrape_page(url: str) -> Optional[dict]:
    """Fetch *url* and extract clean text.

    Tries Trafilatura first; falls back to Jina Reader for JS-heavy pages.

    Returns:
        dict with keys (url, text, title) on success, or None on failure.
    """
    result = _scrape_with_trafilatura(url)
    if result is not None:
        return result
    # Trafilatura failed (JS-rendered site, bot block, etc.) → try Jina Reader
    return _scrape_with_jina(url)


def scrape_multiple(
    urls: list[str],
    max_pages: int | None = None,
) -> tuple[list[dict], list[str]]:
    """Concurrently scrape *urls*.

    Args:
        urls:      List of URLs to scrape.
        max_pages: Cap on number of pages (defaults to config value).

    Returns:
        Tuple of (successful_pages, failed_urls).
    """
    limit = max_pages or config.MAX_PAGES_PER_QUERY
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped = [u for u in urls if u not in seen and not seen.add(u)]  # type: ignore[func-returns-value]
    urls_to_fetch = deduped[:limit]

    if not urls_to_fetch:
        return [], []

    results: list[dict] = []
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=min(len(urls_to_fetch), _MAX_WORKERS)) as executor:
        future_to_url = {executor.submit(scrape_page, url): url for url in urls_to_fetch}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            page = future.result()
            if page:
                results.append(page)
            else:
                failed.append(url)

    return results, failed


# ── Extraction backends ───────────────────────────────────────────────────────

def _scrape_with_trafilatura(url: str) -> Optional[dict]:
    """Primary extractor: Trafilatura over raw HTML (fast, no JS rendering)."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        html = resp.text

        with _extract_lock:
            text = trafilatura.extract(
                html,
                include_links=False,
                include_images=False,
                include_tables=True,
                no_fallback=False,
                favor_precision=False,
            )
            title = _extract_title(html)

        if not text or len(text.strip()) < _MIN_TEXT_LENGTH:
            return None

        return {"url": url, "text": text.strip(), "title": title or url}

    except requests.RequestException:
        return None
    except Exception:
        return None


def _scrape_with_jina(url: str) -> Optional[dict]:
    """Fallback extractor: Jina Reader (r.jina.ai) renders JS and returns
    clean Markdown — handles VFS Global, TLSContact, embassy portals, etc.
    Free, no API key required.
    """
    try:
        jina_url = f"https://r.jina.ai/{url}"
        resp = requests.get(jina_url, headers=_JINA_HEADERS, timeout=_JINA_TIMEOUT)
        resp.raise_for_status()
        text = resp.text.strip()

        if not text or len(text) < _MIN_TEXT_LENGTH:
            return None

        # Jina Reader usually starts the response with "# Page Title"
        lines = text.splitlines()
        title = lines[0].lstrip("# ").strip() if lines else url

        return {"url": url, "text": text, "title": title}

    except Exception:
        return None


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
