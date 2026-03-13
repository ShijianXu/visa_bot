"""Visa research workflow: search → scrape → store → RAG → re-rank → guide."""

import threading
from typing import Callable, Iterator, Optional

import config
from knowledge.models import VisaDocument, VisaQuery
from knowledge.store import KnowledgeStore
from llm.base import LLMProvider
from retrieval.scraper import scrape_multiple
from retrieval.searcher import search_topic, search_visa_info

# ── Cross-encoder re-ranker (lazy init, optional) ────────────────────────────
# Uses flashrank (ONNX, no GPU required).  Silently skipped if not installed
# or if USE_RERANKER=false.

_reranker = None
_reranker_ready = False
_reranker_lock = threading.Lock()


def _get_reranker():
    global _reranker, _reranker_ready
    if not _reranker_ready:
        with _reranker_lock:
            if not _reranker_ready:
                try:
                    from flashrank import Ranker
                    _reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
                except Exception:
                    _reranker = None
                _reranker_ready = True
    return _reranker


def _rerank_docs(query: str, docs: list[dict]) -> list[dict]:
    """Re-rank retrieved docs using a cross-encoder; no-op if unavailable."""
    if not config.USE_RERANKER or len(docs) <= 1:
        return docs
    ranker = _get_reranker()
    if ranker is None:
        return docs
    try:
        from flashrank import RerankRequest
        passages = [{"id": i, "text": doc["content"]} for i, doc in enumerate(docs)]
        results = ranker.rerank(RerankRequest(query=query, passages=passages))
        return [docs[r["id"]] for r in results]
    except Exception:
        return docs

# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a professional visa application assistant with deep expertise in \
international travel requirements. You extract specific, actionable facts from \
official government sources and present them in clear, structured guides. \
You always cite the source URL next to each key fact, note when sources \
conflict, and direct users to the official embassy for final confirmation."""

# Used before storing a scraped page — strips boilerplate, keeps only
# visa-relevant content. Cap input at 5 000 chars to control token use.
_PREPROCESS_SYSTEM = """\
You are a visa information extractor. Extract ONLY visa-relevant content from \
the scraped page and output a structured factual record.

VERBATIM (never paraphrase): URLs, exact fees with currency, bank/payment \
details, physical addresses, phone/email, opening hours.

Extract these sections if present:
- APPLICATION PORTAL: exact URL, registration steps, service used (VFS/TLS/gov)
- DOCUMENTS: each document name + format/spec (copies, photo size, translation)
- SUBMISSION: online URL / in-person address+hours / mail address
- FEE & PAYMENT: exact amount, payment method, refundable on refusal?
- APPOINTMENT: booking URL or phone, lead time
- PROCESSING TIME: standard days/weeks, expedited option
- VISA TYPES: categories, validity, entry type, nationality restrictions

Discard navigation, cookie banners, and unrelated content.
If NO visa info exists, reply: NOT_VISA_RELEVANT"""

_GUIDE_TEMPLATE = """\
TRAVELER PROFILE
  • Nationality : {nationality}
  • Residence   : {residence} — {city_of_residence}
  • Destination : {destination}
  • Purpose     : {purpose}
  • Departure   : {departure_date}  Duration: {duration}  Entry: {entry_type}
{companions_line}
SOURCES
{context}

TASK: Using ONLY the sources above, write a step-by-step visa application guide.

Rules:
1. Quote EXACT URLs — never write "visit the official website". If missing: ⚠ URL not in sources — see [Source N]
2. Quote EXACT fees with currency. If missing: ⚠ Fee not in sources — see [Source N]
3. List EVERY required document by name with format/spec.
4. Cite every fact as [Source N].
5. If sources conflict: ⚠ Sources disagree — confirm with consulate.

## 1. Visa Requirement
Visa-free / on-arrival / e-visa / embassy visa for {nationality} → {destination} ({purpose})?

## 2. Application Portal
Exact URL and any account-registration steps.

## 3. Required Documents
Numbered list: name, format, spec (copies, photo size, translation, validity).

## 4. How to Submit
Online URL / VFS or in-person address + hours + booking URL / mail address.

## 5. Fee & Payment
Exact amount, payment method, refundable on refusal?

## 6. Processing Time
Standard days/weeks. Expedited option. Recommended apply-by date for {departure_date}.

## 7. After Submission
Confirmation, biometrics, passport return, e-visa PDF.

## 8. Sources
All URLs cited."""

_FOLLOWUP_TEMPLATE = """\
TRAVELER SITUATION
  • Nationality : {nationality}
  • Residence   : {residence}
  • Destination : {destination}
  • Purpose     : {purpose}
  • Departure   : {departure_date}

QUESTION
{question}

════════════════════════════════════════
RETRIEVED SOURCE CONTEXT
════════════════════════════════════════
{context}
════════════════════════════════════════

Answer this specific question. The traveler holds a {nationality} passport \
and applies at the {destination} consulate in {residence}.

Rules:
- Quote EXACT URLs, amounts, addresses, and phone numbers verbatim from \
the sources. Never paraphrase them.
- Cite every fact as [Source N] immediately after the fact.
- If the answer requires a URL or fee that is NOT in the sources, write: \
  ⚠ Not found in retrieved sources — try: search: {question}
- Never write vague instructions like "visit the official website"."""

_FALLBACK_TEMPLATE = """\
Provide a general visa guidance overview for:
  • Nationality : {nationality}
  • Destination : {destination}
  • Purpose     : {purpose}
  • Departure   : {departure_date}

Use the same section structure as a standard visa guide. \
State clearly that this is based on general knowledge and the user should \
verify all details with the official embassy."""

_QUERY_GEN_SYSTEM = """\
You are a search query strategist specialised in visa applications. \
You know which countries use VFS Global, TLS Contact, BLS International, \
or direct consulate applications; which ones have online e-visa portals; \
and how embassies typically structure their documentation pages."""

_QUERY_GEN_TEMPLATE = """\
Generate search queries to find official visa application information for:
  • Nationality : {nationality}
  • Lives in    : {city_of_residence}, {residence}
  • Destination : {destination}
  • Purpose     : {purpose}

Use your knowledge of {destination}'s visa system. For example:
- If {destination} uses VFS Global or TLS Contact for applicants in \
{residence}, name them explicitly in the queries.
- Target {city_of_residence} as the application city (that is where the \
consulate or visa centre is).
- Cover all of: online application portal URL, required documents checklist, \
visa fee and exact payment method, appointment booking, document drop-off \
or mailing address.

Output ONLY the search queries, one per line. \
No numbering, no bullets, no explanation."""


# ── Workflow ──────────────────────────────────────────────────────────────────

ProgressCallback = Callable[[str, str], None]


class VisaWorkflow:
    def __init__(self, llm: LLMProvider, store: KnowledgeStore) -> None:
        self.llm = llm
        self.store = store

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def prepare(
        self,
        query: VisaQuery,
        on_progress: Optional[ProgressCallback] = None,
    ) -> dict:
        """Run steps 1–5: cache check, web search, scrape, store, RAG retrieval.

        Returns a dict with keys:
          is_fallback  bool
          docs         list[dict]   (empty if is_fallback)
          sources      list[str]
          from_cache   bool
          failed_urls  list[str]
          guide        str          (only when is_fallback=True)
          docs_count   int
        """

        def step(name: str, detail: str = "") -> None:
            if on_progress:
                on_progress(name, detail)

        failed_urls: list[str] = []

        # 1 ── Cache check ─────────────────────────────────────────────────────
        step("cache_check", f"{query.nationality} → {query.destination}")
        from_cache = self.store.has_recent_data(query.nationality, query.destination)

        if not from_cache:
            # 2 ── Web search ──────────────────────────────────────────────────
            step("query_gen", f"{query.nationality} → {query.destination} …")
            llm_queries = self._generate_search_queries(query)

            step("searching", "querying official sources …")
            try:
                hits = search_visa_info(
                    query.nationality, query.destination,
                    query.residence, query.city_of_residence, query.purpose,
                    extra_queries=llm_queries,
                )
            except RuntimeError:
                hits = []

            if not hits:
                step("fallback", "no web results – using LLM knowledge")
                return self._fallback_result(query)

            # 3 ── Scrape pages ────────────────────────────────────────────────
            # Tavily results include raw_content (full page text) so we don't
            # need to scrape those URLs — only scrape what Tavily didn't fetch.
            pre_fetched: list[dict] = []
            urls_to_scrape: list[str] = []
            for hit in hits:
                content = (hit.get("content") or "").strip()
                if len(content) >= 150:
                    pre_fetched.append({
                        "url": hit["url"],
                        "text": content,
                        "title": hit.get("title", ""),
                    })
                else:
                    urls_to_scrape.append(hit["url"])

            scrape_label = f"fetching {len(urls_to_scrape)} pages"
            if pre_fetched:
                scrape_label += f" ({len(pre_fetched)} already fetched via Tavily)"
            step("scraping", scrape_label + " …")
            scraped_pages, failed_urls = scrape_multiple(urls_to_scrape)
            pages = pre_fetched + scraped_pages

            # 4 ── Store ───────────────────────────────────────────────────────
            # Pre-process pages with LLM (if enabled) then index them.
            preprocess_label = " (preprocessing …)" if config.PREPROCESS_DOCS else ""
            step("storing", f"indexing {len(pages)} documents{preprocess_label}")
            self.store.evict_stale(query.nationality, query.destination)
            for page in pages:
                page = self._preprocess_page(page)
                hit = next((h for h in hits if h["url"] == page["url"]), {})
                doc = VisaDocument(
                    content=page["text"],
                    source_url=page["url"],
                    origin_country=query.nationality,
                    destination_country=query.destination,
                    page_title=hit.get("title") or page.get("title", ""),
                )
                self.store.add_document(doc, _evict=False)

            # For pages that could not be scraped (JS-heavy, bot-blocked, etc.),
            # fall back to the DuckDuckGo snippet — it's short but often contains
            # key facts like address, phone, and eligibility conditions.
            failed_set = set(failed_urls)
            for hit in hits:
                if hit["url"] in failed_set and len(hit.get("snippet", "").strip()) > 60:
                    doc = VisaDocument(
                        content=f"{hit['title']}\n\n{hit['snippet']}",
                        source_url=hit["url"],
                        origin_country=query.nationality,
                        destination_country=query.destination,
                        page_title=hit.get("title", ""),
                    )
                    self.store.add_document(doc)

            if not pages and not any(
                len(h.get("snippet", "").strip()) > 60 for h in hits if h["url"] in failed_set
            ):
                step("fallback", "all pages failed to load – using LLM knowledge")
                return self._fallback_result(query, failed_urls=failed_urls)
        else:
            step("cache_hit", "fresh data found in knowledge base")

        # 5 ── RAG retrieval ───────────────────────────────────────────────────
        step("retrieving", "searching knowledge base …")
        rag_query = (
            f"visa application form portal documents fee payment appointment booking "
            f"{query.nationality} {query.destination} {query.purpose} {query.residence}"
        )
        docs = self.store.search(
            query=rag_query,
            origin=query.nationality,
            destination=query.destination,
            n_results=8,  # fetch more so re-ranker has more to work with
        )

        if not docs:
            return self._fallback_result(query, failed_urls=failed_urls)

        # 6 ── Re-rank ─────────────────────────────────────────────────────────
        step("reranking", "scoring relevance …")
        docs = _rerank_docs(rag_query, docs)
        docs = docs[:4]  # keep top-4 after re-ranking

        sources = list({d["metadata"]["source_url"] for d in docs})
        return {
            "is_fallback": False,
            "docs": docs,
            "sources": sources,
            "from_cache": from_cache,
            "failed_urls": failed_urls,
            "docs_count": len(docs),
        }

    def generate_guide_stream(
        self, query: VisaQuery, docs: list[dict]
    ) -> Iterator[str]:
        """Stream guide tokens for the given retrieved docs."""
        prompt = self._build_guide_prompt(query, docs)
        yield from self.llm.chat_stream(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

    # ── Follow-up ─────────────────────────────────────────────────────────────

    def answer_followup(
        self,
        question: str,
        query: VisaQuery,
        history: Optional[list[dict]] = None,
    ) -> str:
        """Answer a follow-up question via RAG, preserving conversation history."""
        docs = self.store.search(
            query=question,
            origin=query.nationality,
            destination=query.destination,
            n_results=6,  # fetch more, trim after re-ranking
        )

        if not docs:
            return (
                "I don't have enough stored information to answer that. "
                "Please verify directly with the official embassy website."
            )

        docs = _rerank_docs(question, docs)
        docs = docs[:4]

        context = _format_context(docs)
        prompt = _FOLLOWUP_TEMPLATE.format(
            nationality=query.nationality,
            residence=query.residence,
            destination=query.destination,
            purpose=query.purpose,
            departure_date=query.departure_date,
            question=question,
            context=context,
        )

        messages = [{"role": "system", "content": _SYSTEM}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        return self.llm.chat(messages=messages, temperature=0.2, max_tokens=2048)

    def search_and_answer(
        self,
        question: str,
        query: VisaQuery,
        history: Optional[list[dict]] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> tuple[str, list[str]]:
        """Run a fresh web search for *question*, store results, then answer.

        Returns (answer_text, list_of_new_source_urls).
        """

        def step(name: str, detail: str = "") -> None:
            if on_progress:
                on_progress(name, detail)

        step("searching", f"searching: {question[:60]} …")
        try:
            hits = search_topic(question, query.nationality, query.destination, query.residence, query.city_of_residence)
        except RuntimeError:
            hits = []

        if not hits:
            return (
                "I couldn't find additional web results for that topic. "
                "Please check the official embassy website directly.",
                [],
            )

        # Respect Tavily pre-fetched content for follow-up searches too
        pre_fetched_fu: list[dict] = []
        urls_to_scrape_fu: list[str] = []
        for hit in hits:
            content = (hit.get("content") or "").strip()
            if len(content) >= 150:
                pre_fetched_fu.append({"url": hit["url"], "text": content, "title": hit.get("title", "")})
            else:
                urls_to_scrape_fu.append(hit["url"])

        step("scraping", f"fetching {len(urls_to_scrape_fu)} pages …")
        scraped_fu, failed = scrape_multiple(urls_to_scrape_fu)
        pages = pre_fetched_fu + scraped_fu

        if pages:
            preprocess_label = " (preprocessing …)" if config.PREPROCESS_DOCS else ""
            step("storing", f"indexing {len(pages)} new documents{preprocess_label} …")
            self.store.evict_stale(query.nationality, query.destination)
            for page in pages:
                page = self._preprocess_page(page)
                hit = next((h for h in hits if h["url"] == page["url"]), {})
                doc = VisaDocument(
                    content=page["text"],
                    source_url=page["url"],
                    origin_country=query.nationality,
                    destination_country=query.destination,
                    page_title=hit.get("title") or page.get("title", ""),
                )
                self.store.add_document(doc, _evict=False)

        step("generating", "composing answer …")
        answer = self.answer_followup(question, query, history=history)
        new_sources = [p["url"] for p in pages]
        return answer, new_sources

    # ── Private ───────────────────────────────────────────────────────────────

    def _generate_search_queries(self, query: VisaQuery) -> list[str]:
        """Ask the LLM to generate targeted search queries for this application.

        Falls back to [] on any failure so the rule-based queries still run.
        """
        if not config.USE_LLM_QUERY_GEN:
            return []
        try:
            city_display = query.city_of_residence or query.residence
            prompt = _QUERY_GEN_TEMPLATE.format(
                nationality=query.nationality,
                city_of_residence=city_display,
                residence=query.residence,
                destination=query.destination,
                purpose=query.purpose,
            )
            raw = self.llm.chat(
                messages=[
                    {"role": "system", "content": _QUERY_GEN_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=400,
            )
            queries = [
                # Strip common list prefixes the LLM might add despite instructions
                line.strip().lstrip("0123456789.-•) ")
                for line in raw.splitlines()
                if line.strip() and len(line.strip()) > 10
            ]
            return queries[:10]
        except Exception:
            return []

    def _preprocess_page(self, page: dict) -> dict:
        """Use the LLM to strip boilerplate and keep only visa-relevant content.

        Skipped when PREPROCESS_DOCS=false or the page is too short to benefit.
        Input is capped at 5 000 chars to limit token usage.
        """
        if not config.PREPROCESS_DOCS or len(page["text"]) < 1500:
            return page
        try:
            raw = page["text"][:5000]
            cleaned = self.llm.chat(
                messages=[
                    {"role": "system", "content": _PREPROCESS_SYSTEM},
                    {"role": "user", "content": f"Webpage from: {page['url']}\n\n{raw}"},
                ],
                temperature=0,
                max_tokens=1500,
            )
            if cleaned.strip() == "NOT_VISA_RELEVANT":
                return page  # keep original; still indexed in case RAG finds use
            return {**page, "text": cleaned.strip()}
        except Exception:
            return page  # never lose data on preprocessing failure

    def _build_guide_prompt(self, query: VisaQuery, docs: list[dict]) -> str:
        context = _format_context(docs)
        companions_line = (
            f"  • Companions        : {query.companions}" if query.companions else ""
        )
        city_display = query.city_of_residence or query.residence
        return _GUIDE_TEMPLATE.format(
            nationality=query.nationality,
            residence=query.residence,
            city_of_residence=city_display,
            destination=query.destination,
            purpose=query.purpose,
            departure_date=query.departure_date,
            duration=query.duration_of_stay,
            entry_type=query.entry_type,
            companions_line=companions_line,
            context=context,
        )

    def _fallback_result(
        self, query: VisaQuery, failed_urls: Optional[list[str]] = None
    ) -> dict:
        prompt = _FALLBACK_TEMPLATE.format(
            nationality=query.nationality,
            destination=query.destination,
            purpose=query.purpose,
            departure_date=query.departure_date,
        )
        guide = self.llm.chat(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return {
            "is_fallback": True,
            "guide": guide,
            "docs": [],
            "sources": [],
            "from_cache": False,
            "failed_urls": failed_urls or [],
            "docs_count": 0,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_context(docs: list[dict], max_chars_per_doc: int = 1500) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc["metadata"]
        url = meta.get("source_url", "unknown")
        title = meta.get("page_title", "").strip()
        header = f"[Source {i}] {title}\n{url}" if title else f"[Source {i}] {url}"
        body = doc["content"][:max_chars_per_doc].strip()
        parts.append(f"{header}\n{'-' * 40}\n{body}")
    return "\n\n".join(parts)
