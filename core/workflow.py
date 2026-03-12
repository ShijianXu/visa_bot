"""Visa research workflow: search → scrape → store → RAG → guide."""

from typing import Callable, Optional

from knowledge.models import VisaDocument, VisaQuery
from knowledge.store import KnowledgeStore
from llm.base import LLMProvider
from retrieval.scraper import scrape_multiple
from retrieval.searcher import search_topic, search_visa_info

# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a professional visa application assistant with deep expertise in \
international travel requirements.  You provide accurate, well-structured \
information grounded in official government sources.  Always cite source URLs \
and recommend verifying critical details directly with the official embassy or \
consulate if there is any uncertainty."""

_GUIDE_TEMPLATE = """\
Using the retrieved official source information below, write a comprehensive \
visa guide for this traveler:

  • Nationality : {nationality}
  • Residence   : {residence}
  • Destination : {destination}
  • Purpose     : {purpose}
  • Departure   : {departure_date}
  • Duration    : {duration}
  • Entry type  : {entry_type}
{companions_line}
────────────────────────────────────────
RETRIEVED OFFICIAL INFORMATION
────────────────────────────────────────
{context}
────────────────────────────────────────

Structure your response with these exact sections (use markdown headers):

## Visa Requirement
## Application Timeline
## Where to Apply
## Required Documents
## Step-by-Step Application Guide
## Fees & Processing Time
## Important Notes
## Official Sources

Be specific and practical.  Quote processing times and fee amounts where found. \
If information is unavailable or uncertain, say so clearly and direct the user \
to the official embassy website."""

_FOLLOWUP_TEMPLATE = """\
Answer this follow-up question about the visa application for \
{nationality} → {destination}:

Question: {question}

────────────────────────────────────────
RETRIEVED CONTEXT
────────────────────────────────────────
{context}
────────────────────────────────────────

Give a concise, accurate answer and cite the relevant source URLs."""

_FALLBACK_TEMPLATE = """\
Provide a general visa guidance overview for:
  • Nationality : {nationality}
  • Destination : {destination}
  • Purpose     : {purpose}
  • Departure   : {departure_date}

Use the same section structure as a standard visa guide. \
State clearly that this is based on general knowledge and the user should \
verify all details with the official embassy."""


# ── Workflow ──────────────────────────────────────────────────────────────────

ProgressCallback = Callable[[str, str], None]


class VisaWorkflow:
    def __init__(self, llm: LLMProvider, store: KnowledgeStore) -> None:
        self.llm = llm
        self.store = store

    def run(
        self,
        query: VisaQuery,
        on_progress: Optional[ProgressCallback] = None,
    ) -> dict:
        """Execute the full research pipeline and return a result dict."""

        def step(name: str, detail: str = "") -> None:
            if on_progress:
                on_progress(name, detail)

        # 1 ── Cache check ─────────────────────────────────────────────────────
        step("cache_check", f"{query.nationality} → {query.destination}")
        from_cache = self.store.has_recent_data(query.nationality, query.destination)

        if not from_cache:
            # 2 ── Web search ──────────────────────────────────────────────────
            step("searching", "querying official sources …")
            hits = search_visa_info(
                query.nationality, query.destination, query.residence, query.purpose
            )

            if not hits:
                step("fallback", "no web results – using LLM knowledge")
                return self._fallback_guide(query)

            # 3 ── Scrape pages ────────────────────────────────────────────────
            step("scraping", f"fetching {len(hits)} pages …")
            pages = scrape_multiple([h["url"] for h in hits])

            # 4 ── Store ───────────────────────────────────────────────────────
            step("storing", f"indexing {len(pages)} documents …")
            for page in pages:
                title = next(
                    (h["title"] for h in hits if h["url"] == page["url"]), ""
                )
                doc = VisaDocument(
                    content=page["text"],
                    source_url=page["url"],
                    origin_country=query.nationality,
                    destination_country=query.destination,
                    page_title=title or page.get("title", ""),
                )
                self.store.add_document(doc)
        else:
            step("cache_hit", "fresh data found in knowledge base")

        # 5 ── RAG retrieval ───────────────────────────────────────────────────
        step("retrieving", "searching knowledge base …")
        docs = self.store.search(
            query=f"visa requirements {query.nationality} {query.destination} {query.purpose}",
            origin=query.nationality,
            destination=query.destination,
            n_results=6,
        )

        if not docs:
            return self._fallback_guide(query)

        # 6 ── Generate guide ──────────────────────────────────────────────────
        step("generating", "composing visa guide …")
        context = _format_context(docs)
        companions_line = (
            f"  • Companions  : {query.companions}" if query.companions else ""
        )
        prompt = _GUIDE_TEMPLATE.format(
            nationality=query.nationality,
            residence=query.residence,
            destination=query.destination,
            purpose=query.purpose,
            departure_date=query.departure_date,
            duration=query.duration_of_stay,
            entry_type=query.entry_type,
            companions_line=companions_line,
            context=context,
        )

        guide = self.llm.chat(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=2048,
        )

        sources = list({d["metadata"]["source_url"] for d in docs})
        return {
            "guide": guide,
            "sources": sources,
            "from_cache": from_cache,
            "docs_count": len(docs),
        }

    def answer_followup(self, question: str, query: VisaQuery) -> str:
        """Answer a follow-up question via RAG."""
        docs = self.store.search(
            query=question,
            origin=query.nationality,
            destination=query.destination,
            n_results=4,
        )

        if not docs:
            return (
                "I don't have enough stored information to answer that. "
                "Please verify directly with the official embassy website."
            )

        context = _format_context(docs)
        prompt = _FOLLOWUP_TEMPLATE.format(
            nationality=query.nationality,
            destination=query.destination,
            question=question,
            context=context,
        )
        return self.llm.chat(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1024,
        )

    def search_and_answer(
        self,
        question: str,
        query: VisaQuery,
        on_progress: Optional[ProgressCallback] = None,
    ) -> tuple[str, list[str]]:
        """Run a fresh web search for *question*, store results, then answer.

        Returns (answer_text, list_of_new_source_urls).
        """

        def step(name: str, detail: str = "") -> None:
            if on_progress:
                on_progress(name, detail)

        step("searching", f"searching: {question[:60]} …")
        hits = search_topic(question, query.nationality, query.destination, query.residence)

        if not hits:
            return (
                "I couldn't find additional web results for that topic. "
                "Please check the official embassy website directly.",
                [],
            )

        step("scraping", f"fetching {len(hits)} pages …")
        pages = scrape_multiple([h["url"] for h in hits])

        step("storing", f"indexing {len(pages)} new documents …")
        for page in pages:
            title = next(
                (h["title"] for h in hits if h["url"] == page["url"]), ""
            )
            doc = VisaDocument(
                content=page["text"],
                source_url=page["url"],
                origin_country=query.nationality,
                destination_country=query.destination,
                page_title=title or page.get("title", ""),
            )
            self.store.add_document(doc)

        step("generating", "composing answer …")
        answer = self.answer_followup(question, query)
        new_sources = [p["url"] for p in pages]
        return answer, new_sources

    # ── Private ───────────────────────────────────────────────────────────────

    def _fallback_guide(self, query: VisaQuery) -> dict:
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
        return {"guide": guide, "sources": [], "from_cache": False, "docs_count": 0}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_context(docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        url = doc["metadata"].get("source_url", "unknown")
        snippet = doc["content"][:1200].strip()
        parts.append(f"[{i}] {url}\n{snippet}")
    return "\n\n".join(parts)
