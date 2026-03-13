"""Top-level agent: wires together UI, LLM, store, and workflow."""

from knowledge.models import VisaQuery
from knowledge.store import KnowledgeStore
from llm.factory import get_provider
from core.workflow import VisaWorkflow
from ui.display import Display


class VisaAgent:
    def __init__(self) -> None:
        self.display = Display()
        self.llm = get_provider()
        self.store = KnowledgeStore()
        self.workflow = VisaWorkflow(self.llm, self.store)
        self._current_query: VisaQuery | None = None

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the interactive CLI session."""
        import config

        self.display.show_welcome()
        self.display.show_model_info(self.llm.model_name, config.LLM_PROVIDER)

        while True:
            try:
                query = self._collect_inputs()
                self._current_query = query

                initial_guide = self._run_research(query)

                self._followup_loop(query, initial_guide=initial_guide)

                if not self.display.ask_new_query():
                    break

            except KeyboardInterrupt:
                self.display.show_goodbye()
                return

        self.display.show_goodbye()

    # ── Input collection ──────────────────────────────────────────────────────

    def _collect_inputs(self) -> VisaQuery:
        self.display.show_section("Travel Information")

        nationality = _normalize_country(self.display.prompt("Your nationality  (e.g. Chinese, German)"))
        residence = _normalize_country(self.display.prompt("Country of residence  (e.g. Switzerland, USA)"))
        city_of_residence = _normalize_country(self.display.prompt("Your city of residence  (e.g. Geneva, Zurich, London)"))
        destination = _normalize_country(self.display.prompt("Destination country  (e.g. Brazil, Japan)"))
        purpose = self.display.prompt_choice(
            "Travel purpose",
            choices=["tourism", "business", "study", "transit", "other"],
            default="tourism",
        )
        departure_date = self._prompt_departure_date()
        duration = self.display.prompt(
            "Duration of stay  (e.g. 2 weeks, 30 days)"
        )

        self.display.show_optional_hint()
        residence_permit = self.display.prompt_optional(
            "Residence permit status  (if living abroad)"
        )
        entry_type = self.display.prompt_choice(
            "Number of entries",
            choices=["single", "multiple", "unknown"],
            default="single",
        )
        companions = self.display.prompt_optional(
            "Travel companions  (e.g. spouse, children)"
        )

        return VisaQuery(
            nationality=nationality,
            residence=residence,
            city_of_residence=city_of_residence,
            destination=destination,
            purpose=purpose,
            departure_date=departure_date,
            duration_of_stay=duration,
            residence_permit=residence_permit or "",
            entry_type=entry_type,
            companions=companions or "",
        )

    def _prompt_departure_date(self) -> str:
        """Prompt for a departure date, re-asking if the input cannot be parsed."""
        from dateutil import parser as dateutil_parser

        while True:
            raw = self.display.prompt("Planned departure date  (e.g. 10 August 2025)")
            try:
                dateutil_parser.parse(raw)
                return raw
            except (ValueError, OverflowError):
                self.display.show_error(
                    f"Could not understand '{raw}' as a date. "
                    "Please try again (e.g. '15 September 2025')."
                )

    # ── Research pipeline ─────────────────────────────────────────────────────

    def _run_research(self, query: VisaQuery) -> str:
        """Run the full research pipeline and return the generated guide text."""
        self.display.show_research_header(query)

        # Phase 1: cache / search / scrape / store / retrieve (with spinner)
        with self.display.progress_context() as tracker:
            prepared = self.workflow.prepare(query, on_progress=tracker.update)

        # Warn about any pages that could not be fetched
        if prepared.get("failed_urls"):
            self.display.show_scrape_warning(prepared["failed_urls"])

        if prepared["is_fallback"]:
            self.display.show_result(prepared)
            return prepared.get("guide", "")

        # Phase 2: stream the guide (spinner already closed)
        self.display.show_guide_header()
        guide_parts: list[str] = []
        for token in self.workflow.generate_guide_stream(query, prepared["docs"]):
            self.display.stream_token(token)
            guide_parts.append(token)
        guide = "".join(guide_parts)
        self.display.end_stream()

        self.display.show_result_meta(
            {
                "guide": guide,
                "sources": prepared["sources"],
                "from_cache": prepared["from_cache"],
                "docs_count": prepared["docs_count"],
                "failed_urls": prepared.get("failed_urls", []),
            }
        )
        return guide

    # ── Follow-up loop ────────────────────────────────────────────────────────

    def _followup_loop(self, query: VisaQuery, initial_guide: str = "") -> None:
        self.display.show_followup_hint()

        # Seed history with the initial guide so every follow-up has full context
        # of what was already answered, without needing to re-retrieve it.
        history: list[dict] = []
        if initial_guide:
            history.append({
                "role": "assistant",
                "content": (
                    f"I have prepared the following visa guide for your trip "
                    f"({query.nationality} → {query.destination}):\n\n{initial_guide}"
                ),
            })

        while True:
            question = self.display.prompt_followup()
            if not question or question.lower() in {"exit", "quit", "done", "q", "no"}:
                break

            topic, wants_search = _parse_search_intent(question)

            if wants_search:
                self.display.show_searching_more(topic)
                with self.display.progress_context() as tracker:
                    answer, sources = self.workflow.search_and_answer(
                        topic, query, history=history, on_progress=tracker.update
                    )
                self.display.show_followup_answer(topic, answer, sources=sources)
            else:
                with self.display.thinking_context():
                    answer = self.workflow.answer_followup(question, query, history=history)
                self.display.show_followup_answer(question, answer)

            # Append to history for the next turn
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_country(raw: str) -> str:
    """Strip whitespace and title-case the country name."""
    return " ".join(raw.strip().split()).title()


_SEARCH_PREFIXES = (
    "search:", "search for", "find:", "find more", "find about",
    "look up", "lookup", "search more", "more info", "more about",
    "research:", "fetch:", "get more", "latest on", "current on",
    "check online", "check the web", "web search",
    "google", "search online", "find online",
)

# Standalone keywords that suggest a need for fresh web data
_SEARCH_KEYWORDS = frozenset({
    "appointment", "book", "schedule", "wait time", "waiting time",
    "processing time", "vfs", "tls", "biometric", "embassy hours",
    "opening hours", "slot", "availability",
})


def _parse_search_intent(text: str) -> tuple[str, bool]:
    """Return (clean_topic, wants_web_search).

    Triggers a fresh web search when:
    - The question starts with an explicit search prefix, OR
    - The question contains a keyword that implies need for live data.
    """
    lower = text.lower().strip()

    # Explicit prefix triggers
    for prefix in _SEARCH_PREFIXES:
        if lower.startswith(prefix):
            topic = text[len(prefix):].strip(" :") or text
            return topic, True

    # Keyword heuristic for topics that change frequently
    words = set(lower.split())
    if _SEARCH_KEYWORDS & words:
        return text, True

    return text, False
