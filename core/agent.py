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

                result = self._run_research(query)
                self.display.show_result(result)

                self._followup_loop(query)

                if not self.display.ask_new_query():
                    break

            except KeyboardInterrupt:
                self.display.show_goodbye()
                return

        self.display.show_goodbye()

    # ── Input collection ──────────────────────────────────────────────────────

    def _collect_inputs(self) -> VisaQuery:
        self.display.show_section("Travel Information")

        nationality = self.display.prompt("Your nationality  (e.g. Chinese, German)")
        residence = self.display.prompt("Country of residence  (e.g. Switzerland, USA)")
        destination = self.display.prompt("Destination country  (e.g. Brazil, Japan)")
        purpose = self.display.prompt_choice(
            "Travel purpose",
            choices=["tourism", "business", "study", "transit", "other"],
            default="tourism",
        )
        departure_date = self.display.prompt(
            "Planned departure date  (e.g. 10 August 2025)"
        )
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
            destination=destination,
            purpose=purpose,
            departure_date=departure_date,
            duration_of_stay=duration,
            residence_permit=residence_permit or "",
            entry_type=entry_type,
            companions=companions or "",
        )

    # ── Research pipeline ─────────────────────────────────────────────────────

    def _run_research(self, query: VisaQuery) -> dict:
        self.display.show_research_header(query)
        with self.display.progress_context() as tracker:
            result = self.workflow.run(query, on_progress=tracker.update)
        return result

    # ── Follow-up loop ────────────────────────────────────────────────────────

    def _followup_loop(self, query: VisaQuery) -> None:
        self.display.show_followup_hint()

        while True:
            question = self.display.prompt_followup()
            if not question or question.lower() in {"exit", "quit", "done", "q", "no"}:
                break

            topic, wants_search = _parse_search_intent(question)

            if wants_search:
                self.display.show_searching_more(topic)
                with self.display.progress_context() as tracker:
                    answer, sources = self.workflow.search_and_answer(
                        topic, query, on_progress=tracker.update
                    )
                self.display.show_followup_answer(topic, answer, sources=sources)
            else:
                with self.display.thinking_context():
                    answer = self.workflow.answer_followup(question, query)
                self.display.show_followup_answer(question, answer)


# ── Helpers ───────────────────────────────────────────────────────────────────

_SEARCH_PREFIXES = (
    "search:", "search for", "find:", "find more", "find about",
    "look up", "lookup", "search more", "more info", "more about",
    "research:", "fetch:", "get more",
)


def _parse_search_intent(text: str) -> tuple[str, bool]:
    """Return (clean_topic, wants_web_search).

    A question that starts with a recognised trigger prefix is treated as an
    explicit request for a new web search.  The prefix is stripped so the
    remaining text is used as the actual search topic.
    """
    lower = text.lower().strip()
    for prefix in _SEARCH_PREFIXES:
        if lower.startswith(prefix):
            topic = text[len(prefix):].strip(" :") or text
            return topic, True
    return text, False
