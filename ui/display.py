"""Rich-based terminal UI for Visa Bot."""

from contextlib import contextmanager
from typing import Iterator

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from knowledge.models import VisaQuery

# ── Shared console ────────────────────────────────────────────────────────────
console = Console(highlight=False)

# ── Theme constants ───────────────────────────────────────────────────────────
C_BRAND = "cyan"
C_SUCCESS = "green"
C_WARN = "yellow"
C_ERR = "red"
C_DIM = "dim"
C_SRC = "blue"

_BANNER = r"""
 __   ___          ___       _
 \ \ / (_)_____ _ | _ ) ___ | |_
  \ V /| (_-/ _` || _ \/ _ \|  _|
   \_/ |_/__\__,_||___/\___/ \__|
"""

_STEP_LABELS: dict[str, str] = {
    "cache_check":  "Checking knowledge base",
    "cache_hit":    "Knowledge base hit — skipping web search",
    "query_gen":    "Generating targeted search queries",
    "searching":    "Searching official sources",
    "scraping":     "Fetching pages",
    "storing":      "Indexing documents",
    "retrieving":   "Retrieving relevant context",
    "reranking":    "Re-ranking by relevance",
    "generating":   "Composing visa guide",
    "fallback":     "Falling back to LLM knowledge",
    "search_failed":"Search returned no results",
}


# ── Progress tracker ──────────────────────────────────────────────────────────

class _ProgressTracker:
    """Wraps Rich Progress for step-by-step workflow display."""

    def __init__(self) -> None:
        self._prog = Progress(
            SpinnerColumn(style=f"bold {C_BRAND}"),
            TextColumn(f"[bold {C_BRAND}]{{task.description}}"),
            TextColumn(f"[{C_DIM}]{{task.fields[detail]}}"),
            console=console,
            transient=True,
        )
        self._task_id = None

    def update(self, step: str, detail: str = "") -> None:
        label = _STEP_LABELS.get(step, step.replace("_", " ").title())
        if self._task_id is None:
            self._task_id = self._prog.add_task(label, detail=detail, total=None)
        else:
            self._prog.update(self._task_id, description=label, detail=detail)

    def __enter__(self) -> "_ProgressTracker":
        self._prog.start()
        return self

    def __exit__(self, *_) -> None:
        self._prog.stop()


# ── Display ───────────────────────────────────────────────────────────────────

class Display:
    """All terminal output for Visa Bot."""

    # ── Welcome ───────────────────────────────────────────────────────────────

    def show_welcome(self) -> None:
        console.print()
        console.print(Align.center(Text(_BANNER.strip(), style=f"bold {C_BRAND}")))
        console.print(
            Align.center(
                Text("AI-Powered Visa Application Assistant", style="bold white")
            )
        )
        console.print(Align.center(Text("─" * 44, style=f"dim {C_BRAND}")))
        console.print()
        console.print(
            Panel(
                f"[{C_BRAND}]Welcome![/{C_BRAND}]  "
                "I research [bold]official government and embassy sources[/bold] "
                "to give you accurate, step-by-step visa guidance.\n\n"
                f"[{C_DIM}]Tip: press [bold]Ctrl+C[/bold] at any time to exit.[/{C_DIM}]",
                border_style=C_BRAND,
                padding=(0, 2),
            )
        )
        console.print()

    def show_model_info(self, model: str, provider: str) -> None:
        console.print(
            f"[{C_DIM}]  Model [bold]{model}[/bold]  "
            f"via [bold]{provider.upper()}[/bold][/{C_DIM}]"
        )
        console.print()

    # ── Section headers ───────────────────────────────────────────────────────

    def show_section(self, title: str) -> None:
        console.print(Rule(f"[bold {C_BRAND}] {title} [/bold {C_BRAND}]", style=C_BRAND))
        console.print()

    def show_optional_hint(self) -> None:
        console.print()
        console.print(Rule(f"[{C_DIM}] Optional Details [/{C_DIM}]", style=C_DIM))
        console.print(f"[{C_DIM}]  Press Enter to skip any of these.[/{C_DIM}]")
        console.print()

    # ── Prompts ───────────────────────────────────────────────────────────────

    def prompt(self, label: str) -> str:
        return Prompt.ask(f"  [{C_BRAND}]{label}[/{C_BRAND}]")

    def prompt_optional(self, label: str) -> str:
        return Prompt.ask(f"  [{C_DIM}]{label}[/{C_DIM}]", default="")

    def prompt_choice(
        self, label: str, choices: list[str], default: str | None = None
    ) -> str:
        parts = []
        for c in choices:
            if c == default:
                parts.append(f"[bold {C_BRAND}]{c}[/bold {C_BRAND}]")
            else:
                parts.append(f"[{C_DIM}]{c}[/{C_DIM}]")
        choices_display = " / ".join(parts)
        result = Prompt.ask(
            f"  [{C_BRAND}]{label}[/{C_BRAND}] ({choices_display})",
            choices=choices,
            default=default or choices[0],
        )
        return result

    # ── Research flow ─────────────────────────────────────────────────────────

    def show_research_header(self, query: VisaQuery) -> None:
        console.print()
        console.print(
            Panel(
                f"[bold]Researching:[/bold]  "
                f"[{C_BRAND}]{query.nationality}[/{C_BRAND}] passport  →  "
                f"[{C_BRAND}]{query.destination}[/{C_BRAND}]  "
                f"([{C_DIM}]{query.purpose}[/{C_DIM}])",
                border_style=C_BRAND,
                padding=(0, 2),
            )
        )
        console.print()

    def progress_context(self) -> _ProgressTracker:
        return _ProgressTracker()

    @contextmanager
    def thinking_context(self) -> Iterator[None]:
        with console.status(
            f"[{C_BRAND}]Searching knowledge base and generating answer …[/{C_BRAND}]",
            spinner="dots",
        ):
            yield

    # ── Results ───────────────────────────────────────────────────────────────

    def show_result(self, result: dict) -> None:
        """Show a complete result (used for fallback guides generated synchronously)."""
        console.print()
        console.print(
            Rule(f"[bold {C_SUCCESS}] Visa Guide [/bold {C_SUCCESS}]", style=C_SUCCESS)
        )
        console.print()
        console.print(Markdown(result.get("guide", "_No guide generated._")))
        self.show_result_meta(result)

    # ── Streaming guide display ────────────────────────────────────────────────

    def show_guide_header(self) -> None:
        """Print the guide section header before streaming begins."""
        console.print()
        console.print(
            Rule(f"[bold {C_SUCCESS}] Visa Guide [/bold {C_SUCCESS}]", style=C_SUCCESS)
        )
        console.print()

    def stream_token(self, token: str) -> None:
        """Write a single streamed token directly to the console."""
        console.print(token, end="", markup=False)

    def end_stream(self) -> None:
        """Finish the streaming output with a blank line."""
        console.print()
        console.print()

    def show_result_meta(self, result: dict) -> None:
        """Show sources, scrape warnings, and retrieval metadata."""
        sources = result.get("sources") or []
        if sources:
            console.print()
            console.print(
                Rule(f"[bold {C_SRC}] Sources [/bold {C_SRC}]", style=C_SRC)
            )
            console.print()
            tbl = Table(
                box=box.SIMPLE,
                show_header=True,
                header_style=f"bold {C_SRC}",
                padding=(0, 1),
            )
            tbl.add_column("#", style=C_DIM, width=3, no_wrap=True)
            tbl.add_column("Official Source", style=C_BRAND, overflow="fold")
            for i, src in enumerate(dict.fromkeys(sources), 1):
                tbl.add_row(str(i), src)
            console.print(tbl)

        meta_parts = []
        if result.get("from_cache"):
            meta_parts.append(f"[{C_DIM}]served from cache[/{C_DIM}]")
        n = result.get("docs_count", 0)
        if n:
            meta_parts.append(f"[{C_DIM}]{n} chunks retrieved[/{C_DIM}]")
        if meta_parts:
            console.print()
            console.print("  " + "  ·  ".join(meta_parts))
        console.print()

    def show_scrape_warning(self, failed_urls: list[str]) -> None:
        """Warn the user that some pages could not be fetched."""
        console.print()
        console.print(
            Panel(
                f"[{C_WARN}]{len(failed_urls)} page(s) could not be fetched "
                f"and were skipped.[/{C_WARN}]\n"
                + "\n".join(f"  [{C_DIM}]• {u}[/{C_DIM}]" for u in failed_urls[:5]),
                title=f"[bold {C_WARN}]Scrape Warning[/bold {C_WARN}]",
                border_style=C_WARN,
                padding=(0, 2),
            )
        )
        console.print()

    # ── Follow-up ─────────────────────────────────────────────────────────────

    def show_followup_hint(self) -> None:
        console.print(
            Rule(
                f"[bold {C_WARN}] Follow-up Questions [/bold {C_WARN}]",
                style=C_WARN,
            )
        )
        console.print()
        console.print(
            Panel(
                f"[{C_WARN}]Ask anything about documents, fees, appointments, etc.[/{C_WARN}]\n\n"
                f"  [{C_DIM}]• Any question  → answered from the knowledge base[/{C_DIM}]\n"
                f"  [{C_DIM}]• [bold]search: <topic>[/bold]  → triggers a fresh web search[/{C_DIM}]\n"
                f"  [{C_DIM}]• [bold]done[/bold] / [bold]exit[/bold]      → finish[/{C_DIM}]",
                border_style=C_WARN,
                padding=(0, 2),
            )
        )
        console.print()

    def prompt_followup(self) -> str:
        return Prompt.ask(f"  [{C_WARN}]Question[/{C_WARN}]")

    def show_searching_more(self, topic: str) -> None:
        console.print()
        console.print(
            Rule(
                f"[bold {C_BRAND}] Web Search: {topic[:60]} [/bold {C_BRAND}]",
                style=C_BRAND,
            )
        )
        console.print()

    def show_followup_answer(
        self, question: str, answer: str, sources: list[str] | None = None
    ) -> None:
        console.print()
        console.print(
            Panel(
                Markdown(answer),
                title=f"[bold {C_WARN}]{question[:80]}[/bold {C_WARN}]",
                border_style=C_WARN,
                padding=(1, 2),
            )
        )
        if sources:
            tbl = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
            tbl.add_column("", style=C_DIM, width=3, no_wrap=True)
            tbl.add_column("", style=C_BRAND, overflow="fold")
            for i, src in enumerate(dict.fromkeys(sources), 1):
                tbl.add_row(f"[{i}]", src)
            console.print(f"  [{C_DIM}]New sources fetched:[/{C_DIM}]")
            console.print(tbl)
        console.print()

    # ── Navigation ────────────────────────────────────────────────────────────

    def ask_new_query(self) -> bool:
        return Confirm.ask(
            f"\n  [{C_BRAND}]Check visa requirements for another trip?[/{C_BRAND}]",
            default=False,
        )

    def show_goodbye(self) -> None:
        console.print()
        console.print(
            Panel(
                f"[bold {C_BRAND}]Thank you for using Visa Bot![/bold {C_BRAND}]\n"
                f"[{C_DIM}]Always confirm critical information with the official embassy "
                f"before travelling.[/{C_DIM}]  ✈",
                border_style=C_BRAND,
                padding=(1, 4),
            )
        )
        console.print()

    # ── Error handling ────────────────────────────────────────────────────────

    def show_error(self, message: str) -> None:
        console.print()
        console.print(
            Panel(
                f"[{C_ERR}]{message}[/{C_ERR}]",
                title=f"[bold {C_ERR}]Error[/bold {C_ERR}]",
                border_style=C_ERR,
            )
        )
        console.print()
