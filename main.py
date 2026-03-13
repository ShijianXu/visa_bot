#!/usr/bin/env python3
"""Visa Bot - AI-powered visa application assistant.

Usage:
    python main.py

Environment:
    GROQ_API_KEY   required  - your Groq Cloud API key
    LLM_PROVIDER   optional  - default: groq
    GROQ_MODEL     optional  - default: llama-3.3-70b-versatile

Copy .env.example to .env and fill in your keys before running.
"""

import sys
import warnings

# google-genai uses the built-in `any` as a Pydantic type annotation — suppress the noise.
warnings.filterwarnings("ignore", message=".*<built-in function any>.*", category=UserWarning)


def _check_env() -> None:
    """Validate required environment variables before importing heavy deps."""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    missing = []
    if not os.getenv("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY")

    if not missing:
        return

    # Import Rich only here to keep startup fast on success path
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print()
    console.print(
        Panel(
            "[red]Missing required environment variable(s):[/red]\n\n"
            + "\n".join(f"  • [yellow]{v}[/yellow]" for v in missing)
            + "\n\n[dim]Copy [bold].env.example[/bold] to [bold].env[/bold] "
            "and set your API key(s).[/dim]",
            title="[red bold]Configuration Error[/red bold]",
            border_style="red",
            padding=(1, 2),
        )
    )
    console.print()
    sys.exit(1)


def main() -> None:
    _check_env()

    try:
        from core.agent import VisaAgent

        VisaAgent().run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        from rich.console import Console

        Console().print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
