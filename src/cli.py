"""Interactive CLI for the home buyer's research assistant.

Run with:
    python -m src.cli

Environment:
  OPENAI_API_KEY     required
  CENSUS_API_KEY     required for the neighborhood agent
  DEBUG=1            show classifications and raw agent output
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

# Load .env before anything else imports modules that read env vars.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.router.graph import build_workflow  # noqa: E402
from src.streaming import stream_and_render  # noqa: E402


console = Console()


EXAMPLES = [
    "Show me 3BR homes under $500k in Austin with good schools",
    "What's the median home price in 78704?",
    "Best school districts in Nashville under $600k",
    "Show me 4BR homes in 80210",
    "Is 37206 a family-friendly ZIP?",
]


def _preflight() -> bool:
    """Check required env vars and database. Returns True if ready."""
    problems: list[str] = []
    if not os.environ.get("OPENAI_API_KEY"):
        problems.append("OPENAI_API_KEY is not set.")
    if not os.environ.get("CENSUS_API_KEY"):
        problems.append(
            "CENSUS_API_KEY is not set. "
            "The neighborhood agent will fail. "
            "Get a free key at https://api.census.gov/data/key_signup.html"
        )
    db_path = Path(__file__).resolve().parent.parent / "data" / "housing.db"
    if not db_path.exists():
        problems.append(
            f"Database not found at {db_path}. "
            f"Run: python scripts/build_database.py"
        )
    if problems:
        console.print(Panel(
            "\n".join(f"• {p}" for p in problems),
            title="[red]Setup incomplete[/red]",
            border_style="red",
        ))
        return False
    return True


def _split_brief_and_shortlist(answer: str) -> tuple[str, list[dict]]:
    """Pull the embedded SHORTLIST_JSON payload out of the brief text."""
    match = re.search(r"<!--SHORTLIST_JSON:(.*?)-->", answer, re.DOTALL)
    if not match:
        return answer, []
    brief_text = answer[: match.start()].rstrip()
    try:
        shortlist = json.loads(match.group(1))
    except json.JSONDecodeError:
        shortlist = []
    return brief_text, shortlist


def _render_shortlist(shortlist: list[dict]) -> None:
    if not shortlist:
        return
    table = Table(title="Ranked shortlist", show_lines=False, header_style="bold cyan")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("ID", justify="right", style="dim")
    table.add_column("Address")
    table.add_column("Location")
    table.add_column("Price", justify="right")
    table.add_column("Beds/Baths", justify="center")
    table.add_column("Sqft", justify="right")
    for s in shortlist:
        table.add_row(
            f"{s['_score']:.0f}",
            str(s["listing_id"]),
            s["address"],
            f"{s['city']}, {s['state']} {s['zip_code']}",
            f"${s['price']:,}",
            f"{s['bedrooms']}/{s['bathrooms']}",
            f"{s['sqft']:,}",
        )
    console.print(table)


def _render_classifications(classifications: list[dict]) -> None:
    table = Table(title="Router decision", show_header=True, header_style="bold magenta")
    table.add_column("Source")
    table.add_column("Sub-query")
    for c in classifications:
        table.add_row(c["source"], c["query"])
    console.print(table)


def _handle_query(workflow, query: str) -> None:
    console.print(Rule(style="dim"))
    console.print(f"[bold]Query:[/bold] {query}")
    debug = os.environ.get("DEBUG") == "1"

    start = time.perf_counter()
    try:
        result = stream_and_render(workflow, query, console)
    except Exception as e:
        console.print(Panel(str(e), title="[red]Error[/red]", border_style="red"))
        return
    elapsed = time.perf_counter() - start

    if debug:
        for r in result.get("results", []):
            console.print(Panel(
                r["result"],
                title=f"Raw output: {r['source']}",
                border_style="dim",
            ))

    brief_text, shortlist = _split_brief_and_shortlist(result.get("final_answer", ""))
    console.print(Panel(Markdown(brief_text), title="Buyer's brief", border_style="green"))
    _render_shortlist(shortlist)
    console.print(f"[dim]Total time: {elapsed:.1f}s[/dim]")


def _print_help() -> None:
    console.print(Panel(
        "Commands:\n"
        "  /help      Show this help\n"
        "  /examples  Show example queries you can try\n"
        "  /quit      Exit (also Ctrl+D / Ctrl+C)\n\n"
        "Anything else is treated as a home-buying research question.",
        title="Help",
        border_style="cyan",
    ))


def _print_examples() -> None:
    console.print(Panel(
        "\n".join(f"  • {e}" for e in EXAMPLES),
        title="Example queries",
        border_style="cyan",
    ))


def main() -> int:
    console.print(Panel.fit(
        "[bold cyan]US Home Buyer's Research Assistant[/bold cyan]\n"
        "Multi-agent system using the LangChain Router pattern.\n"
        "Type [bold]/help[/bold] for commands, [bold]/quit[/bold] to exit.",
        border_style="cyan",
    ))
    if not _preflight():
        return 1

    with console.status("Compiling the router graph..."):
        workflow = build_workflow()

    while True:
        try:
            query = console.input("[bold cyan]> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return 0
        if not query:
            continue
        if query in ("/quit", "/exit"):
            return 0
        if query == "/help":
            _print_help()
            continue
        if query == "/examples":
            _print_examples()
            continue
        _handle_query(workflow, query)


if __name__ == "__main__":
    sys.exit(main())
