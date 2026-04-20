"""Live streaming renderer for the router workflow.

Listens to ``workflow.stream(..., stream_mode="updates")`` and keeps a
Rich ``Live`` panel in sync with what's happening:

  - Classification (which agents were picked, with their sub-queries)
  - Specialist agents running in parallel, with per-agent durations
  - Synthesis step

The renderer is purely presentational — it yields the final state back
to the caller once the stream completes. No business logic lives here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text


AGENT_LABELS = {
    "listings": "Listings",
    "neighborhood": "Neighborhood",
    "schools": "Schools",
}


@dataclass
class _AgentState:
    name: str
    status: str = "pending"   # pending | running | done | error
    sub_query: str = ""
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def duration(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.perf_counter()
        return end - self.started_at


@dataclass
class _StreamState:
    phase: str = "classifying"   # classifying | running | synthesizing | done
    classifications: list[dict] = field(default_factory=list)
    agents: dict[str, _AgentState] = field(default_factory=dict)
    synth_started_at: float | None = None
    synth_finished_at: float | None = None


def _status_icon(status: str) -> str:
    return {
        "pending": "⋯",
        "running": "…",    # replaced with a Spinner in the renderable
        "done": "✓",
        "error": "✗",
    }.get(status, "?")


def _status_style(status: str) -> str:
    return {
        "pending": "dim",
        "running": "yellow",
        "done": "green",
        "error": "red",
    }.get(status, "white")


def _render(state: _StreamState) -> Panel:
    """Build the Rich panel from the current stream state."""
    rows: list[Any] = []

    # --- Classification row ---
    if state.phase == "classifying":
        rows.append(Group(
            Text.assemble(
                (" 🧭  ", "bold cyan"),
                ("Classifying query...", "bold"),
            ),
        ))
    else:
        picked = ", ".join(c["source"] for c in state.classifications) or "(none)"
        rows.append(Text.assemble(
            (" 🧭  ", "bold cyan"),
            ("Routed to: ", "bold"),
            (picked, "bold magenta"),
        ))

    # --- Agents table ---
    if state.agents:
        rows.append(Text(""))
        rows.append(Text.assemble((" ⚡  ", "bold cyan"), ("Specialists", "bold")))
        tbl = Table.grid(padding=(0, 2))
        tbl.add_column(width=4)
        tbl.add_column(min_width=14)
        tbl.add_column(ratio=1)
        tbl.add_column(justify="right", width=8)
        for name in ("listings", "neighborhood", "schools"):
            ag = state.agents.get(name)
            if ag is None:
                continue
            label = AGENT_LABELS[name]
            style = _status_style(ag.status)
            # Status glyph (use a real spinner while running)
            if ag.status == "running":
                glyph: Any = Spinner("dots", style=style)
            else:
                glyph = Text(_status_icon(ag.status), style=style)
            # Duration
            dur = ag.duration
            dur_text = f"{dur:.1f}s" if dur is not None else ""
            # Sub-query (trimmed)
            sub = ag.sub_query or ""
            if len(sub) > 60:
                sub = sub[:57] + "..."
            tbl.add_row(
                glyph,
                Text(label, style="bold" if ag.status != "pending" else "dim"),
                Text(sub, style="dim"),
                Text(dur_text, style="dim"),
            )
        rows.append(tbl)

    # --- Synthesis row ---
    if state.phase in ("synthesizing", "done"):
        rows.append(Text(""))
        if state.phase == "synthesizing":
            rows.append(Group(
                Text.assemble(
                    (" 📝  ", "bold cyan"),
                    ("Synthesizing buyer's brief...", "bold yellow"),
                ),
            ))
        else:
            dur = (
                (state.synth_finished_at or time.perf_counter())
                - (state.synth_started_at or time.perf_counter())
            )
            rows.append(Text.assemble(
                (" 📝  ", "bold cyan"),
                ("Brief ready ", "bold green"),
                (f"({dur:.1f}s)", "dim"),
            ))

    return Panel(
        Group(*rows),
        title="[cyan]Working[/cyan]",
        border_style="cyan",
        padding=(1, 2),
    )


def stream_and_render(
    workflow: Any,
    query: str,
    console: Console,
) -> dict:
    """Run the workflow with streaming and a live status panel.

    Returns the accumulated final state (matching what ``invoke`` would return).
    """
    state = _StreamState(phase="classifying")
    # We don't know yet which agents will be picked; they're added after classify.

    # We need to accumulate the final state ourselves, since stream() yields
    # per-node deltas. The reducer on `results` is operator.add, so we
    # concatenate. `final_answer`, `classifications`, `query` are overwrites.
    final_state: dict = {"query": query, "results": []}

    with Live(
        _render(state),
        console=console,
        refresh_per_second=12,
        transient=True,
    ) as live:
        for event in workflow.stream({"query": query}, stream_mode="updates"):
            # Each event is {node_name: state_delta}
            for node_name, delta in event.items():
                if node_name == "classify":
                    classifications = delta.get("classifications", [])
                    state.classifications = classifications
                    final_state["classifications"] = classifications
                    state.phase = "running"
                    # Seed agent states — all picked agents start as running.
                    now = time.perf_counter()
                    for c in classifications:
                        src = c["source"]
                        state.agents[src] = _AgentState(
                            name=src,
                            status="running",
                            sub_query=c["query"],
                            started_at=now,
                        )

                elif node_name in AGENT_LABELS:
                    # A specialist finished.
                    ag = state.agents.get(node_name)
                    if ag is not None:
                        ag.status = "done"
                        ag.finished_at = time.perf_counter()
                    # Merge results into final state.
                    final_state["results"].extend(delta.get("results", []))
                    # If every picked agent is now done, flip to synthesizing.
                    if all(
                        state.agents[c["source"]].status == "done"
                        for c in state.classifications
                    ):
                        state.phase = "synthesizing"
                        state.synth_started_at = time.perf_counter()

                elif node_name == "synthesize":
                    state.phase = "done"
                    state.synth_finished_at = time.perf_counter()
                    final_state["final_answer"] = delta.get("final_answer", "")

            live.update(_render(state))

    # When the Live exits (transient=True), the panel disappears. We print
    # a compact summary line so the user has a permanent trace of what ran.
    picked = ", ".join(c["source"] for c in state.classifications) or "none"
    total = sum(
        (a.duration or 0.0) for a in state.agents.values() if a.duration
    )
    parallel_elapsed = max(
        (a.duration or 0.0 for a in state.agents.values()),
        default=0.0,
    )
    synth_dur = (
        (state.synth_finished_at or 0) - (state.synth_started_at or 0)
        if state.synth_started_at else 0.0
    )
    console.print(Text.assemble(
        (" 🧭 ", "cyan"), ("routed → ", "dim"), (picked, "magenta"),
        ("   ⚡ ", "cyan"), ("agents ", "dim"),
        (f"{parallel_elapsed:.1f}s parallel ", "green"),
        (f"(sum {total:.1f}s)", "dim"),
        ("   📝 ", "cyan"), ("synth ", "dim"), (f"{synth_dur:.1f}s", "green"),
    ))
    return final_state