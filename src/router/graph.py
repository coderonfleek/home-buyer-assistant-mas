"""StateGraph assembly for the router workflow.

The graph has one entry (``classify``), three parallel specialist nodes
(``listings``, ``neighborhood``, ``schools``), and one exit node
(``synthesize``). Parallel fan-out is wired via ``add_conditional_edges``
pointing at the ``route_to_agents`` function from ``nodes.py``.
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from src.router.classifier import classify_query
from src.router.nodes import (
    query_listings,
    query_neighborhood,
    query_schools,
    route_to_agents,
    synthesize_results,
)
from src.state import RouterState


@lru_cache(maxsize=1)
def build_workflow():
    """Compile the router graph. Cached so repeated calls in the CLI are free."""
    return (
        StateGraph(RouterState)
        .add_node("classify", classify_query)
        .add_node("listings", query_listings)
        .add_node("neighborhood", query_neighborhood)
        .add_node("schools", query_schools)
        .add_node("synthesize", synthesize_results)
        .add_edge(START, "classify")
        .add_conditional_edges(
            "classify",
            route_to_agents,
            ["listings", "neighborhood", "schools"],
        )
        .add_edge("listings", "synthesize")
        .add_edge("neighborhood", "synthesize")
        .add_edge("schools", "synthesize")
        .add_edge("synthesize", END)
        .compile()
    )
