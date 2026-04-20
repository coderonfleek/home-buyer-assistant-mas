"""Classifier node: decides which specialist agents to invoke.

Uses structured output (Pydantic) so the routing decision is always a
valid list of (source, sub_query) pairs. This is the brain of the router.
"""
from __future__ import annotations

from langchain.chat_models import init_chat_model

from src.state import ClassificationResult, RouterState

# Cheaper/faster model is fine for classification.
ROUTER_MODEL = "openai:gpt-4o-mini"

CLASSIFIER_SYSTEM_PROMPT = """You are a routing classifier for a US home-buying \
research assistant. Analyze the user's query and decide which specialist \
agents to invoke.

Available specialists:
- listings: Searches a database of home listings. Use for anything about \
specific properties, price ranges, bedroom counts, or finding homes.
- neighborhood: Pulls US Census data for a ZIP code (demographics, median \
income, home values, owner-occupancy). Use ONLY when a ZIP code or \
neighborhood-level question is implied.
- schools: Searches a schools database by ZIP. Use for anything about \
school quality, ratings, or education.

Rules:
1. Only include specialists that are actually relevant. Omit the rest.
2. For each included specialist, rewrite the query as a concrete sub-question \
tailored to that specialist's tools.
3. The listings specialist doesn't know about schools or demographics — \
don't ask it about those.
4. The neighborhood specialist needs a ZIP code to work. If the user gives \
only a city, you can still route to listings/schools but skip neighborhood \
unless a ZIP is mentioned.

Examples:

Query: "3BR under $500k in Austin with good schools"
→ listings: "3+ bedroom homes under $500,000 in Austin, TX"
→ schools: "Schools in Austin, TX ZIP codes — we need good ratings"
(neighborhood omitted: no ZIP provided)

Query: "What's the median home price in 78704?"
→ neighborhood: "Housing stats for ZIP 78704"
(listings and schools omitted)

Query: "Show me 4BR homes in 94110"
→ listings: "4+ bedroom homes in ZIP 94110"
(neighborhood and schools not explicitly requested)

Query: "Is 10001 a family-friendly ZIP?"
→ neighborhood: "Demographics and housing stats for ZIP 10001"
→ schools: "Schools in ZIP 10001, especially elementary"
"""


def classify_query(state: RouterState) -> dict:
    """Classify the user query into a list of agent invocations."""
    llm = init_chat_model(ROUTER_MODEL).with_structured_output(ClassificationResult)
    result: ClassificationResult = llm.invoke([
        {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": state["query"]},
    ])
    # Convert Pydantic models to plain dicts for the TypedDict state.
    return {
        "classifications": [
            {"source": c.source, "query": c.query} for c in result.classifications
        ]
    }
