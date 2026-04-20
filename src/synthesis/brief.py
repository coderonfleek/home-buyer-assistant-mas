"""Synthesis layer: builds the final buyer's brief.

The output has two parts:
  A. A 2-3 paragraph prose brief written by the LLM.
  B. A ranked shortlist table (scored in Python, rationale-written by LLM).

We re-query the listings DB directly for structured scoring data rather
than trying to parse the listings agent's prose. This is deliberate: it
keeps the scoring deterministic and the agent's freedom of expression intact.
"""
from __future__ import annotations

import json
import re

from langchain.chat_models import init_chat_model

from src.state import AgentOutput, Classification
from src.synthesis.scoring import rank_listings
from src.tools.listings import fetch_listings_raw

SYNTH_MODEL = "openai:gpt-4o-mini"

SYNTH_SYSTEM_PROMPT = """You are a home-buying research synthesizer. You \
produce a buyer's brief from data gathered by three specialist agents.

Output format — follow exactly:

## Market snapshot
<1-2 short paragraphs about what the data shows for the area(s) in the query. \
Reference specific numbers from the specialists when relevant.>

## How your criteria fit
<1 short paragraph: how well the buyer's criteria hold up against the \
available data. Surface trade-offs (e.g., "the cheapest listings are in \
ZIPs where school ratings are weaker").>

## Top picks
<For each listing in the shortlist provided to you, write ONE line in this \
exact format:
- **#<listing_id>** — <one-sentence rationale drawing on at least two of: \
price, schools, neighborhood stats>.
Do NOT invent listings. Do NOT include listings that aren't in the shortlist.>

Rules:
- Keep it tight. Total response should be under 350 words.
- Only use facts present in the input. Do not invent numbers or school names.
- If the buyer's query didn't trigger certain agents, just omit those \
aspects — don't speculate.
"""


# -----------------------------------------------------------------------------
# Budget / criteria extraction from the original query (lightweight).
# -----------------------------------------------------------------------------


def _extract_criteria(query: str) -> dict:
    """Pull structured-ish search criteria out of the original query.

    Deliberately simple regex-based parsing — the router has already done
    the hard work. We only need enough to re-query the DB for scoring.
    """
    q = query.lower()
    criteria: dict = {}

    # Max price like "$500k", "500k", "$500,000", "under 500000"
    m = re.search(r"\$?\s*(\d{2,4})\s*k\b", q)
    if m:
        criteria["max_price"] = int(m.group(1)) * 1000
    else:
        m = re.search(r"\$\s*([\d,]+)", q)
        if m:
            try:
                criteria["max_price"] = int(m.group(1).replace(",", ""))
            except ValueError:
                pass

    # Min bedrooms: "3BR", "3 bedroom", "3-bedroom", "four bedroom"
    m = re.search(r"(\d+)\s*(?:br|bed|bedroom)", q)
    if m:
        criteria["min_bedrooms"] = int(m.group(1))

    # ZIP code
    m = re.search(r"\b(\d{5})\b", query)
    if m:
        criteria["zip_code"] = m.group(1)

    # City (very rough: look for known cities)
    known_cities = {
        "austin": ("Austin", "TX"),
        "denver": ("Denver", "CO"),
        "nashville": ("Nashville", "TN"),
        "raleigh": ("Raleigh", "NC"),
        "phoenix": ("Phoenix", "AZ"),
    }
    for key, (city, state) in known_cities.items():
        if key in q:
            criteria["city"] = city
            criteria["state"] = state
            break

    return criteria


# -----------------------------------------------------------------------------
# Main entry point.
# -----------------------------------------------------------------------------


def build_final_answer(
    original_query: str,
    classifications: list[Classification],
    results: list[AgentOutput],
) -> str:
    """Build the full buyer's brief.

    Returns a markdown-formatted string. The CLI renders it with ``rich``.
    """
    sources_called = {r["source"] for r in results}

    # Only build a shortlist if the listings agent was actually invoked.
    shortlist: list[dict] = []
    if "listings" in sources_called:
        criteria = _extract_criteria(original_query)
        raw = fetch_listings_raw(
            city=criteria.get("city"),
            state=criteria.get("state"),
            zip_code=criteria.get("zip_code"),
            min_bedrooms=criteria.get("min_bedrooms"),
            max_price=criteria.get("max_price"),
            limit=25,
        )
        ranked = rank_listings(raw, max_price=criteria.get("max_price"))
        shortlist = ranked[:5]

    # Assemble the input for the LLM.
    agent_reports = "\n\n".join(
        f"### From {r['source']}:\n{r['result']}" for r in results
    )

    shortlist_payload = json.dumps(
        [
            {
                "listing_id": s["listing_id"],
                "address": s["address"],
                "city": s["city"],
                "state": s["state"],
                "zip_code": s["zip_code"],
                "price": s["price"],
                "bedrooms": s["bedrooms"],
                "bathrooms": s["bathrooms"],
                "sqft": s["sqft"],
                "score": s["_score"],
                "score_components": s["_components"],
            }
            for s in shortlist
        ],
        indent=2,
    ) if shortlist else "(no listings shortlist — the listings agent was not invoked)"

    user_prompt = (
        f"Original buyer question: {original_query}\n\n"
        f"Agent reports:\n{agent_reports}\n\n"
        f"Shortlist (already ranked by a deterministic scorer):\n{shortlist_payload}\n\n"
        f"Write the buyer's brief now."
    )

    llm = init_chat_model(SYNTH_MODEL)
    response = llm.invoke([
        {"role": "system", "content": SYNTH_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])
    brief = response.content.strip()

    # Append a machine-readable shortlist table for the CLI to render.
    if shortlist:
        brief += "\n\n<!--SHORTLIST_JSON:" + json.dumps(shortlist, default=str) + "-->"

    return brief
