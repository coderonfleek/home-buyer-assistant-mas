"""Specialist agents for each vertical.

We build them lazily (inside functions rather than at import time) so
``scripts/build_database.py`` and other scripts that don't need a live LLM
can import this package without requiring ``OPENAI_API_KEY``.
"""
from __future__ import annotations

from functools import lru_cache

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from src.tools.listings import get_listing_details, search_listings
from src.tools.neighborhood import get_demographics, get_housing_stats
from src.tools.schools import get_school_stats, search_schools_near


# The agent model can be overridden via env var if you want to swap providers.
# gpt-4o-mini is plenty for this project; upgrade to gpt-4o for better tool-use
# reasoning on ambiguous queries.
AGENT_MODEL = "openai:gpt-4o-mini"


def _model():
    return init_chat_model(AGENT_MODEL)


@lru_cache(maxsize=1)
def listings_agent():
    """Specialist for searching home listings in the SQLite database."""
    return create_agent(
        model=_model(),
        tools=[search_listings, get_listing_details],
        system_prompt=(
            "You are a real-estate listings specialist. Your job is to find "
            "home listings in a local database that match a buyer's criteria. "
            "\n\n"
            "Guidelines:\n"
            "- Translate loose buyer language into concrete filters. For "
            "example, 'family home' usually means 3+ bedrooms.\n"
            "- When the buyer mentions a city, always pass both the city "
            "and state to search_listings.\n"
            "- Always include the #listing_id in your response so the caller "
            "can reference specific properties.\n"
            "- Keep your response concise: a one-line summary followed by "
            "the listings you found.\n"
            "- If no listings match, say so plainly — don't make anything up."
        ),
    )


@lru_cache(maxsize=1)
def neighborhood_agent():
    """Specialist for Census-based neighborhood and housing stats."""
    return create_agent(
        model=_model(),
        tools=[get_demographics, get_housing_stats],
        system_prompt=(
            "You are a neighborhood research specialist. You answer questions "
            "about US ZIP codes using US Census Bureau data (demographics and "
            "housing statistics)."
            "\n\n"
            "Guidelines:\n"
            "- If the query mentions a ZIP code, use it directly.\n"
            "- If the query mentions a city but no ZIP, say you need a ZIP "
            "code to look up Census data — do not guess.\n"
            "- Interpret the numbers for the reader. Don't just return raw "
            "stats; say what they imply (e.g., 'high owner-occupancy at 78% "
            "suggests a settled neighborhood').\n"
            "- If Census data is unavailable for a ZIP, say so plainly."
        ),
    )


@lru_cache(maxsize=1)
def schools_agent():
    """Specialist for school quality signals."""
    return create_agent(
        model=_model(),
        tools=[search_schools_near, get_school_stats],
        system_prompt=(
            "You are a schools research specialist. You help a buyer "
            "understand school quality in a given ZIP code."
            "\n\n"
            "Guidelines:\n"
            "- Queries about 'good schools' should translate into searching "
            "the schools database by ZIP and summarizing ratings and "
            "student-teacher ratios.\n"
            "- Always include the ZIP code(s) you searched and the number of "
            "schools found.\n"
            "- If a level is implied (e.g. 'for a young family' = elementary), "
            "filter accordingly, but also mention schools at other levels "
            "when relevant.\n"
            "- Be explicit about data gaps. Ratings are rough proxies, not "
            "definitive quality measures."
        ),
    )
