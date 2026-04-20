"""Python-side match scoring for the shortlist.

The LLM writes the prose brief and per-property rationales; Python
handles the scoring math so rankings are deterministic and testable.

Score is 0-100 and blends three components:
  - Price fit: does the listing fit the buyer's budget well (not too cheap,
    not too expensive relative to the budget)?
  - School quality: avg rating of schools in the listing's ZIP.
  - Neighborhood fit: owner-occupancy rate as a proxy for settled areas.

Components that are unavailable (e.g., no Census data for a ZIP, or no
schools in the DB for that ZIP) are simply dropped and the remaining
components are renormalized.
"""
from __future__ import annotations

from src.tools.neighborhood import fetch_neighborhood_raw
from src.tools.schools import fetch_schools_summary


def _price_fit_score(price: int, max_price: int | None) -> float:
    """1.0 at ~75% of budget, tapering off above and below. 0-1 range."""
    if max_price is None:
        return 0.7  # neutral
    if price > max_price:
        return 0.0
    ratio = price / max_price
    # Reward being comfortably under budget without being absurdly cheap.
    # Peak at ratio ~0.75.
    if ratio <= 0.75:
        return 0.7 + (ratio / 0.75) * 0.3  # 0.7 to 1.0
    else:
        return 1.0 - ((ratio - 0.75) / 0.25) * 0.4  # 1.0 down to 0.6 at budget


def _school_score(zip_code: str) -> float | None:
    summary = fetch_schools_summary(zip_code)
    rating = summary.get("avg_rating")
    if rating is None:
        return None
    return max(0.0, min(1.0, rating / 10.0))


def _neighborhood_score(zip_code: str) -> float | None:
    stats = fetch_neighborhood_raw(zip_code)
    owner_pct = stats.get("owner_occupancy_pct")
    if owner_pct is None:
        return None
    # Owner-occupancy 40% → 0.4, 80% → 1.0 (capped).
    return max(0.0, min(1.0, owner_pct / 80.0))


def score_listing(
    listing: dict,
    max_price: int | None = None,
) -> dict:
    """Compute a 0-100 match score for a single listing.

    Returns a dict with the score and its components (for display).
    """
    components: dict[str, float] = {}

    price_fit = _price_fit_score(listing["price"], max_price)
    components["price_fit"] = price_fit

    sch = _school_score(listing["zip_code"])
    if sch is not None:
        components["school"] = sch

    nbhd = _neighborhood_score(listing["zip_code"])
    if nbhd is not None:
        components["neighborhood"] = nbhd

    # Weighted average of available components.
    # Default weights when all present: price 0.4, school 0.35, neighborhood 0.25.
    weights = {"price_fit": 0.4, "school": 0.35, "neighborhood": 0.25}
    total_w = sum(weights[k] for k in components)
    if total_w == 0:
        score = 0.0
    else:
        score = sum(components[k] * weights[k] for k in components) / total_w

    return {
        "listing_id": listing["listing_id"],
        "score": round(score * 100, 1),
        "components": {k: round(v, 2) for k, v in components.items()},
    }


def rank_listings(listings: list[dict], max_price: int | None = None) -> list[dict]:
    """Score and sort a list of listings. Returns sorted listings with ``_score`` attached."""
    scored = []
    for listing in listings:
        s = score_listing(listing, max_price=max_price)
        scored.append({**listing, "_score": s["score"], "_components": s["components"]})
    scored.sort(key=lambda x: x["_score"], reverse=True)
    return scored
