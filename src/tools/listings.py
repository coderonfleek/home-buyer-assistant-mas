"""Tools for the Listings agent.

These run SQL queries against the local SQLite database built by
``scripts/build_database.py``. Tool outputs are strings — the LLM reads
them directly — but we also expose ``fetch_listings_raw`` for the
synthesis layer, which needs structured records to do its scoring math.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from langchain.tools import tool

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "housing.db"


def _connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. "
            "Run `python scripts/build_database.py` first."
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _format_listing_row(row: sqlite3.Row) -> str:
    """Format a single listing as a compact line for the LLM."""
    return (
        f"#{row['listing_id']}: {row['address']}, {row['city']}, {row['state']} "
        f"{row['zip_code']} — ${row['price']:,}, {row['bedrooms']}bd/"
        f"{row['bathrooms']}ba, {row['sqft']:,} sqft, {row['property_type']}, "
        f"built {row['year_built']}"
    )


# -----------------------------------------------------------------------------
# Tools exposed to the LLM.
# -----------------------------------------------------------------------------


@tool
def search_listings(
    city: str | None = None,
    state: str | None = None,
    zip_code: str | None = None,
    min_bedrooms: int | None = None,
    max_price: int | None = None,
    min_sqft: int | None = None,
    property_type: str | None = None,
    limit: int = 15,
) -> str:
    """Search for home listings matching the given criteria.

    At least one filter should be provided. Returns up to ``limit`` matching
    listings (default 15), sorted by price ascending. Each listing includes
    a ``#listing_id`` you can pass to ``get_listing_details`` for more info.

    Args:
        city: City name, e.g. "Austin".
        state: Two-letter state code, e.g. "TX".
        zip_code: 5-digit ZIP code.
        min_bedrooms: Minimum number of bedrooms.
        max_price: Maximum price in dollars.
        min_sqft: Minimum square footage.
        property_type: One of "Single Family", "Townhouse", "Condo".
        limit: Maximum number of results to return (default 15, max 50).
    """
    limit = max(1, min(50, limit))
    clauses: list[str] = []
    params: list = []

    if city:
        clauses.append("LOWER(city) = LOWER(?)")
        params.append(city)
    if state:
        clauses.append("UPPER(state) = UPPER(?)")
        params.append(state)
    if zip_code:
        clauses.append("zip_code = ?")
        params.append(str(zip_code))
    if min_bedrooms is not None:
        clauses.append("bedrooms >= ?")
        params.append(min_bedrooms)
    if max_price is not None:
        clauses.append("price <= ?")
        params.append(max_price)
    if min_sqft is not None:
        clauses.append("sqft >= ?")
        params.append(min_sqft)
    if property_type:
        clauses.append("LOWER(property_type) = LOWER(?)")
        params.append(property_type)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT listing_id, address, city, state, zip_code, price,
               bedrooms, bathrooms, sqft, year_built, property_type
        FROM listings
        {where}
        ORDER BY price ASC
        LIMIT ?
    """
    params.append(limit)

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        return "No listings matched those criteria."

    header = f"Found {len(rows)} listing(s):"
    lines = [_format_listing_row(r) for r in rows]
    return header + "\n" + "\n".join(lines)


@tool
def get_listing_details(listing_id: int) -> str:
    """Get the full details of a specific listing by its ID.

    Use this after ``search_listings`` when you need more information about
    a particular property.
    """
    with _connect() as conn:
        row = conn.execute(
            """SELECT listing_id, address, city, state, zip_code, price,
                      bedrooms, bathrooms, sqft, year_built, property_type
               FROM listings WHERE listing_id = ?""",
            (listing_id,),
        ).fetchone()

    if not row:
        return f"No listing found with ID {listing_id}."

    return (
        f"Listing #{row['listing_id']}:\n"
        f"  Address: {row['address']}, {row['city']}, {row['state']} {row['zip_code']}\n"
        f"  Price: ${row['price']:,}\n"
        f"  Beds / Baths: {row['bedrooms']} / {row['bathrooms']}\n"
        f"  Square feet: {row['sqft']:,}\n"
        f"  Year built: {row['year_built']}\n"
        f"  Type: {row['property_type']}"
    )


# -----------------------------------------------------------------------------
# Non-tool helpers used by the synthesis layer (not exposed to the LLM).
# -----------------------------------------------------------------------------


def fetch_listings_raw(
    city: str | None = None,
    state: str | None = None,
    zip_code: str | None = None,
    min_bedrooms: int | None = None,
    max_price: int | None = None,
    min_sqft: int | None = None,
    limit: int = 30,
) -> list[dict]:
    """Return matching listings as plain dicts (for scoring).

    This is a normal Python function — not a tool. The synthesis node uses
    it to get structured data it can compute with.
    """
    clauses: list[str] = []
    params: list = []

    if city:
        clauses.append("LOWER(city) = LOWER(?)")
        params.append(city)
    if state:
        clauses.append("UPPER(state) = UPPER(?)")
        params.append(state)
    if zip_code:
        clauses.append("zip_code = ?")
        params.append(str(zip_code))
    if min_bedrooms is not None:
        clauses.append("bedrooms >= ?")
        params.append(min_bedrooms)
    if max_price is not None:
        clauses.append("price <= ?")
        params.append(max_price)
    if min_sqft is not None:
        clauses.append("sqft >= ?")
        params.append(min_sqft)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT listing_id, address, city, state, zip_code, price,
               bedrooms, bathrooms, sqft, year_built, property_type
        FROM listings {where}
        ORDER BY price ASC
        LIMIT ?
    """
    params.append(limit)

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def fetch_listing_by_id(listing_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM listings WHERE listing_id = ?", (listing_id,)
        ).fetchone()
    return dict(row) if row else None
