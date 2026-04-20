"""Tools for the Neighborhood agent.

These call the US Census Bureau API (ACS 5-Year) and return per-ZIP
demographic and housing statistics. The Census API is free but requires
a key — get one at https://api.census.gov/data/key_signup.html.

Each tool returns a human-readable string. We cache responses per-session
to avoid repeat round-trips — the Census API is slow enough that caching
noticeably improves the demo feel.
"""
from __future__ import annotations

import os
from functools import lru_cache

import httpx
from langchain.tools import tool

CENSUS_BASE = "https://api.census.gov/data/2022/acs/acs5"
CENSUS_TIMEOUT = 15.0

# ACS variable codes — see https://api.census.gov/data/2022/acs/acs5/variables.html
VARIABLES = {
    "total_population": "B01003_001E",
    "median_age": "B01002_001E",
    "median_household_income": "B19013_001E",
    "median_home_value": "B25077_001E",
    "median_gross_rent": "B25064_001E",
    "owner_occupied_units": "B25003_002E",
    "total_occupied_units": "B25003_001E",
    "mean_travel_time": "B08013_001E",  # Not ZCTA-available; handled gracefully.
}


def _api_key() -> str:
    key = os.environ.get("CENSUS_API_KEY")
    if not key:
        raise RuntimeError(
            "CENSUS_API_KEY is not set. "
            "Get a free key at https://api.census.gov/data/key_signup.html "
            "and add it to your .env file."
        )
    return key


@lru_cache(maxsize=256)
def _fetch_acs(zip_code: str, variables: tuple[str, ...]) -> dict | None:
    """Fetch ACS variables for a ZCTA. Returns None if the ZIP isn't found."""
    params = {
        "get": ",".join(variables),
        "for": f"zip code tabulation area:{zip_code}",
        "key": _api_key(),
    }
    try:
        resp = httpx.get(CENSUS_BASE, params=params, timeout=CENSUS_TIMEOUT)
        resp.raise_for_status()
    except httpx.HTTPError:
        return None

    data = resp.json()
    # Shape is [[header_row], [values_row]]
    if not isinstance(data, list) or len(data) < 2:
        return None
    headers, values = data[0], data[1]
    return dict(zip(headers, values))


def _safe_int(value: str | None) -> int | None:
    """Census returns -666666666 and similar sentinels for missing data."""
    if value is None:
        return None
    try:
        n = int(value)
    except (ValueError, TypeError):
        return None
    if n < 0:
        return None
    return n


# -----------------------------------------------------------------------------
# Tools.
# -----------------------------------------------------------------------------


@tool
def get_demographics(zip_code: str) -> str:
    """Get population, median age, and median household income for a ZIP.

    Uses the US Census Bureau's American Community Survey (ACS 5-Year).
    Returns a summary string or an "unavailable" message if the ZIP has
    no ACS data.
    """
    zip_code = str(zip_code).zfill(5)
    vars_ = (
        VARIABLES["total_population"],
        VARIABLES["median_age"],
        VARIABLES["median_household_income"],
    )
    data = _fetch_acs(zip_code, vars_)
    if data is None:
        return f"Census data unavailable for ZIP {zip_code}."

    population = _safe_int(data.get(VARIABLES["total_population"]))
    median_age = _safe_int(data.get(VARIABLES["median_age"]))
    income = _safe_int(data.get(VARIABLES["median_household_income"]))

    lines = [f"Demographics for ZIP {zip_code}:"]
    if population is not None:
        lines.append(f"  Population: {population:,}")
    if median_age is not None:
        lines.append(f"  Median age: {median_age}")
    if income is not None:
        lines.append(f"  Median household income: ${income:,}")
    if len(lines) == 1:
        return f"Census returned no usable demographic data for ZIP {zip_code}."
    return "\n".join(lines)


@tool
def get_housing_stats(zip_code: str) -> str:
    """Get housing stats for a ZIP: median home value, rent, ownership rate.

    Uses the ACS 5-Year dataset. Returns a summary string or an
    "unavailable" message if the ZIP has no ACS data.
    """
    zip_code = str(zip_code).zfill(5)
    vars_ = (
        VARIABLES["median_home_value"],
        VARIABLES["median_gross_rent"],
        VARIABLES["owner_occupied_units"],
        VARIABLES["total_occupied_units"],
    )
    data = _fetch_acs(zip_code, vars_)
    if data is None:
        return f"Census data unavailable for ZIP {zip_code}."

    home_value = _safe_int(data.get(VARIABLES["median_home_value"]))
    rent = _safe_int(data.get(VARIABLES["median_gross_rent"]))
    owner = _safe_int(data.get(VARIABLES["owner_occupied_units"]))
    total = _safe_int(data.get(VARIABLES["total_occupied_units"]))

    lines = [f"Housing stats for ZIP {zip_code}:"]
    if home_value is not None:
        lines.append(f"  Median home value: ${home_value:,}")
    if rent is not None:
        lines.append(f"  Median gross rent: ${rent:,}/month")
    if owner is not None and total is not None and total > 0:
        pct = 100 * owner / total
        lines.append(f"  Owner-occupancy: {pct:.0f}% ({owner:,} of {total:,} units)")
    if len(lines) == 1:
        return f"Census returned no usable housing data for ZIP {zip_code}."
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Non-tool helpers used by synthesis.
# -----------------------------------------------------------------------------


def fetch_neighborhood_raw(zip_code: str) -> dict:
    """Return a flat dict of stats for a ZIP. Missing values become None.

    Used by the synthesis scoring layer.
    """
    zip_code = str(zip_code).zfill(5)
    data = _fetch_acs(
        zip_code,
        (
            VARIABLES["total_population"],
            VARIABLES["median_household_income"],
            VARIABLES["median_home_value"],
            VARIABLES["owner_occupied_units"],
            VARIABLES["total_occupied_units"],
        ),
    )
    if data is None:
        return {
            "zip_code": zip_code,
            "population": None,
            "median_household_income": None,
            "median_home_value": None,
            "owner_occupancy_pct": None,
        }

    owner = _safe_int(data.get(VARIABLES["owner_occupied_units"]))
    total = _safe_int(data.get(VARIABLES["total_occupied_units"]))
    owner_pct = (100 * owner / total) if owner is not None and total else None

    return {
        "zip_code": zip_code,
        "population": _safe_int(data.get(VARIABLES["total_population"])),
        "median_household_income": _safe_int(
            data.get(VARIABLES["median_household_income"])
        ),
        "median_home_value": _safe_int(data.get(VARIABLES["median_home_value"])),
        "owner_occupancy_pct": owner_pct,
    }
