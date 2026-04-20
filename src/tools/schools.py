"""Tools for the Schools agent.

We query the local SQLite database. The real-world version of this
project would hit the Urban Institute Education Data Portal API; the
tutorial substitutes a preloaded dataset so lessons don't break when an
external API is flaky. The tool surface is identical either way.
"""
from __future__ import annotations

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


def _format_school_row(row: sqlite3.Row) -> str:
    return (
        f"#{row['school_id']}: {row['name']} ({row['level']}) — "
        f"ZIP {row['zip_code']}, rating {row['rating']}/10, "
        f"{row['enrollment']:,} students, {row['student_teacher_ratio']:.1f}:1 ratio"
    )


# -----------------------------------------------------------------------------
# Tools.
# -----------------------------------------------------------------------------


@tool
def search_schools_near(
    zip_code: str,
    level: str | None = None,
    limit: int = 10,
) -> str:
    """Find schools in a specific ZIP code, optionally filtered by level.

    Args:
        zip_code: 5-digit ZIP code.
        level: One of "elementary", "middle", "high". If omitted, returns all levels.
        limit: Maximum results (default 10, max 25).
    """
    limit = max(1, min(25, limit))
    zip_code = str(zip_code).zfill(5)

    clauses = ["zip_code = ?"]
    params: list = [zip_code]
    if level:
        clauses.append("LOWER(level) = LOWER(?)")
        params.append(level)

    sql = f"""
        SELECT school_id, name, zip_code, city, state, level,
               enrollment, student_teacher_ratio, rating
        FROM schools
        WHERE {' AND '.join(clauses)}
        ORDER BY rating DESC, student_teacher_ratio ASC
        LIMIT ?
    """
    params.append(limit)

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        return f"No schools found in ZIP {zip_code}" + (f" for level '{level}'." if level else ".")

    header = f"Found {len(rows)} school(s) in ZIP {zip_code}:"
    lines = [_format_school_row(r) for r in rows]
    return header + "\n" + "\n".join(lines)


@tool
def get_school_stats(school_id: int) -> str:
    """Get full details for a specific school by its ID."""
    with _connect() as conn:
        row = conn.execute(
            """SELECT school_id, name, zip_code, city, state, level,
                      enrollment, student_teacher_ratio, rating
               FROM schools WHERE school_id = ?""",
            (school_id,),
        ).fetchone()

    if not row:
        return f"No school found with ID {school_id}."

    return (
        f"School #{row['school_id']}:\n"
        f"  Name: {row['name']}\n"
        f"  Location: {row['city']}, {row['state']} {row['zip_code']}\n"
        f"  Level: {row['level']}\n"
        f"  Enrollment: {row['enrollment']:,}\n"
        f"  Student-teacher ratio: {row['student_teacher_ratio']:.1f}:1\n"
        f"  Rating: {row['rating']}/10"
    )


# -----------------------------------------------------------------------------
# Non-tool helpers used by synthesis.
# -----------------------------------------------------------------------------


def fetch_schools_summary(zip_code: str) -> dict:
    """Return aggregate school stats for a ZIP (for scoring).

    Returns a dict with ``school_count``, ``avg_rating``, and
    ``avg_student_teacher_ratio``. All values are ``None`` if there are
    no schools in that ZIP.
    """
    zip_code = str(zip_code).zfill(5)
    with _connect() as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS n, AVG(rating) AS avg_rating,
                      AVG(student_teacher_ratio) AS avg_ratio
               FROM schools WHERE zip_code = ?""",
            (zip_code,),
        ).fetchone()

    n = row["n"] or 0
    if n == 0:
        return {
            "zip_code": zip_code,
            "school_count": 0,
            "avg_rating": None,
            "avg_student_teacher_ratio": None,
        }
    return {
        "zip_code": zip_code,
        "school_count": n,
        "avg_rating": round(row["avg_rating"], 1),
        "avg_student_teacher_ratio": round(row["avg_ratio"], 1),
    }
