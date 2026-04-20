"""Build the SQLite database that backs the Listings and Schools agents.

Behavior:
  1. If data/raw/listings.csv is present, load it. Otherwise, generate a
     realistic synthetic dataset across several US metros so the project is
     runnable out-of-the-box.
  2. Same pattern for data/raw/schools.csv.

The goal is that a student can clone the repo, pip install, add their API
keys, and run the CLI within 5 minutes — without needing to hunt down a
Kaggle dataset first. When they're ready, they can drop a real CSV into
data/raw/ with the expected column names and re-run this script.

Expected columns (both CSVs):

  listings.csv:  listing_id, address, city, state, zip_code, price,
                 bedrooms, bathrooms, sqft, year_built, property_type

  schools.csv:   school_id, name, zip_code, city, state, level,
                 enrollment, student_teacher_ratio, rating
"""
from __future__ import annotations

import csv
import random
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "housing.db"

# Metros we seed with synthetic data. ZIPs are real (representative of the
# metro) so the Census API returns real demographics for them.
METROS = [
    # (city, state, [(zip, median_price_center, desirability_0_to_1)])
    ("Austin", "TX", [
        ("78704", 820_000, 0.85),
        ("78745", 520_000, 0.60),
        ("78741", 410_000, 0.45),
        ("78702", 720_000, 0.75),
        ("78758", 460_000, 0.55),
    ]),
    ("Denver", "CO", [
        ("80203", 680_000, 0.80),
        ("80205", 620_000, 0.70),
        ("80210", 780_000, 0.82),
        ("80219", 420_000, 0.45),
        ("80247", 480_000, 0.55),
    ]),
    ("Nashville", "TN", [
        ("37206", 620_000, 0.75),
        ("37208", 540_000, 0.65),
        ("37211", 380_000, 0.50),
        ("37216", 510_000, 0.60),
        ("37013", 330_000, 0.40),
    ]),
    ("Raleigh", "NC", [
        ("27601", 520_000, 0.70),
        ("27607", 610_000, 0.78),
        ("27610", 310_000, 0.40),
        ("27612", 560_000, 0.68),
        ("27603", 380_000, 0.50),
    ]),
    ("Phoenix", "AZ", [
        ("85004", 540_000, 0.70),
        ("85016", 620_000, 0.75),
        ("85033", 310_000, 0.40),
        ("85044", 450_000, 0.58),
        ("85053", 380_000, 0.48),
    ]),
]

STREET_NAMES = [
    "Oak", "Maple", "Cedar", "Elm", "Pine", "Birch", "Walnut", "Willow",
    "Chestnut", "Aspen", "Magnolia", "Cypress", "Juniper", "Dogwood",
    "Hickory", "Sycamore", "Linden",
]
STREET_TYPES = ["St", "Ave", "Blvd", "Ln", "Dr", "Ct", "Way", "Pl"]
PROPERTY_TYPES = ["Single Family", "Townhouse", "Condo"]


def _generate_listings() -> list[dict]:
    """Generate ~400 synthetic listings across the metros."""
    random.seed(42)  # Reproducibility — students see the same data we do.
    listings: list[dict] = []
    listing_id = 1000

    for city, state, zips in METROS:
        for zip_code, price_center, _desirability in zips:
            # 12-20 listings per ZIP
            n = random.randint(12, 20)
            for _ in range(n):
                beds = random.choices([2, 3, 4, 5], weights=[15, 45, 30, 10])[0]
                baths = max(1.0, beds - random.choice([0, 0, 1, 1]))
                # Price roughly tracks bed count around the ZIP's center.
                price_multiplier = {2: 0.75, 3: 1.0, 4: 1.35, 5: 1.8}[beds]
                noise = random.uniform(0.80, 1.20)
                price = int(price_center * price_multiplier * noise / 1000) * 1000
                sqft = int((800 + beds * 450) * random.uniform(0.85, 1.15))
                year_built = random.randint(1950, 2023)

                street_num = random.randint(100, 9999)
                street = f"{random.choice(STREET_NAMES)} {random.choice(STREET_TYPES)}"
                address = f"{street_num} {street}"

                listings.append({
                    "listing_id": listing_id,
                    "address": address,
                    "city": city,
                    "state": state,
                    "zip_code": zip_code,
                    "price": price,
                    "bedrooms": beds,
                    "bathrooms": baths,
                    "sqft": sqft,
                    "year_built": year_built,
                    "property_type": random.choice(PROPERTY_TYPES),
                })
                listing_id += 1

    return listings


def _generate_schools() -> list[dict]:
    """Generate 4-6 schools per ZIP across the metros."""
    random.seed(43)
    schools: list[dict] = []
    school_id = 5000

    school_name_prefixes = [
        "Lincoln", "Jefferson", "Washington", "Roosevelt", "Kennedy",
        "Madison", "Monroe", "Hamilton", "Franklin", "Adams", "Jackson",
        "Riverside", "Oakwood", "Hillcrest", "Parkview", "Summit",
        "Meadowbrook", "Fairview",
    ]

    for city, state, zips in METROS:
        for zip_code, _price_center, desirability in zips:
            # 4-6 schools per ZIP, mix of levels
            n_schools = random.randint(4, 6)
            levels = random.choices(
                ["elementary", "middle", "high"],
                weights=[3, 2, 2],
                k=n_schools,
            )
            for level in levels:
                name = f"{random.choice(school_name_prefixes)} {level.title()} School"
                enrollment = {
                    "elementary": random.randint(300, 700),
                    "middle": random.randint(500, 1100),
                    "high": random.randint(900, 2200),
                }[level]
                # Better ZIPs tend to have lower student-teacher ratios and
                # higher ratings. Add noise so it's not perfectly correlated.
                base_ratio = 22 - (desirability * 8)
                ratio = round(base_ratio + random.uniform(-2, 2), 1)
                base_rating = 3 + (desirability * 6)
                rating = max(1, min(10, round(base_rating + random.uniform(-1.5, 1.5))))

                schools.append({
                    "school_id": school_id,
                    "name": name,
                    "zip_code": zip_code,
                    "city": city,
                    "state": state,
                    "level": level,
                    "enrollment": enrollment,
                    "student_teacher_ratio": ratio,
                    "rating": rating,
                })
                school_id += 1

    return schools


def _load_csv_if_present(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return list(csv.DictReader(f))


def _coerce_listing(row: dict) -> dict:
    """Coerce CSV string values to the right Python types."""
    return {
        "listing_id": int(row["listing_id"]),
        "address": row["address"],
        "city": row["city"],
        "state": row["state"],
        "zip_code": str(row["zip_code"]).zfill(5),
        "price": int(float(row["price"])),
        "bedrooms": int(row["bedrooms"]),
        "bathrooms": float(row["bathrooms"]),
        "sqft": int(row["sqft"]),
        "year_built": int(row.get("year_built") or 0),
        "property_type": row.get("property_type") or "Single Family",
    }


def _coerce_school(row: dict) -> dict:
    return {
        "school_id": int(row["school_id"]),
        "name": row["name"],
        "zip_code": str(row["zip_code"]).zfill(5),
        "city": row["city"],
        "state": row["state"],
        "level": row["level"],
        "enrollment": int(row["enrollment"]),
        "student_teacher_ratio": float(row["student_teacher_ratio"]),
        "rating": int(row["rating"]),
    }


def build_database() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Load listings.
    raw_listings = _load_csv_if_present(RAW_DIR / "listings.csv")
    if raw_listings is not None:
        print(f"→ Loading {len(raw_listings)} listings from data/raw/listings.csv")
        listings = [_coerce_listing(r) for r in raw_listings]
    else:
        print("→ No data/raw/listings.csv found. Generating synthetic listings.")
        listings = _generate_listings()

    # Load schools.
    raw_schools = _load_csv_if_present(RAW_DIR / "schools.csv")
    if raw_schools is not None:
        print(f"→ Loading {len(raw_schools)} schools from data/raw/schools.csv")
        schools = [_coerce_school(r) for r in raw_schools]
    else:
        print("→ No data/raw/schools.csv found. Generating synthetic schools.")
        schools = _generate_schools()

    # Build the database.
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE listings (
            listing_id     INTEGER PRIMARY KEY,
            address        TEXT NOT NULL,
            city           TEXT NOT NULL,
            state          TEXT NOT NULL,
            zip_code       TEXT NOT NULL,
            price          INTEGER NOT NULL,
            bedrooms       INTEGER NOT NULL,
            bathrooms      REAL NOT NULL,
            sqft           INTEGER NOT NULL,
            year_built     INTEGER,
            property_type  TEXT
        )
    """)
    cur.execute("CREATE INDEX idx_listings_city ON listings(city, state)")
    cur.execute("CREATE INDEX idx_listings_zip ON listings(zip_code)")
    cur.execute("CREATE INDEX idx_listings_price ON listings(price)")

    cur.execute("""
        CREATE TABLE schools (
            school_id              INTEGER PRIMARY KEY,
            name                   TEXT NOT NULL,
            zip_code               TEXT NOT NULL,
            city                   TEXT NOT NULL,
            state                  TEXT NOT NULL,
            level                  TEXT NOT NULL,
            enrollment             INTEGER NOT NULL,
            student_teacher_ratio  REAL NOT NULL,
            rating                 INTEGER NOT NULL
        )
    """)
    cur.execute("CREATE INDEX idx_schools_zip ON schools(zip_code)")
    cur.execute("CREATE INDEX idx_schools_level ON schools(level)")

    cur.executemany(
        """INSERT INTO listings VALUES (
            :listing_id, :address, :city, :state, :zip_code, :price,
            :bedrooms, :bathrooms, :sqft, :year_built, :property_type
        )""",
        listings,
    )
    cur.executemany(
        """INSERT INTO schools VALUES (
            :school_id, :name, :zip_code, :city, :state, :level,
            :enrollment, :student_teacher_ratio, :rating
        )""",
        schools,
    )

    conn.commit()
    conn.close()

    print(f"✓ Built {DB_PATH}")
    print(f"  {len(listings)} listings, {len(schools)} schools")


if __name__ == "__main__":
    build_database()
