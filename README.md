# US Home Buyer's Research Assistant

A multi-agent CLI tool that helps a prospective US home buyer research properties and markets. Built with **LangChain v1.0** and **LangGraph** using the **Router pattern**.

Ask a natural-language question like *"Show me 3BR homes under $500k in Austin with good schools"* and the system:

1. **Classifies** the query and decomposes it into sub-questions
2. **Routes** them in parallel to three specialized agents (Listings · Neighborhood · Schools)
3. **Synthesizes** the results into a buyer's brief with a ranked shortlist

This is a teaching project accompanying the Udemy course on multi-agent patterns. See `project-plan.md` (if included) for the full blueprint.

---

## What's inside

```
home-buyer-assistant/
├── README.md                   ← you are here
├── requirements.txt
├── .env.example                ← copy to .env and fill in keys
├── data/
│   ├── raw/                    ← optional: drop your own CSVs here
│   └── housing.db              ← built by build_database.py
├── scripts/
│   └── build_database.py       ← seeds the SQLite DB
└── src/
    ├── state.py                ← TypedDicts + Pydantic schemas
    ├── cli.py                  ← entry point
    ├── streaming.py            ← live progress renderer
    ├── stateful.py             ← optional conversational wrapper
    ├── tools/                  ← @tool functions per vertical
    │   ├── listings.py
    │   ├── neighborhood.py     ← Census API
    │   └── schools.py
    ├── agents/
    │   └── specialists.py      ← 3 create_agent specialists
    ├── router/
    │   ├── classifier.py       ← classify node (structured output)
    │   ├── nodes.py            ← agent-invoker nodes + route_to_agents
    │   └── graph.py            ← StateGraph assembly
    └── synthesis/
        ├── scoring.py          ← Python match-score math
        └── brief.py            ← LLM writes the brief
```

---

## Prerequisites

- **Python 3.11 or newer**
- **An OpenAI API key** — https://platform.openai.com/api-keys
- **A free US Census API key** — https://api.census.gov/data/key_signup.html

You do *not* need a Kaggle account or any real estate dataset to run the project. The database builder seeds a realistic synthetic dataset across five US metros (Austin, Denver, Nashville, Raleigh, Phoenix) using real ZIP codes, so the Census API returns real demographics. You can swap in a real dataset later — see [Using your own data](#using-your-own-data).

---

## Setup (5 minutes)

**1. Create a virtual environment and install dependencies**

```bash
cd home-buyer-assistant
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Add your API keys**

```bash
cp .env.example .env
# Edit .env and paste in your OPENAI_API_KEY and CENSUS_API_KEY
```

**3. Build the database**

```bash
python scripts/build_database.py
```

You should see:

```
→ No data/raw/listings.csv found. Generating synthetic listings.
→ No data/raw/schools.csv found. Generating synthetic schools.
✓ Built data/housing.db
  399 listings, 126 schools
```

**4. Run it**

```bash
python -m src.cli
```

A prompt appears. Try:

```
> Show me 3BR homes under $500k in Austin with good schools
```

---

## Example session

```
> Show me 3BR homes under $500k in Austin with good schools

╭─ Working ────────────────────────────────────────────────────╮
│  🧭  Routed to: listings, schools                             │
│                                                               │
│  ⚡  Specialists                                              │
│  ✓   Listings    3+ bedroom homes under $500k in Austin  2.2s│
│  ⠋   Schools     Schools in Austin ZIP codes             1.5s│
│                                                               │
│  📝  Synthesizing buyer's brief...                            │
╰───────────────────────────────────────────────────────────────╯

 🧭 routed → listings, schools   ⚡ agents 2.2s parallel (sum 3.7s)   📝 synth 0.8s

╭─ Buyer's brief ──────────────────────────────────────────────╮
│ ## Market snapshot                                            │
│ Austin's sub-$500k segment clusters in 78741 and 78745, with  │
│ Census data showing median household incomes of ~$58k and     │
│ ~$75k respectively. Owner-occupancy sits around 40-55%.       │
│                                                               │
│ ## How your criteria fit                                      │
│ Your criteria are realistic. The main trade-off: 78741        │
│ listings are cheapest but its schools average 5.8/10, while   │
│ 78745's average 7.2/10 for a ~$60k premium.                   │
│                                                               │
│ ## Top picks                                                  │
│ - **#1042** — Strong on schools (8.1/10 in ZIP) and well       │
│   under budget at $385k.                                       │
│ - **#1029** — Best price-fit in Austin, newer build.          │
│ ...                                                           │
╰───────────────────────────────────────────────────────────────╯

                    Ranked shortlist
┏━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━┓
┃ Score ┃  ID  ┃ Address         ┃ Location         ┃    Price ┃ Beds/Baths┃ Sqft ┃
┡━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━┩
│    82 │ 1042 │ 2834 Oak St     │ Austin, TX 78745 │ $385,000 │   3 / 2   │ 2123 │
│    78 │ 1029 │ 1738 Cedar Ln   │ Austin, TX 78741 │ $329,000 │   3 / 3   │ 2228 │
│  ...  │ ...  │ ...             │ ...              │   ...    │    ...    │ ...  │
└───────┴──────┴─────────────────┴──────────────────┴──────────┴───────────┴──────┘
Total time: 4.2s
```

The `⚡ agents 2.2s parallel (sum 3.7s)` line is worth watching. It's proof that `Send` is actually running the specialists concurrently — the wall-clock time (2.2s) is less than the sum of per-agent times (3.7s). If the graph were sequential, those numbers would be equal.

---

## CLI commands

Inside the REPL:

| Command | Does |
|---|---|
| `/help` | Show commands |
| `/examples` | Show example queries |
| `/quit` | Exit (also Ctrl+D) |

Set `DEBUG=1` in your `.env` to see the raw output of each specialist agent beneath the buyer's brief:

```
DEBUG=1 python -m src.cli
```

---

## How the Router pattern works here

1. **Classify.** A cheap LLM (`gpt-4o-mini`) analyzes the query with **structured output** (Pydantic) and returns a list of `(source, sub_question)` pairs. Only relevant specialists are included.
2. **Fan out.** `route_to_agents` emits a `Send` per classification. LangGraph dispatches them in parallel.
3. **Specialists run concurrently.** Each has its own tools and system prompt:
   - **Listings** → SQL over SQLite (`data/housing.db`)
   - **Neighborhood** → US Census ACS 5-Year API
   - **Schools** → SQL over the same DB
4. **Reduce.** The `results` field in state has `Annotated[list, operator.add]`, so parallel outputs concatenate into one list.
5. **Synthesize.** A Python scoring function ranks listings deterministically; an LLM writes the prose brief and per-property rationales.

The CLI uses `workflow.stream(..., stream_mode="updates")` to show each of these steps live — you can see the classification land, watch the specialists tick through, and confirm they're actually running in parallel (wall-clock time vs. sum of per-agent times is surfaced in the summary line).

The shape of the graph:

```
        START
          │
       classify ──────────┐
          │               │ Send() fan-out
    ┌─────┼─────┐
    ▼     ▼     ▼
 listings nbhd schools     (run in parallel)
    │     │     │
    └─────┼─────┘
          │  (reducer merges results)
      synthesize
          │
         END
```

---

## Optional: multi-turn mode

The tutorial covers wrapping the stateless router as a tool so a conversational agent can refine across turns. See `src/stateful.py`:

```python
from src.stateful import build_chat_agent

agent = build_chat_agent()
config = {"configurable": {"thread_id": "user-1"}}

agent.invoke(
    {"messages": [{"role": "user", "content": "3BR in Austin under $500k"}]},
    config=config,
)
agent.invoke(
    {"messages": [{"role": "user", "content": "Now only ones with 8+ school ratings"}]},
    config=config,
)
```

---

## Using your own data

The SQLite database is seeded from CSVs in `data/raw/` if they exist. Drop in your own and re-run the builder:

**data/raw/listings.csv** columns:
```
listing_id, address, city, state, zip_code, price, bedrooms, bathrooms,
sqft, year_built, property_type
```

**data/raw/schools.csv** columns:
```
school_id, name, zip_code, city, state, level, enrollment,
student_teacher_ratio, rating
```

- `state` is a 2-letter code (`TX`, `CO`, ...)
- `zip_code` is a 5-digit string
- `level` is `elementary`, `middle`, or `high`
- `rating` is 1-10

Then:

```bash
python scripts/build_database.py
```

Real-data sources worth knowing about:

- **Listings**: Kaggle — "USA Real Estate Dataset" by Ahmed Shahriar Sakib, or Zillow/Redfin metro snapshots.
- **Schools**: the NCES Common Core of Data (free CSVs), or the Urban Institute Education Data Portal API (free, no key).

---

## Troubleshooting

**`CENSUS_API_KEY is not set`** — Add it to `.env`. The API key is free and arrives by email within minutes.

**`Database not found at .../housing.db`** — Run `python scripts/build_database.py`.

**Neighborhood agent says "Census data unavailable"** — The Census API is occasionally slow or returns no data for certain ZCTAs. The agent handles this gracefully; the other specialists will still run.

**Slow responses** — The first query in a session is slower because the graph is being compiled and the agents are being built. Subsequent queries are faster.

**Want to swap LLM providers?** — Change `AGENT_MODEL` in `src/agents/specialists.py`, `ROUTER_MODEL` in `src/router/classifier.py`, and `SYNTH_MODEL` in `src/synthesis/brief.py`. `init_chat_model` accepts any LangChain-supported provider string.

---

## Reference

- LangChain Router pattern docs: https://docs.langchain.com/oss/python/langchain/multi-agent/router
- LangChain Router tutorial (knowledge base): https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base
- US Census ACS API: https://www.census.gov/data/developers/data-sets/acs-5year.html