# US Home Buyer's Research Assistant

### A multi-agent CLI system built with LangChain v1.0 and LangGraph using the Router pattern

---

## 1. Project overview

You're going to build a command-line research assistant that helps a prospective US home buyer evaluate properties and markets. The buyer types a natural-language question — for example, *"Show me 3BR homes under $500k in Austin with good schools"* — and the system produces a **buyer's brief**: a short prose analysis followed by a ranked shortlist of properties.

Under the hood, a **router** classifies the query, decomposes it into targeted sub-questions, and dispatches them **in parallel** to three specialized agents:

- A **Listings agent** that searches a property database
- A **Neighborhood agent** that pulls demographic and housing statistics from the US Census
- A **Schools agent** that pulls school quality data from public education datasets

A final **synthesis node** merges the results into the buyer's brief.

### Why this project?

This is a near-ideal teaching vehicle for the Router pattern because:

- The three verticals are genuinely distinct — different data shapes, different tools, different prompts
- Most realistic buyer queries naturally hit 2–3 agents at once, which is exactly when parallel fan-out with `Send` is worth using
- The synthesis step does real work: it has to join listings to ZIP-level context and reason about trade-offs, not just concatenate
- The domain is tangible — you can evaluate whether the output is actually good by reading it

### What you'll learn

- Building specialized agents with `create_agent` and domain-specific tools
- Defining `TypedDict` state schemas and using **reducers** (`operator.add`) to collect parallel results
- Using **structured output** (Pydantic) to make routing decisions reliable
- Fanning out to multiple agents in parallel with `Send`
- Composing a `StateGraph` with conditional edges
- Wrapping a stateless router as a tool for a conversational agent (the stateful layer)
- Shipping a real CLI with nice formatted output

---

## 2. System architecture

### 2.1 High-level flow

```
┌─────────────────┐
│  User query     │  "3BR under $500k in Austin with good schools"
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Classifier    │  LLM with structured output → list[Classification]
└────────┬────────┘
         │ Send(...)
         ├──────────────┬──────────────┐
         ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Listings    │ │ Neighborhood │ │   Schools    │
│    agent     │ │    agent     │ │    agent     │
│ (SQLite)     │ │  (Census API)│ │ (Edu API)    │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        │ results (reducer: operator.add)
                        ▼
                ┌───────────────┐
                │  Synthesizer  │  Brief + ranked shortlist
                └───────┬───────┘
                        ▼
                    Final answer
```

### 2.2 Graph nodes

| Node | Role | Input | Output |
|---|---|---|---|
| `classify` | Decide which agents to invoke and with what sub-question | `RouterState` | `{"classifications": [...]}` |
| `listings` | Query the property database | `AgentInput` | `{"results": [{"source": "listings", ...}]}` |
| `neighborhood` | Query Census API | `AgentInput` | `{"results": [{"source": "neighborhood", ...}]}` |
| `schools` | Query education API | `AgentInput` | `{"results": [{"source": "schools", ...}]}` |
| `synthesize` | Build the buyer's brief | `RouterState` | `{"final_answer": "..."}` |

### 2.3 State schema

You'll define four types:

- **`AgentInput`** — Minimal input passed to each specialist (`{query: str}`)
- **`AgentOutput`** — What each specialist returns (`{source: str, result: str}`)
- **`Classification`** — A single routing decision (`{source, query}`)
- **`RouterState`** — The workflow state carried through the graph, with `results` using `Annotated[list, operator.add]` as the reducer

---

## 3. The three verticals

### 3.1 Listings agent

**Purpose:** Search and retrieve property listings.

**Data source:** A multi-metro US housing dataset loaded into a local SQLite database. Recommended candidates:

- Kaggle: *"US Real Estate Dataset"* (Zillow/Realtor snapshot, multiple metros, includes price, beds, baths, sqft, ZIP) — **first choice**
- Kaggle: *"USA Real Estate Dataset"* by Ahmed Shahriar Sakib
- Fallback: a cleaned synthetic CSV we generate in Episode 1 if no Kaggle dataset fits cleanly

The dataset needs at minimum: `city`, `state`, `zip_code`, `price`, `bedrooms`, `bathrooms`, `sqft`, and ideally `address` and `listing_id`.

**Why static is the right call for teaching:** Real MLS access is gated behind brokerage relationships. A SQLite file students can inspect makes debugging transparent, removes API flakiness from the classroom, and mirrors a legitimate real-world prototyping pattern (many proptech startups work this way).

**Tools:**

- `search_listings(city, state, min_beds, max_price, min_sqft, zip_code)` — filtered SQL query, returns top N results
- `get_listing_details(listing_id)` — full record for one property

**System prompt emphasis:** The agent should translate loose buyer criteria ("a family home") into concrete filter parameters, and always return the listing IDs so the synthesizer can reference them.

---

### 3.2 Neighborhood agent

**Purpose:** Provide demographic and housing context for a ZIP code or metro.

**Data source:** The **US Census Bureau API**. Free, the API key is instant, and the American Community Survey (ACS) 5-Year endpoint gives ZIP-level data.

- Base URL: `https://api.census.gov/data/`
- Dataset for this project: ACS 5-Year (`/2022/acs/acs5`)
- Get a free key: https://api.census.gov/data/key_signup.html

**Tools:**

- `get_demographics(zip_code)` — median household income, median age, population
- `get_housing_stats(zip_code)` — median home value, median rent, owner-occupancy rate
- `get_commute_stats(zip_code)` — mean travel time to work *(optional; nice for synthesis)*

**Implementation notes:**

- Census uses ZCTA (ZIP Code Tabulation Area) codes, which align with ZIP codes for our purposes
- Wrap all three tools in a tiny `httpx` client with a `CENSUS_API_KEY` env var
- Cache responses in-memory per session — the Census API is slow enough that this noticeably improves demo feel

**System prompt emphasis:** The agent should return *interpreted* stats (e.g., "median income in 78704 is $82k, above the Austin metro average") rather than raw numbers, so the synthesizer gets usable context.

---

### 3.3 Schools agent

**Purpose:** Provide school quality signals for a location.

**Data source (primary):** The **Urban Institute Education Data Portal API**. Free, no API key required, covers the NCES Common Core of Data (public schools) and private school universe.

- Docs: https://educationdata.urban.edu/documentation/
- Example endpoint: `https://educationdata.urban.edu/api/v1/schools/ccd/directory/{year}/`

**Data source (fallback):** If the Urban Institute API is down or rate-limited during a lesson, we preload an NCES Common Core of Data CSV extract into the same SQLite file as the listings. Episode 1 will set this up regardless.

**Tools:**

- `search_schools_near(zip_code, level)` — schools in or near a ZIP, filtered by level (`elementary`, `middle`, `high`)
- `get_school_stats(school_id)` — enrollment, student-teacher ratio, and any available performance indicators

**System prompt emphasis:** Schools data is notoriously messy — test scores are missing for many schools, ratings vary by state. The agent must be explicit about confidence and gaps, and prefer structural metrics (student-teacher ratio, enrollment stability) when performance data is unavailable.

---

## 4. The router

### 4.1 Classifier node

A single LLM call using **structured output** (Pydantic) to return a `ClassificationResult` containing a list of `Classification` objects. Each classification has:

- `source` — one of `"listings" | "neighborhood" | "schools"`
- `query` — a sub-question tailored to that source

The classifier prompt explicitly instructs the model to **only include relevant sources** and to rewrite the buyer's query for each source (e.g., the listings agent gets concrete filter criteria, the neighborhood agent gets a ZIP-level question).

### 4.2 Routing function

A `route_to_agents` function that maps each `Classification` to a `Send` object:

```python
return [Send(c["source"], {"query": c["query"]}) for c in classifications]
```

Wired into the graph via `add_conditional_edges`, this gives LangGraph permission to execute the selected agents **in parallel**.

### 4.3 Reducer

`RouterState.results` is `Annotated[list[AgentOutput], operator.add]`, so when the three parallel nodes each return `{"results": [one_output]}`, LangGraph concatenates them into a single list automatically.

---

## 5. The synthesis node — the "buyer's brief"

This is where the project earns its keep. The synthesizer produces a two-part output:

### Part A — Prose brief (2–4 short paragraphs)

- **Market snapshot.** What the dataset and Census show about the area(s) the buyer asked about.
- **Criteria fit.** How the buyer's criteria hold up against the data. Are they realistic? Are there trade-offs?
- **Key trade-offs.** The synthesizer explicitly surfaces tensions — for example, "the sub-$500k listings cluster in ZIPs where the schools have weaker student-teacher ratios than the metro average."

### Part B — Ranked shortlist (top 3–5 properties)

For each property:

- Address, price, beds / baths / sqft
- A one-line **"Why this one"** rationale that references data from at least two of the three agents
- A composite **match score** (0–100) that the synthesizer computes as a weighted blend of:
  - Price fit vs. the buyer's budget
  - School quality signal for the property's ZIP
  - Neighborhood fit (e.g., owner-occupancy rate as a proxy for settled neighborhoods)

The LLM handles the prose. The ranking logic can be either LLM-driven (simpler, less deterministic) or Python-driven (more complex, reproducible). **We'll implement a hybrid:** Python does the scoring math on the structured results, the LLM writes the rationales.

---

## 6. Representative queries and expected routing

| Query | Listings | Neighborhood | Schools |
|---|:-:|:-:|:-:|
| "3BR under $500k in Austin with good schools" | ✅ | ✅ | ✅ |
| "What's the median home price in 78704?" | ❌ | ✅ | ❌ |
| "Best school districts in the Austin metro under $600k" | ✅ | ❌ | ✅ |
| "Show me 4BR homes in 94110" | ✅ | ❌ | ❌ |
| "Is 10001 a family-friendly ZIP?" | ❌ | ✅ | ✅ |

This mix is important — you want students to see the router **omit** irrelevant verticals as well as fan out to all three.

---

## 7. Tech stack

| Layer | Choice | Reason |
|---|---|---|
| Language | Python 3.11+ | `TypedDict` features, modern type hints |
| Agent framework | `langchain` + `langgraph` (v1.0) | Course focus |
| LLM provider | OpenAI via `init_chat_model` | Default; easy to swap per docs |
| Router model | `gpt-4o-mini` or `gpt-4.1-mini` | Fast, cheap, structured output is reliable |
| Agent model | `gpt-4o` or `gpt-4.1` | Better tool-use reasoning |
| Listings DB | `sqlite3` (stdlib) | Zero setup, inspectable |
| HTTP client | `httpx` | Modern, async-ready if needed later |
| Validation | `pydantic` v2 | For the classifier's structured output |
| CLI rendering | `rich` | Tables, panels, and colored output for the brief |
| Env management | `python-dotenv` | Load API keys |
| Observability (optional) | LangSmith | Inspect traces when routing goes wrong |

---

## 8. Project structure

```
home-buyer-assistant/
├── README.md
├── requirements.txt
├── .env.example
├── .env                       # (gitignored) OpenAI + Census keys
├── data/
│   ├── listings.db            # Built in Episode 1
│   ├── schools.db             # Built in Episode 1 (or a single combined DB)
│   └── raw/                   # Source CSVs you download
│       └── .gitkeep
├── scripts/
│   └── build_database.py      # Loads CSVs into SQLite
└── src/
    ├── __init__.py
    ├── state.py               # TypedDicts + Pydantic schemas
    ├── tools/
    │   ├── __init__.py
    │   ├── listings.py        # SQLite-backed tools
    │   ├── neighborhood.py    # Census API tools
    │   └── schools.py         # Education API tools
    ├── agents/
    │   ├── __init__.py
    │   ├── listings.py        # create_agent(...) for listings
    │   ├── neighborhood.py
    │   └── schools.py
    ├── router/
    │   ├── __init__.py
    │   ├── classifier.py      # classify_query node
    │   ├── nodes.py           # query_* and synthesize nodes
    │   └── graph.py           # StateGraph assembly
    ├── synthesis/
    │   ├── __init__.py
    │   ├── scoring.py         # Python-side match scoring
    │   └── brief.py           # Prose brief rendering
    ├── stateful.py            # Optional: conversational wrapper
    └── cli.py                 # Entry point
```

Run with `python -m src.cli`.

---

## 9. Build plan (episode breakdown)

The project is split into eight episodes. Episodes 1–7 deliver a working stateless router. Episode 8 is an optional stretch that adds multi-turn memory.

### Episode 1 — Setup, data, and database

**Goal:** Working Python project with a populated SQLite database.

- Set up the repo, virtualenv, and `requirements.txt`
- Register for OpenAI and Census API keys, populate `.env`
- Download the listings Kaggle dataset and a schools CSV
- Write `scripts/build_database.py` to load both into SQLite
- Sanity-check with a few `sqlite3` CLI queries

**Deliverable:** `python scripts/build_database.py` produces a database with at least a few thousand listings across several metros.

---

### Episode 2 — Tools for each vertical

**Goal:** Each tool is independently callable and returns clean strings.

- Implement the two listings tools over SQLite
- Implement the three neighborhood tools over the Census API
- Implement the two schools tools (Urban Institute API primary, SQLite fallback)
- Unit-test each tool manually from a Python REPL

**Deliverable:** A set of `@tool`-decorated functions, each verified end-to-end against real data.

**Teaching moment:** Why tools return strings, not structured data, and how to keep tool outputs concise but LLM-parseable.

---

### Episode 3 — Specialist agents

**Goal:** Three working agents, each usable in isolation.

- Build `listings_agent`, `neighborhood_agent`, `schools_agent` with `create_agent`
- Write a focused system prompt for each (see §3.1–§3.3)
- Test each by invoking it directly with a query that only needs that vertical

**Deliverable:** Each agent answers a vertical-specific question correctly on its own.

**Teaching moment:** Why specialist system prompts beat one generic prompt — and how to keep them short.

---

### Episode 4 — State schemas and the classifier

**Goal:** A working classifier node.

- Define `AgentInput`, `AgentOutput`, `Classification`, `RouterState` in `state.py`
- Define the `ClassificationResult` Pydantic model
- Implement `classify_query` using `router_llm.with_structured_output(ClassificationResult)`
- Test the classifier standalone on all five representative queries from §6

**Deliverable:** Given any query, the classifier returns the expected list of `Classification` objects.

**Teaching moment:** The contract between the classifier and the downstream routing function — `source` strings must match node names exactly.

---

### Episode 5 — Routing with `Send`, and graph assembly

**Goal:** A compiled, end-to-end graph (minus synthesis).

- Implement the three agent-invoking nodes (`query_listings`, `query_neighborhood`, `query_schools`)
- Implement `route_to_agents` returning `list[Send]`
- Assemble the `StateGraph` with `add_conditional_edges` pointing from `classify` to the three agent nodes
- Add a **stub** synthesis node that just concatenates `state["results"]`
- Test with the three-vertical query and verify all three agents fire in parallel (check LangSmith or print timestamps)

**Deliverable:** `workflow.invoke({"query": "..."})` returns results from all relevant agents.

**Teaching moment:** How `Send` + `operator.add` work together, and why the agent nodes take `AgentInput`, not the full `RouterState`.

---

### Episode 6 — The synthesis node

**Goal:** Real buyer's briefs.

- Implement `scoring.py`: a Python function that computes match scores from the raw listings results
- Implement the `synthesize_results` node:
  - Parse the listings agent's result back into structured records
  - Compute scores in Python
  - Call the LLM to write the prose brief and the per-property rationales
  - Return a formatted `final_answer`
- Iterate on the synthesis prompt until the briefs feel useful

**Deliverable:** Running the full graph produces a readable, two-part buyer's brief.

**Teaching moment:** Why synthesis is a "context engineering" problem — what you pass to the synthesizer matters more than how you prompt it.

---

### Episode 7 — CLI

**Goal:** A polished interactive experience.

- Build `cli.py` with a `while True` input loop
- Render the brief with `rich.panel.Panel`, the shortlist with `rich.table.Table`
- Add commands: `/help`, `/examples`, `/quit`
- Print tracing info when `DEBUG=1` in the env

**Deliverable:** `python -m src.cli` opens a prompt, and a user can ask realistic questions and get formatted briefs back.

---

### Episode 8 (optional) — Stateful conversation

**Goal:** Multi-turn refinement.

- Wrap the compiled workflow as an `@tool` called `search_homes`
- Create a conversational agent with that one tool and an `InMemorySaver` checkpointer
- Update the CLI to route through the conversational agent when `--chat` is passed
- Demo a refinement flow: initial query → follow-up that narrows by school rating → follow-up that switches cities

**Deliverable:** `python -m src.cli --chat` gives a buyer who can refine their search across turns.

**Teaching moment:** The trade-off between the **tool wrapper approach** (clean separation, recommended) and **full router persistence** (more complex, rarely necessary).

---

## 10. Testing strategy

You don't need a full test suite for a teaching project, but do this at minimum:

- **Per-tool smoke tests** in a scratch notebook — each tool hits its real data source and returns something sensible
- **Classifier table test** — loop through the five queries in §6 and assert expected sources
- **Graph smoke test** — run one query of each type (single-vertical, two-vertical, three-vertical) and inspect the output
- **Synthesis eyeball test** — read five briefs and check that the prose references data from the agents that ran

LangSmith tracing is the best single debugging tool here. If an agent produces nonsense, the trace will show you whether it's a bad tool call, a bad prompt, or a bad classification.

---

## 11. Known risks and gotchas

| Risk | Mitigation |
|---|---|
| Census API returns no data for a ZIP (ZCTA mismatch) | Wrap all Census calls in try/except and have the tool return "data unavailable for this ZIP" — the agent can work with that |
| Schools API flakiness during a lesson | Episode 1 preloads an NCES CSV into SQLite as a fallback; have the schools tool try the API first, SQLite second |
| Classifier routes to the wrong agent | Add explicit examples to the classifier prompt; use structured output (Pydantic), not free-form JSON |
| LLM invents listing IDs in the synthesis step | Always pass the raw listings data to the synthesizer in the prompt, not just the agent's prose response |
| Costs creep up on repeated runs | Router on `gpt-4o-mini`, aggressive result caching at the tool level |
| Student hits OpenAI rate limits during recording | Have a local fallback model path (e.g., Ollama + Llama 3) documented in the README |

---

## 12. Success criteria

By the end of Episode 7, you should be able to:

1. Run `python -m src.cli`
2. Ask *"Show me 3BR homes under $500k in Austin with good schools"*
3. See the router classify the query and dispatch to all three agents
4. Watch the three agents run in parallel (visible in LangSmith)
5. Get back a two-paragraph market brief and a table of 3–5 ranked listings, each with a rationale that references data from at least two verticals

If you can do that, you've built a real Router-pattern multi-agent system — and you understand why the pattern exists.

---

## 13. Before you start

Make sure you have:

- [ ] Python 3.11+
- [ ] An OpenAI API key (or equivalent — the project is provider-agnostic)
- [ ] A free Census API key
- [ ] A Kaggle account (for the listings dataset download)
- [ ] Comfort with basic SQL and HTTP requests
- [ ] Read the LangChain Router docs: https://docs.langchain.com/oss/python/langchain/multi-agent/router
- [ ] Skimmed the reference tutorial: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

Ready? Move on to Episode 1.
