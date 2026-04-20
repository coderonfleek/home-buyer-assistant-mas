"""Graph nodes: agent invokers, routing function, and the synthesis node."""
from __future__ import annotations

from langgraph.types import Send

from src.agents.specialists import listings_agent, neighborhood_agent, schools_agent
from src.state import AgentInput, RouterState
from src.synthesis.brief import build_final_answer


# -----------------------------------------------------------------------------
# Routing function — fans out to specialists in parallel.
# -----------------------------------------------------------------------------


def route_to_agents(state: RouterState) -> list[Send]:
    """Map each classification to a Send; LangGraph runs them in parallel."""
    return [
        Send(c["source"], {"query": c["query"]})
        for c in state["classifications"]
    ]


# -----------------------------------------------------------------------------
# Specialist-invoking nodes. Each takes AgentInput, returns a results list.
# -----------------------------------------------------------------------------


def _invoke_agent(agent, query: str) -> str:
    """Invoke a create_agent-built agent and return its final message text."""
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


def query_listings(state: AgentInput) -> dict:
    text = _invoke_agent(listings_agent(), state["query"])
    return {"results": [{"source": "listings", "result": text}]}


def query_neighborhood(state: AgentInput) -> dict:
    text = _invoke_agent(neighborhood_agent(), state["query"])
    return {"results": [{"source": "neighborhood", "result": text}]}


def query_schools(state: AgentInput) -> dict:
    text = _invoke_agent(schools_agent(), state["query"])
    return {"results": [{"source": "schools", "result": text}]}


# -----------------------------------------------------------------------------
# Synthesis node.
# -----------------------------------------------------------------------------


def synthesize_results(state: RouterState) -> dict:
    """Build the buyer's brief from agent results + scored listings."""
    if not state["results"]:
        return {"final_answer": "No results found from any knowledge source."}

    answer = build_final_answer(
        original_query=state["query"],
        classifications=state["classifications"],
        results=state["results"],
    )
    return {"final_answer": answer}
