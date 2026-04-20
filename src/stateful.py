"""Optional stateful wrapper: multi-turn refinement.

The recommended approach from the docs: wrap the stateless router as a
tool and give it to a conversational agent that handles memory. The
router stays stateless; the agent manages context across turns.

Usage:
    from src.stateful import build_chat_agent
    agent = build_chat_agent()
    config = {"configurable": {"thread_id": "user-1"}}
    agent.invoke(
        {"messages": [{"role": "user", "content": "3BR in Austin under $500k"}]},
        config=config,
    )
"""
from __future__ import annotations

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from src.router.graph import build_workflow

CHAT_MODEL = "openai:gpt-4o-mini"


@tool
def search_homes(query: str) -> str:
    """Research homes for a buyer across listings, neighborhood stats, and schools.

    Use this for any question about finding homes, comparing neighborhoods,
    or evaluating school quality. Pass the user's question verbatim or
    slightly rewritten for clarity.
    """
    workflow = build_workflow()
    result = workflow.invoke({"query": query})
    return result["final_answer"]


def build_chat_agent():
    """Build a conversational agent that uses the router as one tool."""
    return create_agent(
        model=init_chat_model(CHAT_MODEL),
        tools=[search_homes],
        system_prompt=(
            "You are a helpful home-buying research assistant. "
            "Use the search_homes tool to answer any question about "
            "homes, neighborhoods, or schools. Remember what the user "
            "has asked about previously so they can refine their search "
            "across turns — e.g., 'narrow to under $500k' or 'switch to "
            "Denver'. Combine prior context with the new refinement before "
            "calling the tool."
        ),
        checkpointer=InMemorySaver(),
    )
