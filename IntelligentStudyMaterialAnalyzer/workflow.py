"""
workflow.py — LangGraph StateGraph definition for the Study Companion agent.

Graph topology
──────────────

    [START]
       │
       ▼
  ┌─────────┐
  │  router  │   ← classifies intent
  └────┬────┘
       │ edge: route_to_tool()
       ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  One of:                                                          │
  │  summarize_tool | mcq_tool | keyword_tool | analytics_tool |     │
  │  qa_tool | flashcard_tool | revision_notes_tool |                │
  │  full_analysis_tool | unknown_handler                            │
  └──────────────────────────────────────────────────────────────────┘
       │  (all edges lead here)
       ▼
  ┌───────────┐
  │ responder │   ← shapes final JSON response
  └─────┬─────┘
        │
       [END]

Usage
─────
    from modules.agent.workflow import build_graph

    graph = build_graph()          # call once at app startup
    result_state = graph.invoke(initial_state)
    response_dict = result_state["response"]
"""

from __future__ import annotations
import logging

# LangGraph imports — langgraph must be installed (see requirements_agent.txt)
from langgraph.graph import StateGraph, END

from .state import (
    AgentState,
    INTENT_SUMMARIZE, INTENT_MCQ, INTENT_KEYWORDS, INTENT_ANALYTICS,
    INTENT_QA, INTENT_FLASHCARDS, INTENT_REVISION_NOTES,
    INTENT_FULL_ANALYSIS, INTENT_UNKNOWN,
)
from .router import router_node
from .tools import (
    summarize_tool, mcq_tool, keyword_tool, analytics_tool,
    qa_tool, flashcard_tool, revision_notes_tool, full_analysis_tool,
)
from .responder import responder_node

logger = logging.getLogger(__name__)


# ── Unknown-intent handler ───────────────────────────────────────────────────

def unknown_handler(state: AgentState) -> AgentState:
    """Reached when the router cannot classify the request."""
    msg = (
        "I couldn't determine what you'd like to do. "
        "Try requests like: 'Summarize this document', 'Generate 5 MCQs', "
        "'Create flashcards', 'Make revision notes', or ask a question directly."
    )
    return {**state, "error": msg}


# ── Conditional edge: router → tool node ─────────────────────────────────────

def route_to_tool(state: AgentState) -> str:
    """
    Returns the name of the next node based on state["intent"].
    This is the conditional edge function LangGraph calls after the router.
    """
    intent_to_node = {
        INTENT_SUMMARIZE:      "summarize_tool",
        INTENT_MCQ:            "mcq_tool",
        INTENT_KEYWORDS:       "keyword_tool",
        INTENT_ANALYTICS:      "analytics_tool",
        INTENT_QA:             "qa_tool",
        INTENT_FLASHCARDS:     "flashcard_tool",
        INTENT_REVISION_NOTES: "revision_notes_tool",
        INTENT_FULL_ANALYSIS:  "full_analysis_tool",
    }
    intent = state.get("intent", INTENT_UNKNOWN)
    node = intent_to_node.get(intent, "unknown_handler")
    logger.info(f"[workflow] routing intent='{intent}' → node='{node}'")
    return node


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph StateGraph.
    Call once at Flask app startup and store the result.
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("router",               router_node)
    graph.add_node("summarize_tool",       summarize_tool)
    graph.add_node("mcq_tool",             mcq_tool)
    graph.add_node("keyword_tool",         keyword_tool)
    graph.add_node("analytics_tool",       analytics_tool)
    graph.add_node("qa_tool",              qa_tool)
    graph.add_node("flashcard_tool",       flashcard_tool)
    graph.add_node("revision_notes_tool",  revision_notes_tool)
    graph.add_node("full_analysis_tool",   full_analysis_tool)
    graph.add_node("unknown_handler",      unknown_handler)
    graph.add_node("responder",            responder_node)

    # Entry point
    graph.set_entry_point("router")

    # Router → tool (conditional edge)
    graph.add_conditional_edges(
        "router",
        route_to_tool,
        {
            "summarize_tool":      "summarize_tool",
            "mcq_tool":            "mcq_tool",
            "keyword_tool":        "keyword_tool",
            "analytics_tool":      "analytics_tool",
            "qa_tool":             "qa_tool",
            "flashcard_tool":      "flashcard_tool",
            "revision_notes_tool": "revision_notes_tool",
            "full_analysis_tool":  "full_analysis_tool",
            "unknown_handler":     "unknown_handler",
        },
    )

    # Every tool/handler → responder
    for node_name in [
        "summarize_tool", "mcq_tool", "keyword_tool", "analytics_tool",
        "qa_tool", "flashcard_tool", "revision_notes_tool",
        "full_analysis_tool", "unknown_handler",
    ]:
        graph.add_edge(node_name, "responder")

    # Responder → END
    graph.add_edge("responder", END)

    return graph.compile()
