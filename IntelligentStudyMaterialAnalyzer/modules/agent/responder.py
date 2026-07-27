"""
responder.py — Final node in the LangGraph graph.

Reads whatever tool outputs are present in state and assembles a
clean, JSON-serialisable `response` dict for the Flask route to return.

This node runs AFTER the tool node regardless of which tool was chosen,
so it is the single place that translates internal state → API response.
"""

from __future__ import annotations
import logging
from .state import AgentState, INTENT_UNKNOWN

logger = logging.getLogger(__name__)


def responder_node(state: AgentState) -> AgentState:
    """
    Reads:  any populated output fields in state
    Writes: state["response"]  — the dict Flask will jsonify()
    """
    if state.get("error"):
        response = {
            "success": False,
            "error": state["error"],
            "intent": state.get("intent", INTENT_UNKNOWN),
        }
        return {**state, "response": response}

    intent = state.get("intent", INTENT_UNKNOWN)
    payload: dict = {"success": True, "intent": intent}

    # Each key is only included when the corresponding tool actually ran.
    if state.get("summary") is not None:
        payload["summary"] = state["summary"]

    if state.get("mcqs") is not None:
        payload["mcqs"] = state["mcqs"]

    if state.get("keywords") is not None:
        payload["keywords"] = state["keywords"]

    if state.get("analytics") is not None:
        payload["analytics"] = state["analytics"]

    if state.get("qa_answer") is not None:
        payload["question"] = state.get("tool_kwargs", {}).get(
            "question", state.get("user_request", "")
        )
        payload["answer"] = state["qa_answer"]

    if state.get("flashcards") is not None:
        payload["flashcards"] = state["flashcards"]

    if state.get("revision_notes") is not None:
        payload["revision_notes"] = state["revision_notes"]

    if len(payload) == 2:          # only success + intent — nothing ran
        payload["message"] = (
            f"Request understood as intent '{intent}' but no tool produced output. "
            "Please try rephrasing your request."
        )

    logger.info(f"[responder] built response for intent={intent}")
    return {**state, "response": payload}
