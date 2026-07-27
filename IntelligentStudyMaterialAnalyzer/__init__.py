"""
modules/agent — LangGraph orchestration layer for the Study Companion.

Public surface (imported by Flask routes):
  build_graph   → compile the StateGraph once at startup
  run_agent     → invoke the graph for a single request
  AgentState    → TypedDict that flows through every node
"""

from .workflow import build_graph
from .state import AgentState

__all__ = ["build_graph", "AgentState"]


def run_agent(graph, upload_id: int, document_text: str, user_request: str) -> dict:
    """
    Convenience wrapper used by Flask routes.

    Parameters
    ----------
    graph          : compiled LangGraph StateGraph (created by build_graph())
    upload_id      : int FK into the uploads table
    document_text  : processed_text from the DB
    user_request   : raw string from the HTTP request body

    Returns
    -------
    dict — the JSON-serialisable response payload (state["response"])
    """
    initial_state: AgentState = {
        "upload_id":     upload_id,
        "user_request":  user_request,
        "document_text": document_text,
    }
    final_state = graph.invoke(initial_state)
    return final_state.get("response", {"success": False, "error": "No response generated"})
