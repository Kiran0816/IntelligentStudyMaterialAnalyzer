"""
AgentState — the single shared data structure that flows through every
node in the LangGraph workflow.

Design rules:
  * ALL fields are Optional so nodes can add data incrementally.
  * The graph never mutates a field once another node has written it;
    it only appends to lists or sets previously-None scalars.
  * Flask routes build the initial state and read the final state —
    they never touch LangGraph internals directly.
"""

from __future__ import annotations
from typing import Any, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────
    upload_id: int                     # FK → uploads.id
    user_request: str                  # Raw user message, e.g. "summarize this"
    document_text: str                 # processed_text pulled from DB

    # ── Router decision ───────────────────────────────────────────────────
    intent: str                        # One of the INTENT_* constants below
    tool_kwargs: dict[str, Any]        # Extra params parsed from the request
                                       # e.g. {"difficulty": "hard", "count": 10}

    # ── Tool outputs (each is None until the relevant tool runs) ──────────
    summary: Optional[str]
    mcqs: Optional[list[dict]]         # [{question, options, correct_answer}]
    keywords: Optional[list[str]]
    analytics: Optional[dict]
    qa_answer: Optional[str]
    flashcards: Optional[list[dict]]   # [{front, back}]
    revision_notes: Optional[str]

    # ── Final response assembled by the responder node ────────────────────
    response: Optional[dict]           # JSON-serialisable payload sent to Flask

    # ── Error channel ─────────────────────────────────────────────────────
    error: Optional[str]


# ── Intent constants ──────────────────────────────────────────────────────
# The router node sets state["intent"] to one of these values.
INTENT_SUMMARIZE       = "summarize"
INTENT_MCQ             = "generate_mcqs"
INTENT_KEYWORDS        = "extract_keywords"
INTENT_ANALYTICS       = "analyze_difficulty"
INTENT_QA              = "answer_question"
INTENT_FLASHCARDS      = "generate_flashcards"
INTENT_REVISION_NOTES  = "generate_revision_notes"
INTENT_FULL_ANALYSIS   = "full_analysis"   # runs the existing /api/analyze pipeline
INTENT_UNKNOWN         = "unknown"

ALL_INTENTS = {
    INTENT_SUMMARIZE,
    INTENT_MCQ,
    INTENT_KEYWORDS,
    INTENT_ANALYTICS,
    INTENT_QA,
    INTENT_FLASHCARDS,
    INTENT_REVISION_NOTES,
    INTENT_FULL_ANALYSIS,
}
