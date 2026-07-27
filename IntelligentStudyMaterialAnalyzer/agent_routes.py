"""
agent_routes.py

Drop-in Flask Blueprint that wires the LangGraph agent into your
existing app.py.  No existing routes are modified.

Integration steps (add to your app.py):
─────────────────────────────────────────
    # 1. Import and register the blueprint
    from agent_routes import agent_bp, init_agent
    app.register_blueprint(agent_bp)

    # 2. Build the graph once after init_db()
    init_agent(app)

That's it.  Your existing routes (/api/upload, /api/analyze, etc.)
are completely untouched.
"""

from __future__ import annotations
import logging

from flask import Blueprint, request, jsonify, current_app

# Existing DB helpers (unchanged)
from modules.database import get_upload, save_summary, save_mcqs, save_keywords, save_analytics

# New DB helpers for agent-only tables
from modules.database_agent import (
    save_flashcards, get_flashcards,
    save_revision_notes, get_revision_notes,
    log_agent_request,
)

# Agent layer
from modules.agent import build_graph, run_agent
from modules.agent.state import (
    INTENT_SUMMARIZE, INTENT_MCQ, INTENT_KEYWORDS, INTENT_ANALYTICS,
    INTENT_FLASHCARDS, INTENT_REVISION_NOTES, INTENT_FULL_ANALYSIS,
)

logger = logging.getLogger(__name__)
agent_bp = Blueprint("agent", __name__, url_prefix="/api/agent")


# ── App-level setup ──────────────────────────────────────────────────────────

def init_agent(app) -> None:
    """
    Build and cache the compiled LangGraph graph on the Flask app object.
    Call after app creation: init_agent(app)
    """
    with app.app_context():
        app.agent_graph = build_graph()
        logger.info("[agent] LangGraph workflow compiled and cached.")


def _get_graph():
    return current_app.agent_graph


# ── Helper: persist agent outputs to DB so they appear in existing routes ───

def _persist_outputs(upload_id: int, intent: str, result: dict) -> None:
    """
    Re-uses your existing save_* functions so agent-generated results
    are visible through the original /api/analyze endpoint as well.
    """
    try:
        if intent in (INTENT_SUMMARIZE, INTENT_FULL_ANALYSIS) and result.get("summary"):
            save_summary(upload_id, result["summary"])

        if intent in (INTENT_MCQ, INTENT_FULL_ANALYSIS) and result.get("mcqs"):
            save_mcqs(upload_id, result["mcqs"])

        if intent in (INTENT_KEYWORDS, INTENT_FULL_ANALYSIS) and result.get("keywords"):
            # save_keywords expects [(word, score), ...]; agent returns [word, ...]
            save_keywords(upload_id, [(kw, 0.0) for kw in result["keywords"]])

        if intent in (INTENT_ANALYTICS, INTENT_FULL_ANALYSIS) and result.get("analytics"):
            d = result["analytics"]
            save_analytics(
                upload_id,
                d.get("difficulty_level", "Unknown"),
                d.get("sentence_count", 0),
                d.get("word_count", 0),
                d.get("estimated_study_time", 0),
            )

        if intent == INTENT_FLASHCARDS and result.get("flashcards"):
            save_flashcards(upload_id, result["flashcards"])

        if intent == INTENT_REVISION_NOTES and result.get("revision_notes"):
            save_revision_notes(upload_id, result["revision_notes"])

    except Exception as e:
        logger.warning(f"[agent_routes] non-critical persist error: {e}")


# ── Routes ────────────────────────────────────────────────────────────────────

@agent_bp.route("/ask/<int:upload_id>", methods=["POST"])
def agent_ask(upload_id: int):
    """
    POST /api/agent/ask/<upload_id>
    Body: {"request": "Generate 5 hard MCQs"}

    The single entry point for ALL agent interactions.
    The LangGraph router decides which tool to call.
    """
    data = request.get_json(silent=True) or {}
    user_request = (data.get("request") or data.get("question") or "").strip()

    if not user_request:
        return jsonify({"success": False, "error": "No request provided"}), 400

    upload = get_upload(upload_id)
    if not upload:
        return jsonify({"success": False, "error": "Upload not found"}), 404

    document_text = upload.get("processed_text") or ""
    if not document_text:
        return jsonify({"success": False, "error": "No text content available"}), 400

    graph = _get_graph()
    result = run_agent(graph, upload_id, document_text, user_request)

    # Persist outputs back to DB (best-effort)
    intent = result.get("intent", "")
    _persist_outputs(upload_id, intent, result)

    # Log the request
    log_agent_request(upload_id, user_request, intent, success=result.get("success", False))

    return jsonify(result)


@agent_bp.route("/flashcards/<int:upload_id>", methods=["GET"])
def get_flashcards_route(upload_id: int):
    """
    GET /api/agent/flashcards/<upload_id>
    Returns cached flashcards for this upload (if any).
    """
    cards = get_flashcards(upload_id)
    if cards is None:
        return jsonify({
            "success": False,
            "error": "No flashcards found. POST to /api/agent/ask/<id> with 'Generate flashcards'."
        }), 404
    return jsonify({"success": True, "flashcards": cards, "count": len(cards)})


@agent_bp.route("/revision-notes/<int:upload_id>", methods=["GET"])
def get_revision_notes_route(upload_id: int):
    """
    GET /api/agent/revision-notes/<upload_id>
    Returns cached revision notes (Markdown string).
    """
    notes = get_revision_notes(upload_id)
    if notes is None:
        return jsonify({
            "success": False,
            "error": "No revision notes found. POST to /api/agent/ask/<id> with 'Create revision notes'."
        }), 404
    return jsonify({"success": True, "revision_notes": notes})


@agent_bp.route("/capabilities", methods=["GET"])
def capabilities():
    """
    GET /api/agent/capabilities
    Returns a description of what the agent can do — useful for frontend UI hints.
    """
    return jsonify({
        "success": True,
        "capabilities": [
            {
                "intent": "summarize",
                "description": "Summarize the document",
                "example_requests": ["Summarize this document", "Give me a brief overview", "TL;DR"]
            },
            {
                "intent": "generate_mcqs",
                "description": "Generate multiple-choice questions",
                "example_requests": ["Generate 5 MCQs", "Create 10 hard quiz questions", "Practice questions"]
            },
            {
                "intent": "extract_keywords",
                "description": "Extract key concepts and terms",
                "example_requests": ["What are the key topics?", "Extract keywords", "List key concepts"]
            },
            {
                "intent": "analyze_difficulty",
                "description": "Assess reading difficulty and estimate study time",
                "example_requests": ["How hard is this?", "Difficulty analysis", "How long to study?"]
            },
            {
                "intent": "answer_question",
                "description": "Answer a question based on the document",
                "example_requests": ["What is photosynthesis?", "Explain inheritance", "How does X work?"]
            },
            {
                "intent": "generate_flashcards",
                "description": "Generate front/back study flashcards",
                "example_requests": ["Create flashcards", "Make 15 flashcards", "Flash cards for revision"]
            },
            {
                "intent": "generate_revision_notes",
                "description": "Generate structured Markdown revision notes",
                "example_requests": ["Create revision notes", "Study notes", "Cheat sheet"]
            },
            {
                "intent": "full_analysis",
                "description": "Run the complete analysis pipeline",
                "example_requests": ["Full analysis", "Analyze everything", "Run all tools"]
            },
        ]
    })
