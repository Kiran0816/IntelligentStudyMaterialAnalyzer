"""
router.py — Intent classification node for the LangGraph agent.

This is the FIRST node every request passes through.  It reads
state["user_request"] and writes state["intent"] + state["tool_kwargs"].

Strategy
--------
We call Claude claude-sonnet-4-6 with a structured JSON prompt so the LLM
becomes a zero-shot intent classifier.  This avoids brittle regex while
remaining fully deterministic for callers (the output is always one of
the ALL_INTENTS constants).

Fallback: if the API call fails we fall back to a lightweight keyword
matcher so the agent degrades gracefully even without a live API.
"""

from __future__ import annotations
import json
import logging
import re
from typing import Any

from .state import (
    AgentState,
    INTENT_SUMMARIZE, INTENT_MCQ, INTENT_KEYWORDS, INTENT_ANALYTICS,
    INTENT_QA, INTENT_FLASHCARDS, INTENT_REVISION_NOTES,
    INTENT_FULL_ANALYSIS, INTENT_UNKNOWN, ALL_INTENTS,
)

logger = logging.getLogger(__name__)

# ── Keyword fallback map (order matters — more specific first) ─────────────
_KEYWORD_MAP: list[tuple[list[str], str]] = [
    (["revision note", "revision notes", "study note", "cheat sheet"], INTENT_REVISION_NOTES),
    (["flashcard", "flash card", "flash-card"],                         INTENT_FLASHCARDS),
    (["mcq", "multiple choice", "quiz", "interview question",
      "practice question", "hard question", "easy question"],           INTENT_MCQ),
    (["keyword", "key concept", "key term", "topic"],                   INTENT_KEYWORDS),
    (["difficulty", "reading level", "study time", "analytics"],        INTENT_ANALYTICS),
    (["summarize", "summary", "summarise", "tldr", "overview",
      "brief", "gist"],                                                 INTENT_SUMMARIZE),
    (["full analysis", "analyze", "analyse", "run analysis"],           INTENT_FULL_ANALYSIS),
]


def _keyword_fallback(request: str) -> tuple[str, dict[str, Any]]:
    """Cheap fallback when the LLM is unavailable."""
    lower = request.lower()
    for keywords, intent in _KEYWORD_MAP:
        if any(kw in lower for kw in keywords):
            kwargs: dict[str, Any] = {}
            # Parse count hint e.g. "10 MCQs"
            m = re.search(r"(\d+)\s*(mcq|question|flashcard)", lower)
            if m:
                kwargs["count"] = int(m.group(1))
            # Parse difficulty hint
            for level in ("easy", "medium", "hard"):
                if level in lower:
                    kwargs["difficulty"] = level
            return intent, kwargs
    # If the request looks like a question, route to QA
    if "?" in request or lower.startswith(("what", "how", "why", "when",
                                            "who", "explain", "define")):
        return INTENT_QA, {"question": request}
    return INTENT_UNKNOWN, {}


def _llm_classify(request: str) -> tuple[str, dict[str, Any]]:
    """
    Call Claude claude-sonnet-4-6 to classify the intent.
    Returns (intent_string, tool_kwargs_dict).
    """
    import urllib.request

    system_prompt = (
        "You are an intent classifier for an AI study assistant. "
        "Classify the user's request into exactly one intent and extract "
        "optional parameters.  Respond ONLY with a valid JSON object — "
        "no preamble, no markdown fences.\n\n"
        "Valid intents:\n"
        f"  {json.dumps(sorted(ALL_INTENTS))}\n\n"
        "Optional params you may include in 'kwargs':\n"
        "  count (int)       — number of items requested (MCQs, flashcards, etc.)\n"
        "  difficulty (str)  — 'easy' | 'medium' | 'hard'\n"
        "  question (str)    — the verbatim question when intent is answer_question\n\n"
        "Output format:\n"
        '{"intent": "<intent>", "kwargs": {<optional params>}}'
    )

    payload = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 200,
        "system": system_prompt,
        "messages": [{"role": "user", "content": request}],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())

    raw = data["content"][0]["text"].strip()
    parsed = json.loads(raw)
    intent = parsed.get("intent", INTENT_UNKNOWN)
    if intent not in ALL_INTENTS:
        intent = INTENT_UNKNOWN
    return intent, parsed.get("kwargs", {})


def router_node(state: AgentState) -> AgentState:
    """
    LangGraph node: classifies intent and writes it back to state.

    Reads:  state["user_request"]
    Writes: state["intent"], state["tool_kwargs"]
    """
    request = state.get("user_request", "").strip()
    if not request:
        return {**state, "intent": INTENT_UNKNOWN, "tool_kwargs": {}}

    try:
        intent, kwargs = _llm_classify(request)
        logger.info(f"[router] LLM classified → intent={intent} kwargs={kwargs}")
    except Exception as exc:
        logger.warning(f"[router] LLM classify failed ({exc}), using keyword fallback")
        intent, kwargs = _keyword_fallback(request)
        logger.info(f"[router] keyword fallback → intent={intent} kwargs={kwargs}")

    return {**state, "intent": intent, "tool_kwargs": kwargs}
