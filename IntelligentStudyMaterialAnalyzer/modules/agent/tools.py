"""
tools.py — LangGraph tool nodes.

Each function is a self-contained node:
  * Receives the full AgentState.
  * Calls ONE existing module (or the new ones below).
  * Returns a partial state dict that LangGraph merges back.

Existing modules called (unchanged):
  modules.summarizer         → generate_summary
  modules.mcq_generator      → generate_mcqs
  modules.keyword_extractor  → extract_keywords
  modules.difficulty_analyzer→ analyze_difficulty
  modules.qa_system          → answer_question

New tools (implemented here, no extra files needed for Feature 1):
  flashcard_tool             → generates front/back flashcard pairs
  revision_notes_tool        → generates concise bullet-style revision notes

Both new tools call Claude claude-sonnet-4-6 directly so they integrate
naturally with the Anthropic API already used in your stack, and they
do NOT require any new pip packages beyond what you already have.
"""

from __future__ import annotations
import json
import logging
import urllib.request
from typing import Any

from .state import AgentState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: call Claude claude-sonnet-4-6 and return the raw text reply
# ─────────────────────────────────────────────────────────────────────────────

def _claude(system: str, user: str, max_tokens: int = 1500) -> str:
    payload = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Wrappers around your existing modules
# ─────────────────────────────────────────────────────────────────────────────

def summarize_tool(state: AgentState) -> AgentState:
    """Calls the existing generate_summary module."""
    try:
        from modules.summarizer import generate_summary
        text = state["document_text"]
        summary = generate_summary(text)
        logger.info("[tool:summarize] done")
        return {**state, "summary": summary}
    except Exception as e:
        logger.error(f"[tool:summarize] {e}")
        return {**state, "error": str(e)}


def mcq_tool(state: AgentState) -> AgentState:
    """Calls the existing generate_mcqs module with optional count/difficulty."""
    try:
        from modules.mcq_generator import generate_mcqs
        kwargs: dict[str, Any] = state.get("tool_kwargs", {})
        count = int(kwargs.get("count", 5))
        # generate_mcqs signature: (text, count) — difficulty is an
        # enhancement you can add later inside mcq_generator.py
        mcqs = generate_mcqs(state["document_text"], count=count)
        logger.info(f"[tool:mcq] generated {len(mcqs)} MCQs")
        return {**state, "mcqs": mcqs}
    except Exception as e:
        logger.error(f"[tool:mcq] {e}")
        return {**state, "error": str(e)}


def keyword_tool(state: AgentState) -> AgentState:
    """Calls the existing extract_keywords module."""
    try:
        from modules.keyword_extractor import extract_keywords
        kwargs: dict[str, Any] = state.get("tool_kwargs", {})
        top_n = int(kwargs.get("count", 10))
        kw_list = extract_keywords(state["document_text"], top_n=top_n)
        keywords = [kw[0] for kw in kw_list]   # keep only the word, drop score
        logger.info(f"[tool:keyword] extracted {len(keywords)} keywords")
        return {**state, "keywords": keywords}
    except Exception as e:
        logger.error(f"[tool:keyword] {e}")
        return {**state, "error": str(e)}


def analytics_tool(state: AgentState) -> AgentState:
    """Calls the existing analyze_difficulty module."""
    try:
        from modules.difficulty_analyzer import analyze_difficulty
        diff_data = analyze_difficulty(state["document_text"])
        logger.info(f"[tool:analytics] difficulty={diff_data.get('difficulty_level')}")
        return {**state, "analytics": diff_data}
    except Exception as e:
        logger.error(f"[tool:analytics] {e}")
        return {**state, "error": str(e)}


def qa_tool(state: AgentState) -> AgentState:
    """Calls the existing answer_question module."""
    try:
        from modules.qa_system import answer_question
        kwargs: dict[str, Any] = state.get("tool_kwargs", {})
        # The question might have been parsed by the router, or it IS the
        # user_request itself (e.g. "What is photosynthesis?")
        question = kwargs.get("question") or state.get("user_request", "")
        answer = answer_question(state["document_text"], question)
        logger.info("[tool:qa] answered")
        return {**state, "qa_answer": answer}
    except Exception as e:
        logger.error(f"[tool:qa] {e}")
        return {**state, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# NEW TOOL 1 — Flashcard Generator
# ─────────────────────────────────────────────────────────────────────────────

_FLASHCARD_SYSTEM = """
You are an expert study assistant that creates flashcards from academic text.
Generate exactly {count} flashcards.

Rules:
- Each card has a short FRONT (concept/term/question, ≤15 words)
  and a concise BACK (definition/answer, ≤40 words).
- Cover the most important concepts; avoid trivial details.
- Respond ONLY with a valid JSON array — no markdown, no preamble.

Format:
[
  {{"front": "...", "back": "..."}},
  ...
]
"""

def _fallback_generate_flashcards(text: str, count: int = 10) -> list[dict[str, str]]:
    from modules.keyword_extractor import extract_keywords
    from nltk.tokenize import sent_tokenize
    
    # 1. Get top keywords
    kw_list = extract_keywords(text, top_n=count * 2)
    keywords = [kw[0] for kw in kw_list]
    
    # 2. Get sentences
    sentences = sent_tokenize(text)
    
    flashcards = []
    used_sentences = set()
    
    for kw in keywords:
        if len(flashcards) >= count:
            break
        # Find a sentence containing this keyword
        for sent in sentences:
            sent_clean = sent.strip()
            if kw.lower() in sent_clean.lower() and sent_clean not in used_sentences:
                # Limit sentence length to a reasonable size
                sent_words = sent_clean.split()
                if 8 < len(sent_words) < 40:
                    flashcards.append({
                        "front": f"What is the significance or definition of '{kw.title()}'?",
                        "back": sent_clean
                    })
                    used_sentences.add(sent_clean)
                    break
                    
    # Heuristic fallback if not enough cards generated
    if len(flashcards) < count:
        for sent in sentences:
            if len(flashcards) >= count:
                break
            sent_clean = sent.strip()
            sent_words = sent_clean.split()
            if 12 < len(sent_words) < 35 and sent_clean not in used_sentences:
                front = " ".join(sent_words[:4]) + "..."
                back = sent_clean
                flashcards.append({
                    "front": front,
                    "back": back
                })
                used_sentences.add(sent_clean)
                
    return flashcards


def _fallback_generate_revision_notes(text: str) -> str:
    from modules.keyword_extractor import extract_keywords
    from modules.summarizer import generate_summary
    from nltk.tokenize import sent_tokenize
    
    # 1. Get keywords
    kw_list = extract_keywords(text, top_n=8)
    keywords = [kw[0].title() for kw in kw_list]
    
    # 2. Get summary and split into sentences
    summary_text = generate_summary(text)
    summary_sentences = sent_tokenize(summary_text)
    
    notes = []
    notes.append("# Study Revision Notes\n")
    
    # Use keywords to group sentences
    used_sentences = set()
    
    for kw in keywords:
        section_sentences = []
        for sent in summary_sentences:
            if kw.lower() in sent.lower() and sent not in used_sentences:
                section_sentences.append(sent)
                used_sentences.add(sent)
        
        if section_sentences:
            notes.append(f"## Key Concept: {kw}")
            for sent in section_sentences[:3]:
                notes.append(f"- {sent}")
            notes.append("")
            
    # Put remaining sentences in general summary
    remaining = [sent for sent in summary_sentences if sent not in used_sentences]
    if remaining:
        notes.append("## General Overview")
        for sent in remaining[:5]:
            notes.append(f"- {sent}")
        notes.append("")
        
    notes.append("## Key Takeaways")
    # Take top 3-4 sentences from general summary
    for i, sent in enumerate(summary_sentences[:4]):
        notes.append(f"- {i+1}. **Point {i+1}:** {sent}")
        
    return "\n".join(notes)


def flashcard_tool(state: AgentState) -> AgentState:
    """NEW — generates flashcards by calling Claude. Falls back to local NLP generators if unauthorized."""
    try:
        kwargs: dict[str, Any] = state.get("tool_kwargs", {})
        count = int(kwargs.get("count", 10))

        system = _FLASHCARD_SYSTEM.format(count=count)
        # Truncate to ~3000 words to stay well within the context budget
        text_snippet = " ".join(state["document_text"].split()[:3000])
        user_msg = f"Create {count} flashcards from this study material:\n\n{text_snippet}"

        try:
            raw = _claude(system, user_msg, max_tokens=1000)
            # Strip any accidental markdown fences before parsing
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            flashcards: list[dict] = json.loads(raw)
            logger.info(f"[tool:flashcard] generated {len(flashcards)} cards via Claude")
        except Exception as api_err:
            logger.warning(f"[tool:flashcard] Claude API failed ({api_err}), running local fallback")
            flashcards = _fallback_generate_flashcards(state["document_text"], count)
            logger.info(f"[tool:flashcard] generated {len(flashcards)} cards via local fallback")

        return {**state, "flashcards": flashcards}
    except Exception as e:
        logger.error(f"[tool:flashcard] {e}")
        return {**state, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# NEW TOOL 2 — Revision Notes Generator
# ─────────────────────────────────────────────────────────────────────────────

_REVISION_SYSTEM = """
You are an expert tutor creating concise revision notes for a student.

Instructions:
- Produce well-structured revision notes in Markdown.
- Use ## headings for major topics, bullet points for sub-points.
- Highlight key terms in **bold**.
- Keep each bullet to one sentence.
- End with a "## Key Takeaways" section (max 5 bullets).
- Do NOT reproduce the source text verbatim — paraphrase and distil.
- Output ONLY the Markdown — no preamble, no triple-backtick fences.
"""

def revision_notes_tool(state: AgentState) -> AgentState:
    """NEW — generates structured revision notes. Falls back to local NLP generators if unauthorized."""
    try:
        # Truncate to ~4000 words
        text_snippet = " ".join(state["document_text"].split()[:4000])
        user_msg = f"Generate revision notes from this study material:\n\n{text_snippet}"
        try:
            notes = _claude(_REVISION_SYSTEM, user_msg, max_tokens=1500)
            logger.info("[tool:revision_notes] done via Claude")
        except Exception as api_err:
            logger.warning(f"[tool:revision_notes] Claude API failed ({api_err}), running local fallback")
            notes = _fallback_generate_revision_notes(state["document_text"])
            logger.info("[tool:revision_notes] done via local fallback")
        return {**state, "revision_notes": notes}
    except Exception as e:
        logger.error(f"[tool:revision_notes] {e}")
        return {**state, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# FULL ANALYSIS — runs the existing pipeline (same as /api/analyze)
# ─────────────────────────────────────────────────────────────────────────────

def full_analysis_tool(state: AgentState) -> AgentState:
    """
    Orchestrates the existing four-step pipeline in one shot.
    Equivalent to calling /api/analyze but driven from inside the agent.
    """
    state = summarize_tool(state)
    if state.get("error"):
        return state
    state = mcq_tool(state)
    if state.get("error"):
        return state
    state = keyword_tool(state)
    if state.get("error"):
        return state
    state = analytics_tool(state)
    return state
