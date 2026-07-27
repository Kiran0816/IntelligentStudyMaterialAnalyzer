"""
modules/database_agent.py

New DB helper functions for the agent layer.

These extend your existing modules/database.py WITHOUT modifying it.
Import alongside the existing module:

    from modules.database import get_upload, save_summary, ...
    from modules.database_agent import (
        save_flashcards, get_flashcards,
        save_revision_notes, get_revision_notes,
        log_agent_request,
    )
"""

from __future__ import annotations
import json
import sqlite3
import logging
import os

logger = logging.getLogger(__name__)

# Reuse the same DB path logic as your existing database.py
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "analyzer.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── Flashcards ───────────────────────────────────────────────────────────────

def save_flashcards(upload_id: int, cards: list[dict]) -> None:
    """
    Persists a list of flashcard dicts [{front, back}] for an upload.
    Replaces any previously stored cards for the same upload_id.
    """
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM flashcards WHERE upload_id = ?", (upload_id,))
        cur.executemany(
            "INSERT INTO flashcards (upload_id, front, back) VALUES (?, ?, ?)",
            [(upload_id, c["front"], c["back"]) for c in cards],
        )
        conn.commit()
        logger.info(f"Saved {len(cards)} flashcards for upload_id={upload_id}")
    finally:
        conn.close()


def get_flashcards(upload_id: int) -> list[dict] | None:
    """Returns cached flashcards or None if none exist."""
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT front, back FROM flashcards WHERE upload_id = ? ORDER BY id",
            (upload_id,),
        )
        rows = cur.fetchall()
        if not rows:
            return None
        return [{"front": r["front"], "back": r["back"]} for r in rows]
    finally:
        conn.close()


# ── Revision Notes ────────────────────────────────────────────────────────────

def save_revision_notes(upload_id: int, notes_text: str) -> None:
    """Upserts revision notes for an upload (one set per upload)."""
    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT INTO revision_notes (upload_id, notes_text)
            VALUES (?, ?)
            ON CONFLICT(upload_id) DO UPDATE SET
                notes_text = excluded.notes_text,
                created_at = CURRENT_TIMESTAMP
            """,
            (upload_id, notes_text),
        )
        conn.commit()
        logger.info(f"Saved revision notes for upload_id={upload_id}")
    finally:
        conn.close()


def get_revision_notes(upload_id: int) -> str | None:
    """Returns cached revision notes text or None."""
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT notes_text FROM revision_notes WHERE upload_id = ?",
            (upload_id,),
        )
        row = cur.fetchone()
        return row["notes_text"] if row else None
    finally:
        conn.close()


# ── Agent Request Log ─────────────────────────────────────────────────────────

def log_agent_request(
    upload_id: int,
    user_request: str,
    intent: str | None,
    success: bool = True,
) -> None:
    """Logs every agent invocation for debugging / analytics."""
    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT INTO agent_requests (upload_id, user_request, intent, success)
            VALUES (?, ?, ?, ?)
            """,
            (upload_id, user_request, intent, 1 if success else 0),
        )
        conn.commit()
    except Exception as e:
        logger.warning(f"Failed to log agent request: {e}")
    finally:
        conn.close()
