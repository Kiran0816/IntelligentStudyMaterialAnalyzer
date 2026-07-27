"""
migrations/add_agent_tables.py

Run this ONCE against your existing analyzer.db to add the two new
tables required by Feature 1.  It is fully additive — it does not
alter or drop any existing table.

Usage:
    python migrations/add_agent_tables.py
    python migrations/add_agent_tables.py --db /path/to/analyzer.db
"""

import argparse
import sqlite3
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Default: db lives next to app.py (one level above migrations/)
DEFAULT_DB = os.path.join(os.path.dirname(__file__), "..", "analyzer.db")

NEW_TABLES_SQL = [
    # ── Flashcards ──────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS flashcards (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        upload_id   INTEGER NOT NULL,
        front       TEXT    NOT NULL,
        back        TEXT    NOT NULL,
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (upload_id) REFERENCES uploads(id) ON DELETE CASCADE
    );
    """,

    # ── Revision Notes ───────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS revision_notes (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        upload_id    INTEGER NOT NULL UNIQUE,   -- one set of notes per upload
        notes_text   TEXT    NOT NULL,
        created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (upload_id) REFERENCES uploads(id) ON DELETE CASCADE
    );
    """,

    # ── Agent Request Log (optional but useful for debugging) ─────────────
    """
    CREATE TABLE IF NOT EXISTS agent_requests (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        upload_id    INTEGER NOT NULL,
        user_request TEXT    NOT NULL,
        intent       TEXT,
        success      INTEGER DEFAULT 1,
        created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (upload_id) REFERENCES uploads(id) ON DELETE CASCADE
    );
    """,
]


def run_migration(db_path: str) -> None:
    db_path = os.path.abspath(db_path)
    if not os.path.exists(db_path):
        logger.warning(f"Database not found at {db_path} — will be created fresh.")

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        for sql in NEW_TABLES_SQL:
            cur.executescript(sql)
        conn.commit()
        logger.info(f"Migration complete → {db_path}")

        # Show what tables now exist
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [r[0] for r in cur.fetchall()]
        logger.info(f"Tables in DB: {tables}")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add agent tables to analyzer.db")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to analyzer.db")
    args = parser.parse_args()
    run_migration(args.db)
