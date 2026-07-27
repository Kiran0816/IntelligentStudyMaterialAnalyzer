"""
migrations/add_url_tables.py

Adds the url_metadata table required by Feature 2 (URL Ingestion).
Fully additive — zero changes to existing tables.

Run ONCE against your analyzer.db:
    python migrations/add_url_tables.py
    python migrations/add_url_tables.py --db /path/to/analyzer.db

Safe to run multiple times (all CREATE statements use IF NOT EXISTS).
"""

import argparse
import logging
import os
import sqlite3

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DB = os.path.join(os.path.dirname(__file__), "..", "analyzer.db")

MIGRATION_SQL = """
-- ── url_metadata ─────────────────────────────────────────────────────────────
-- Stores rich page metadata for URL-type uploads.
-- The uploads table already handles the text content (raw_text, processed_text)
-- and uses filepath to store the source URL and file_type='URL' as the marker.

CREATE TABLE IF NOT EXISTS url_metadata (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    upload_id          INTEGER NOT NULL UNIQUE,
    source_url         TEXT    NOT NULL,
    page_title         TEXT    NOT NULL DEFAULT 'Untitled Page',
    page_description   TEXT             DEFAULT '',
    domain             TEXT    NOT NULL DEFAULT '',
    word_count         INTEGER NOT NULL DEFAULT 0,
    fetch_time_ms      INTEGER          DEFAULT 0,
    extraction_status  TEXT             DEFAULT 'success',
    created_at         TIMESTAMP        DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (upload_id) REFERENCES uploads(id) ON DELETE CASCADE
);

-- Index for fast upload_id lookup (used in every GET)
CREATE INDEX IF NOT EXISTS idx_url_metadata_upload_id
    ON url_metadata(upload_id);

-- Index for domain-grouped queries and duplicate detection by URL
CREATE INDEX IF NOT EXISTS idx_url_metadata_domain
    ON url_metadata(domain);

CREATE INDEX IF NOT EXISTS idx_url_metadata_source_url
    ON url_metadata(source_url);
"""


def run_migration(db_path: str) -> None:
    db_path = os.path.abspath(db_path)

    if not os.path.exists(db_path):
        logger.warning(f"Database not found at {db_path} — will be created.")

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(MIGRATION_SQL)
        conn.commit()

        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [r[0] for r in cur.fetchall()]
        logger.info(f"Migration complete → {db_path}")
        logger.info(f"All tables: {tables}")

        # Verify url_metadata columns
        cur.execute("PRAGMA table_info(url_metadata)")
        cols = [r[1] for r in cur.fetchall()]
        logger.info(f"url_metadata columns: {cols}")

    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add url_metadata table to analyzer.db (Feature 2)"
    )
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to analyzer.db")
    args = parser.parse_args()
    run_migration(args.db)
