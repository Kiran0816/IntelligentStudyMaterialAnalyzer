"""
modules/database_url.py

Additive DB layer for Feature 2 (URL Ingestion).

Zero changes to your existing database.py or schema.
The uploads table already stores URLs perfectly:
  - uploads.filename   = slugified page title  (e.g. "java-oop-concepts.url")
  - uploads.filepath   = the actual URL         (e.g. "https://geeksforgeeks.org/...")
  - uploads.file_type  = "URL"
  - uploads.raw_text   = scraped raw text
  - uploads.processed_text = cleaned text

This module adds ONE new table (url_metadata) that stores the
rich page metadata (title, domain, description, fetch stats)
that doesn't fit the generic uploads schema.

Import alongside your existing database.py:

    from modules.database import get_upload, save_summary, ...
    from modules.database_url import save_url_metadata, get_url_metadata, get_url_uploads
"""

from __future__ import annotations
import logging
import os
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)

# Mirror the same DB path pattern as your existing database.py
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "analyzer.db")


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


# ── url_metadata table ────────────────────────────────────────────────────────

def init_url_tables() -> None:
    """
    Creates the url_metadata table if it doesn't exist.
    Call this once at app startup, after your existing init_db().
    Safe to call multiple times (IF NOT EXISTS).
    """
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS url_metadata (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_id      INTEGER NOT NULL UNIQUE,
                source_url     TEXT    NOT NULL,
                page_title     TEXT    NOT NULL DEFAULT 'Untitled Page',
                page_description TEXT  DEFAULT '',
                domain         TEXT    NOT NULL DEFAULT '',
                word_count     INTEGER NOT NULL DEFAULT 0,
                fetch_time_ms  INTEGER DEFAULT 0,
                extraction_status TEXT DEFAULT 'success',
                created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (upload_id) REFERENCES uploads(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_url_metadata_upload_id
                ON url_metadata(upload_id);

            CREATE INDEX IF NOT EXISTS idx_url_metadata_domain
                ON url_metadata(domain);
        """)
    logger.info("[db_url] url_metadata table ready")


# ── CRUD helpers ──────────────────────────────────────────────────────────────

def save_url_metadata(
    upload_id: int,
    source_url: str,
    page_title: str,
    domain: str,
    page_description: str = "",
    word_count: int = 0,
    fetch_time_ms: int = 0,
    extraction_status: str = "success",
) -> None:
    """
    Inserts or replaces url_metadata for a given upload_id.
    Safe to call multiple times (upsert via ON CONFLICT).
    """
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO url_metadata
                (upload_id, source_url, page_title, page_description,
                 domain, word_count, fetch_time_ms, extraction_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(upload_id) DO UPDATE SET
                source_url         = excluded.source_url,
                page_title         = excluded.page_title,
                page_description   = excluded.page_description,
                domain             = excluded.domain,
                word_count         = excluded.word_count,
                fetch_time_ms      = excluded.fetch_time_ms,
                extraction_status  = excluded.extraction_status,
                created_at         = CURRENT_TIMESTAMP
            """,
            (upload_id, source_url, page_title, page_description,
             domain, word_count, fetch_time_ms, extraction_status),
        )
    logger.info(f"[db_url] saved metadata for upload_id={upload_id} domain={domain}")


def get_url_metadata(upload_id: int) -> Optional[dict]:
    """
    Returns url_metadata for an upload, or None if not a URL upload.
    """
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM url_metadata WHERE upload_id = ?", (upload_id,)
        ).fetchone()
    return dict(row) if row else None


def get_url_uploads(limit: int = 50, offset: int = 0) -> list[dict]:
    """
    Returns all uploads of file_type='URL', joined with url_metadata.
    Useful for a URL-specific listing endpoint.
    """
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT
                u.id, u.filename, u.filepath, u.file_type,
                u.uploaded_at,
                um.source_url, um.page_title, um.domain,
                um.page_description, um.word_count, um.fetch_time_ms
            FROM uploads u
            LEFT JOIN url_metadata um ON um.upload_id = u.id
            WHERE u.file_type = 'URL'
            ORDER BY u.uploaded_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


def url_already_ingested(source_url: str) -> Optional[int]:
    """
    Checks if a URL has already been scraped.
    Returns the existing upload_id, or None if not found.

    Used to avoid redundant scraping of the same URL.
    """
    with _conn() as conn:
        row = conn.execute(
            "SELECT upload_id FROM url_metadata WHERE source_url = ?",
            (source_url,),
        ).fetchone()
    return row["upload_id"] if row else None


def delete_url_metadata(upload_id: int) -> None:
    """
    Deletes url_metadata for an upload.
    Note: ON DELETE CASCADE on the FK handles this automatically
    when delete_upload() is called — this is an explicit override.
    """
    with _conn() as conn:
        conn.execute(
            "DELETE FROM url_metadata WHERE upload_id = ?", (upload_id,)
        )
    logger.info(f"[db_url] deleted url_metadata for upload_id={upload_id}")
