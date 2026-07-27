"""
url_routes.py

Flask Blueprint: Feature 2 — Website URL Ingestion.

Registers these routes (all under /api/url):
  POST   /api/url/ingest              — scrape a URL → store as study material
  GET    /api/url/list                — list all URL-type uploads
  GET    /api/url/metadata/<id>       — rich metadata for a URL upload
  POST   /api/url/re-scrape/<id>      — re-fetch and update an existing URL upload
  GET    /api/url/preview/<id>        — first 500 words of extracted content

After ingestion, the upload_id is identical to a file upload —
callers use the same existing endpoints:
  POST /api/analyze/<upload_id>       — runs full analysis pipeline
  POST /api/ask/<upload_id>           — Q&A on page content
  GET  /api/download/<upload_id>      — download study guide
  POST /api/agent/ask/<upload_id>     — LangGraph agent (Feature 1)

Integration steps (add to app.py):
────────────────────────────────────
    from url_routes import url_bp, init_url_feature
    app.register_blueprint(url_bp)
    # after init_db():
    init_url_feature()
"""

from __future__ import annotations
import logging

from flask import Blueprint, request, jsonify

# Existing DB helpers — untouched
from modules.database import (
    save_upload, update_upload_texts, get_upload,
    delete_upload, get_all_uploads,
)

# New DB helpers (Feature 2)
from modules.database_url import (
    save_url_metadata, get_url_metadata,
    get_url_uploads, url_already_ingested,
    init_url_tables, delete_url_metadata,
)

# Scraper service
from modules.web_scraper import scrape_url, validate_url, URLValidationError

logger = logging.getLogger(__name__)
url_bp = Blueprint("url_ingestion", __name__, url_prefix="/api/url")


# ── App-level init ────────────────────────────────────────────────────────────

def init_url_feature() -> None:
    """
    Creates url_metadata table if not present.
    Call once at startup after init_db():
        init_url_feature()
    """
    init_url_tables()
    logger.info("[url_routes] URL ingestion feature initialised.")


# ── Internal helper ───────────────────────────────────────────────────────────

def _build_upload_response(upload_id: int, scraped, force_rescrape: bool = False) -> dict:
    """Assembles the standard success payload for ingest/re-scrape."""
    return {
        "success": True,
        "upload_id": upload_id,
        "filename": scraped.filename,
        "source_url": scraped.source_url,
        "page_title": scraped.metadata.title,
        "domain": scraped.metadata.domain,
        "word_count": scraped.metadata.word_count,
        "fetch_time_ms": scraped.metadata.fetch_time_ms,
        "rescrape": force_rescrape,
        "message": (
            "URL content extracted and stored. "
            "You can now call POST /api/analyze/<upload_id> to run full analysis, "
            "or POST /api/agent/ask/<upload_id> for AI-powered interactions."
        ),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@url_bp.route("/ingest", methods=["POST"])
def ingest_url():
    """
    POST /api/url/ingest
    Body: {"url": "https://www.geeksforgeeks.org/java/"}
          {"url": "...", "force": true}   # re-scrape even if already stored

    Scrapes the URL, extracts clean text, and stores it exactly like
    a file upload. Returns upload_id for use with existing endpoints.

    Duplicate detection: if the same URL has been ingested before and
    force=false (default), returns the existing upload_id immediately.
    """
    data = request.get_json(silent=True) or {}
    raw_url = (data.get("url") or "").strip()
    force_rescrape = bool(data.get("force", False))

    if not raw_url:
        return jsonify({"success": False, "error": "No URL provided in request body."}), 400

    # --- Validate URL first (fast, no network) ---
    try:
        validated_url = validate_url(raw_url)
    except URLValidationError as e:
        return jsonify({"success": False, "error": str(e)}), 422

    # --- Duplicate detection ---
    if not force_rescrape:
        existing_id = url_already_ingested(validated_url)
        if existing_id is not None:
            meta = get_url_metadata(existing_id)
            logger.info(f"[url_routes] duplicate URL detected, returning upload_id={existing_id}")
            return jsonify({
                "success": True,
                "upload_id": existing_id,
                "cached": True,
                "source_url": validated_url,
                "page_title": meta.get("page_title", "") if meta else "",
                "domain": meta.get("domain", "") if meta else "",
                "word_count": meta.get("word_count", 0) if meta else 0,
                "message": (
                    "This URL was already ingested. "
                    "Pass force=true to re-scrape. "
                    f"Use upload_id={existing_id} with /api/analyze/ or /api/agent/ask/."
                ),
            })

    # --- Scrape ---
    logger.info(f"[url_routes] scraping: {validated_url}")
    scraped = scrape_url(validated_url)

    if not scraped.success:
        return jsonify({
            "success": False,
            "error": scraped.error,
            "url": validated_url,
        }), 422

    # --- Persist into uploads table (same as file upload pipeline) ---
    try:
        upload_id = save_upload(
            filename=scraped.filename,
            filepath=scraped.source_url,   # filepath stores the URL
            file_type="URL",
        )

        update_upload_texts(
            upload_id=upload_id,
            raw_text=scraped.raw_text,
            processed_text=scraped.processed_text,
        )

        save_url_metadata(
            upload_id=upload_id,
            source_url=scraped.source_url,
            page_title=scraped.metadata.title,
            domain=scraped.metadata.domain,
            page_description=scraped.metadata.description,
            word_count=scraped.metadata.word_count,
            fetch_time_ms=scraped.metadata.fetch_time_ms,
            extraction_status="success",
        )

    except Exception as e:
        logger.error(f"[url_routes] DB save error: {e}")
        # Best-effort cleanup
        try:
            if 'upload_id' in locals():
                delete_upload(upload_id)
        except Exception:
            pass
        return jsonify({
            "success": False,
            "error": f"Content was extracted but could not be saved: {e}",
        }), 500

    logger.info(
        f"[url_routes] ingested upload_id={upload_id} "
        f"url={validated_url} words={scraped.metadata.word_count}"
    )
    return jsonify(_build_upload_response(upload_id, scraped)), 201


@url_bp.route("/list", methods=["GET"])
def list_url_uploads():
    """
    GET /api/url/list?limit=20&offset=0

    Returns all URL-type uploads with their rich metadata.
    """
    try:
        limit = min(int(request.args.get("limit", 20)), 100)
        offset = max(int(request.args.get("offset", 0)), 0)
    except ValueError:
        return jsonify({"success": False, "error": "limit and offset must be integers"}), 400

    uploads = get_url_uploads(limit=limit, offset=offset)
    return jsonify({
        "success": True,
        "count": len(uploads),
        "uploads": uploads,
    })


@url_bp.route("/metadata/<int:upload_id>", methods=["GET"])
def get_url_metadata_route(upload_id: int):
    """
    GET /api/url/metadata/<upload_id>

    Returns rich URL metadata for a given upload.
    Includes the source URL, page title, domain, word count, etc.
    Also returns the base upload record for convenience.
    """
    upload = get_upload(upload_id)
    if not upload:
        return jsonify({"success": False, "error": "Upload not found"}), 404

    if upload.get("file_type") != "URL":
        return jsonify({
            "success": False,
            "error": f"Upload {upload_id} is a {upload.get('file_type')}, not a URL."
        }), 400

    meta = get_url_metadata(upload_id)
    if not meta:
        return jsonify({
            "success": False,
            "error": "URL metadata not found. This upload may predate Feature 2."
        }), 404

    return jsonify({
        "success": True,
        "upload_id": upload_id,
        "upload": {
            "filename": upload["filename"],
            "file_type": upload["file_type"],
            "uploaded_at": upload["uploaded_at"],
        },
        "metadata": meta,
    })


@url_bp.route("/re-scrape/<int:upload_id>", methods=["POST"])
def re_scrape_url(upload_id: int):
    """
    POST /api/url/re-scrape/<upload_id>

    Re-fetches the source URL and updates the stored text + metadata.
    Use this when a page's content has changed since initial ingestion.
    Preserves the upload_id so existing analysis results remain linked.
    """
    upload = get_upload(upload_id)
    if not upload:
        return jsonify({"success": False, "error": "Upload not found"}), 404

    if upload.get("file_type") != "URL":
        return jsonify({
            "success": False,
            "error": "Re-scrape is only available for URL uploads."
        }), 400

    source_url = upload.get("filepath", "")
    if not source_url:
        return jsonify({"success": False, "error": "No source URL found for this upload"}), 400

    logger.info(f"[url_routes] re-scraping upload_id={upload_id} url={source_url}")
    scraped = scrape_url(source_url)

    if not scraped.success:
        return jsonify({
            "success": False,
            "error": scraped.error,
            "upload_id": upload_id,
        }), 422

    try:
        update_upload_texts(upload_id, scraped.raw_text, scraped.processed_text)
        save_url_metadata(
            upload_id=upload_id,
            source_url=scraped.source_url,
            page_title=scraped.metadata.title,
            domain=scraped.metadata.domain,
            page_description=scraped.metadata.description,
            word_count=scraped.metadata.word_count,
            fetch_time_ms=scraped.metadata.fetch_time_ms,
            extraction_status="re-scraped",
        )
    except Exception as e:
        logger.error(f"[url_routes] re-scrape DB error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

    return jsonify(_build_upload_response(upload_id, scraped, force_rescrape=True))


@url_bp.route("/preview/<int:upload_id>", methods=["GET"])
def preview_url_content(upload_id: int):
    """
    GET /api/url/preview/<upload_id>?words=500

    Returns the first N words of the extracted content
    plus the page metadata. Useful for UI previews before analysis.
    """
    try:
        word_limit = min(int(request.args.get("words", 500)), 2000)
    except ValueError:
        word_limit = 500

    upload = get_upload(upload_id)
    if not upload:
        return jsonify({"success": False, "error": "Upload not found"}), 404

    if upload.get("file_type") != "URL":
        return jsonify({
            "success": False,
            "error": f"Upload {upload_id} is not a URL upload."
        }), 400

    processed_text = upload.get("processed_text") or ""
    words = processed_text.split()
    preview = " ".join(words[:word_limit])
    truncated = len(words) > word_limit

    meta = get_url_metadata(upload_id) or {}

    return jsonify({
        "success": True,
        "upload_id": upload_id,
        "source_url": meta.get("source_url", upload.get("filepath", "")),
        "page_title": meta.get("page_title", upload.get("filename", "")),
        "domain": meta.get("domain", ""),
        "total_words": meta.get("word_count", len(words)),
        "preview_words": word_limit,
        "truncated": truncated,
        "preview_text": preview,
    })


@url_bp.route("/validate", methods=["POST"])
def validate_url_route():
    """
    POST /api/url/validate
    Body: {"url": "https://..."}

    Lightweight URL validation without fetching.
    Useful for instant front-end feedback before submission.
    """
    data = request.get_json(silent=True) or {}
    raw_url = (data.get("url") or "").strip()

    if not raw_url:
        return jsonify({"valid": False, "error": "URL cannot be empty."}), 400

    try:
        validated = validate_url(raw_url)
        already_ingested_id = url_already_ingested(validated)
        return jsonify({
            "valid": True,
            "url": validated,
            "already_ingested": already_ingested_id is not None,
            "existing_upload_id": already_ingested_id,
        })
    except URLValidationError as e:
        return jsonify({"valid": False, "error": str(e)}), 422
