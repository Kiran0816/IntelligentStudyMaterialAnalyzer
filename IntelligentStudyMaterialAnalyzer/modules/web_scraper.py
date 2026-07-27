"""
modules/web_scraper.py

Web content ingestion service for the Study Companion.

Architecture
────────────
Layer 1  — URL validation (security: blocks private IPs, bad schemes)
Layer 2  — HTTP fetching  (requests with realistic headers + timeout)
Layer 3A — Trafilatura extraction  (primary: ML-based boilerplate removal)
Layer 3B — BeautifulSoup extraction (fallback: tag-based cleaning)
Layer 4  — Post-processing  (clean whitespace, min-length guard)
Layer 5  — Metadata extraction  (title, description, domain, word count)

The public API is a single function:
    scrape_url(url: str) -> ScrapedContent

That result plugs directly into the existing upload pipeline:
    save_upload(filename, filepath=url, file_type='URL')
    update_upload_texts(upload_id, raw_text, processed_text)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum extractable text to consider a page usable
MIN_WORD_COUNT = 50

# Maximum words we'll store (prevents absurdly large DB entries)
MAX_WORD_COUNT = 50_000

REQUEST_TIMEOUT = 15      # seconds
MAX_REDIRECTS   = 5
MAX_URL_LENGTH  = 2048

# Hosts/prefixes that are never allowed (SSRF prevention)
_BLOCKED_HOSTS: frozenset[str] = frozenset({
    "localhost", "127.0.0.1", "0.0.0.0", "::1",
    "169.254.169.254",  # AWS metadata
    "metadata.google.internal",
})
_BLOCKED_PREFIXES: tuple[str, ...] = ("192.168.", "10.", "172.16.", "172.17.")

# Realistic browser headers to avoid simple bot-detection
_REQUEST_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "DNT": "1",
}

# CSS class / id keywords that typically mark noise elements
_NOISE_PATTERNS: tuple[str, ...] = (
    "ad", "ads", "advert", "advertisement", "banner", "popup",
    "sidebar", "widget", "cookie", "newsletter", "subscribe",
    "social", "share", "comment", "related", "recommended",
    "promo", "sponsor", "footer", "header", "nav", "navigation",
    "menu", "breadcrumb", "pagination", "toolbar",
)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PageMetadata:
    title: str       = "Untitled Page"
    description: str = ""
    domain: str      = ""
    source_url: str  = ""
    word_count: int  = 0
    fetch_time_ms: int = 0


@dataclass
class ScrapedContent:
    """Returned by scrape_url() — maps directly to DB columns."""
    raw_text: str              # Full extracted text before any trimming
    processed_text: str        # Cleaned, whitespace-normalised text
    filename: str              # Slug used as the 'filename' in uploads table
    source_url: str            # Stored in uploads.filepath
    metadata: PageMetadata     = field(default_factory=PageMetadata)
    success: bool              = True
    error: Optional[str]       = None


# ── URL Validation ────────────────────────────────────────────────────────────

class URLValidationError(ValueError):
    """Raised when a URL fails security or format validation."""


def validate_url(url: str) -> str:
    """
    Validates and normalises a URL string.
    Returns the cleaned URL or raises URLValidationError.
    """
    url = url.strip()

    if not url:
        raise URLValidationError("URL cannot be empty.")

    if len(url) > MAX_URL_LENGTH:
        raise URLValidationError(f"URL exceeds maximum length of {MAX_URL_LENGTH} characters.")

    try:
        parsed = urlparse(url)
    except Exception as e:
        raise URLValidationError(f"Malformed URL: {e}")

    if parsed.scheme not in ("http", "https"):
        raise URLValidationError(
            f"Only http:// and https:// URLs are supported. Got: '{parsed.scheme}://'."
        )

    if not parsed.netloc:
        raise URLValidationError("URL has no domain component.")

    host = (parsed.hostname or "").lower()

    if host in _BLOCKED_HOSTS:
        raise URLValidationError("Access to local/private addresses is not permitted.")

    if any(host.startswith(prefix) for prefix in _BLOCKED_PREFIXES):
        raise URLValidationError("Access to private network ranges is not permitted.")

    return url


# ── HTTP Fetching ─────────────────────────────────────────────────────────────

class FetchError(IOError):
    """Raised when HTTP fetch fails after retries."""


def _fetch_html(url: str, retries: int = 2) -> tuple[str, int]:
    """
    Fetches the raw HTML of a URL.
    Returns (html_text, elapsed_ms).
    Raises FetchError on failure.
    """
    last_err: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            t0 = time.monotonic()
            session = requests.Session()
            session.max_redirects = MAX_REDIRECTS
            resp = session.get(
                url,
                headers=_REQUEST_HEADERS,
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
                stream=False,
            )
            elapsed_ms = int((time.monotonic() - t0) * 1000)

            # Raise for 4xx/5xx
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                raise FetchError(
                    f"Expected HTML content but received: {content_type}. "
                    "Only web pages are supported; for PDFs use file upload."
                )

            # Respect encoding
            resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text, elapsed_ms

        except requests.exceptions.SSLError as e:
            raise FetchError(f"SSL certificate error: {e}") from e
        except requests.exceptions.ConnectionError as e:
            last_err = FetchError(f"Connection failed: {e}")
        except requests.exceptions.Timeout:
            last_err = FetchError(
                f"Request timed out after {REQUEST_TIMEOUT}s. "
                "The site may be slow or unavailable."
            )
        except requests.exceptions.TooManyRedirects:
            raise FetchError("Too many redirects — the URL may be a redirect loop.")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            if status == 403:
                raise FetchError(
                    "Access denied (403 Forbidden). "
                    "This site blocks automated access."
                )
            if status == 404:
                raise FetchError("Page not found (404).")
            if status == 429:
                raise FetchError("Rate limited (429). Please try again in a few minutes.")
            raise FetchError(f"HTTP error {status}: {e}") from e
        except FetchError:
            raise
        except Exception as e:
            last_err = FetchError(f"Unexpected fetch error: {e}")

        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))

    raise last_err or FetchError("Failed to fetch URL after retries.")


# ── Text Extraction ───────────────────────────────────────────────────────────

def _extract_via_trafilatura(html: str, url: str) -> Optional[str]:
    """
    Primary extraction using Trafilatura.
    Returns clean text or None if extraction fails/insufficient.
    """
    try:
        result = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            include_images=False,
            include_links=False,
            no_fallback=False,
            favor_recall=True,       # prioritise completeness over precision
            deduplicate=True,
        )
        if result and len(result.split()) >= MIN_WORD_COUNT:
            return result
        return None
    except Exception as e:
        logger.warning(f"[scraper] trafilatura extraction failed: {e}")
        return None


def _extract_via_beautifulsoup(html: str) -> Optional[str]:
    """
    Fallback extraction using BeautifulSoup.
    Removes known noise tags/classes, then extracts from article/main/body.
    Returns clean text or None if insufficient.
    """
    try:
        soup = BeautifulSoup(html, "lxml")

        # 1. Remove structural noise tags entirely
        for tag in soup(["nav", "header", "footer", "script", "style",
                          "aside", "iframe", "noscript", "form",
                          "button", "figure", "figcaption"]):
            tag.decompose()

        # 2. Remove elements by noisy class/id patterns
        for element in soup.find_all(True):
            attrs = " ".join([
                " ".join(element.get("class", [])),
                element.get("id", ""),
            ]).lower()
            if any(pattern in attrs for pattern in _NOISE_PATTERNS):
                element.decompose()

        # 3. Prefer semantic content containers
        content_tag = (
            soup.find("article")
            or soup.find("main")
            or soup.find(id=re.compile(r"content|main|article", re.I))
            or soup.find(class_=re.compile(r"content|main|article|post", re.I))
            or soup.find("body")
        )

        if not content_tag:
            return None

        # 4. Extract text, preserving paragraph breaks
        paragraphs = []
        for elem in content_tag.find_all(
            ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th", "pre"]
        ):
            text = elem.get_text(separator=" ", strip=True)
            if text and len(text.split()) >= 3:   # skip very short fragments
                paragraphs.append(text)

        result = "\n".join(paragraphs)

        if len(result.split()) >= MIN_WORD_COUNT:
            return result
        return None

    except Exception as e:
        logger.warning(f"[scraper] BeautifulSoup extraction failed: {e}")
        return None


# ── Metadata Extraction ───────────────────────────────────────────────────────

def _extract_metadata(html: str, url: str, extracted_text: str, fetch_time_ms: int) -> PageMetadata:
    """Extracts title, description, domain from raw HTML."""
    try:
        soup = BeautifulSoup(html, "lxml")

        # Title: prefer og:title > <title> > domain
        title = (
            (soup.find("meta", property="og:title") or {}).get("content")
            or (soup.find("meta", attrs={"name": "twitter:title"}) or {}).get("content")
            or (soup.title and soup.title.get_text(strip=True))
            or urlparse(url).netloc
            or "Untitled Page"
        )

        # Description
        description = (
            (soup.find("meta", attrs={"name": "description"}) or {}).get("content")
            or (soup.find("meta", property="og:description") or {}).get("content")
            or ""
        )

        domain = urlparse(url).netloc.replace("www.", "")
        word_count = len(extracted_text.split())

        return PageMetadata(
            title=title.strip()[:255],
            description=description.strip()[:500],
            domain=domain,
            source_url=url,
            word_count=word_count,
            fetch_time_ms=fetch_time_ms,
        )
    except Exception as e:
        logger.warning(f"[scraper] metadata extraction error: {e}")
        return PageMetadata(source_url=url, domain=urlparse(url).netloc)


# ── Text Post-Processing ──────────────────────────────────────────────────────

def _post_process(text: str) -> str:
    """
    Cleans extracted text:
    - Normalises unicode whitespace
    - Collapses repeated blank lines
    - Trims to MAX_WORD_COUNT
    """
    # Normalise line endings and unicode whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)       # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)       # max 2 consecutive blank lines
    text = text.strip()

    # Hard cap
    words = text.split()
    if len(words) > MAX_WORD_COUNT:
        text = " ".join(words[:MAX_WORD_COUNT])
        text += f"\n\n[Content truncated at {MAX_WORD_COUNT} words]"

    return text


def _slugify(text: str, max_len: int = 80) -> str:
    """Creates a filesystem-safe slug from page title."""
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return (slug[:max_len] or "webpage")


# ── Public API ────────────────────────────────────────────────────────────────

def scrape_url(url: str) -> ScrapedContent:
    """
    Main entry point. Fetches and extracts clean study text from a URL.

    Pipeline:
        validate_url → _fetch_html → trafilatura → BS4 fallback → post_process

    Returns a ScrapedContent with success=True on success,
    or success=False + error message on any failure.
    """
    # Step 1: Validate
    try:
        url = validate_url(url)
    except URLValidationError as e:
        return ScrapedContent(
            raw_text="", processed_text="", filename="invalid",
            source_url=url, success=False, error=str(e),
        )

    logger.info(f"[scraper] fetching: {url}")

    # Step 2: Fetch
    try:
        html, fetch_time_ms = _fetch_html(url)
    except FetchError as e:
        logger.error(f"[scraper] fetch failed: {e}")
        return ScrapedContent(
            raw_text="", processed_text="", filename="fetch-failed",
            source_url=url, success=False, error=str(e),
        )

    logger.info(f"[scraper] fetched {len(html)} bytes in {fetch_time_ms}ms")

    # Step 3: Extract text (Trafilatura → BS4 fallback)
    raw_text = _extract_via_trafilatura(html, url)
    extraction_method = "trafilatura"

    if not raw_text:
        logger.info("[scraper] trafilatura insufficient, trying BeautifulSoup fallback")
        raw_text = _extract_via_beautifulsoup(html)
        extraction_method = "beautifulsoup"

    if not raw_text:
        return ScrapedContent(
            raw_text="", processed_text="", filename="no-content",
            source_url=url, success=False,
            error=(
                "Could not extract readable text from this page. "
                "The page may be JavaScript-rendered, behind a login, "
                "or contain only images/video."
            ),
        )

    logger.info(f"[scraper] extracted {len(raw_text.split())} words via {extraction_method}")

    # Step 4: Post-process
    processed_text = _post_process(raw_text)

    # Step 5: Metadata
    metadata = _extract_metadata(html, url, processed_text, fetch_time_ms)

    # Step 6: Build filename (slug of page title, used in uploads.filename)
    filename = _slugify(metadata.title) + ".url"

    logger.info(
        f"[scraper] done — title='{metadata.title}' "
        f"words={metadata.word_count} domain={metadata.domain}"
    )

    return ScrapedContent(
        raw_text=raw_text,
        processed_text=processed_text,
        filename=filename,
        source_url=url,
        metadata=metadata,
        success=True,
        error=None,
    )
