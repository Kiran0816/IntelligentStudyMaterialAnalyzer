"""
modules/tts_service.py  —  Text-to-Speech via pyttsx3 + espeak-ng

Architecture
────────────
Input  : any text (plain, Markdown, JSON summary, MCQ list, etc.)
Process: smart_prepare_for_speech() → pyttsx3 → WAV → pydub MP3 compress
Output : SpeechResult dataclass with base64-encoded MP3 audio

Design decisions
────────────────
* pyttsx3 engine is re-created per request (not a singleton) because
  pyttsx3's espeak driver is not reliably thread-safe for concurrent writes.
  The overhead is small (<20 ms) compared to synthesis time.
* Output format: MP3 at 64 kbps — 5.4x smaller than WAV, plays natively
  in all browsers via <audio src="data:audio/mpeg;base64,...">
* smart_prepare_for_speech() converts structured agent output (MCQs, 
  flashcards, bullet lists) into natural spoken English before synthesis.
* Hard cap: 1200 words. Longer responses are truncated with a spoken notice.
  This prevents synthesis of multi-thousand-word summaries nobody wants read aloud.
* WHISPER_VOICE env var: set to any espeak voice id for a different accent.
  Default: en-us (American English).
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# Override with: TTS_VOICE=gmw/en  (British) or  TTS_VOICE=gmw/en-gb-scotland etc.
_PREFERRED_VOICE_HINT: str = os.environ.get("TTS_VOICE", "en-us")
_SPEECH_RATE:   int = int(os.environ.get("TTS_RATE",   "155"))   # wpm (default pyttsx3=200)
_SPEECH_VOLUME: float = float(os.environ.get("TTS_VOLUME", "1.0"))
_MP3_BITRATE: str = "64k"

_MAX_WORDS = 1200    # spoken word cap before truncation


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class SpeechResult:
    audio_base64: str         # base64-encoded MP3 (empty on failure)
    mime_type: str = "audio/mpeg"
    success: bool = True
    error: Optional[str] = None
    word_count: int = 0
    truncated: bool = False


# ── Text preparation ──────────────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """Converts Markdown to plain speakable text."""
    # Remove fenced code blocks entirely (code is not speakable)
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)    # inline code → bare text

    # Convert headers to natural phrasing
    text = re.sub(r"^#{1,6}\s+(.+)$", r"\1.", text, flags=re.MULTILINE)

    # Remove bold/italic markers, keep content
    text = re.sub(r"\*{1,2}([^*\n]+)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}([^_\n]+)_{1,2}", r"\1", text)

    # Convert bullet/numbered lists to comma-separated phrases
    text = re.sub(r"^[\*\-\+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+[\.\)]\s+", "", text, flags=re.MULTILINE)

    # Remove horizontal rules
    text = re.sub(r"^[\-\*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Remove markdown links but keep link text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # Collapse excessive whitespace and blank lines
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def _format_mcqs_for_speech(mcqs: list[dict]) -> str:
    """Converts MCQ list to natural spoken format."""
    parts = []
    for i, mcq in enumerate(mcqs[:5], 1):      # read max 5 MCQs aloud
        q = mcq.get("question", "")
        opts = mcq.get("options", [])
        ans = mcq.get("correct_answer", "")

        parts.append(f"Question {i}. {q}")
        for j, opt in enumerate(opts):
            label = chr(65 + j)                 # A, B, C, D
            parts.append(f"Option {label}: {opt}")
        parts.append(f"The correct answer is: {ans}.")
        parts.append("")                        # brief pause between questions

    return " ".join(parts)


def _format_flashcards_for_speech(cards: list[dict]) -> str:
    """Reads flashcards as Front / Back pairs."""
    parts = []
    for i, card in enumerate(cards[:10], 1):   # read max 10 cards aloud
        front = card.get("front", "")
        back  = card.get("back", "")
        parts.append(f"Card {i}. Front: {front}. Back: {back}.")
    return " ".join(parts)


def _format_keywords_for_speech(keywords: list[str]) -> str:
    """Reads keywords as a natural list."""
    if not keywords:
        return "No keywords were extracted."
    joined = ", ".join(keywords[:-1]) + f", and {keywords[-1]}" if len(keywords) > 1 else keywords[0]
    return f"The key concepts are: {joined}."


def _format_analytics_for_speech(analytics: dict) -> str:
    """Converts analytics dict to a natural spoken sentence."""
    level   = analytics.get("difficulty_level", "unknown")
    words   = analytics.get("word_count", 0)
    time    = analytics.get("estimated_study_time", 0)
    sents   = analytics.get("sentence_count", 0)
    return (
        f"This document has a {level} difficulty level. "
        f"It contains {words} words across {sents} sentences, "
        f"with an estimated study time of {time} minutes."
    )


def smart_prepare_for_speech(agent_response: dict) -> str:
    """
    Converts any agent response dict into speakable plain text.

    Handles: summary, mcqs, keywords, analytics, flashcards,
             revision_notes, qa_answer, and plain string.
    """
    if isinstance(agent_response, str):
        return _strip_markdown(agent_response)

    parts: list[str] = []

    # Q&A answer — most common voice interaction
    if "answer" in agent_response:
        parts.append(_strip_markdown(str(agent_response["answer"])))

    # Summary
    elif "summary" in agent_response and agent_response["summary"]:
        parts.append(_strip_markdown(str(agent_response["summary"])))

    # MCQs
    elif "mcqs" in agent_response and agent_response["mcqs"]:
        n = len(agent_response["mcqs"])
        parts.append(f"I generated {n} multiple choice question{'s' if n != 1 else ''}.")
        parts.append(_format_mcqs_for_speech(agent_response["mcqs"]))

    # Flashcards
    elif "flashcards" in agent_response and agent_response["flashcards"]:
        n = len(agent_response["flashcards"])
        parts.append(f"Here are {n} flashcard{'s' if n != 1 else ''} for you.")
        parts.append(_format_flashcards_for_speech(agent_response["flashcards"]))

    # Revision notes
    elif "revision_notes" in agent_response and agent_response["revision_notes"]:
        parts.append("Here are your revision notes.")
        parts.append(_strip_markdown(str(agent_response["revision_notes"])))

    # Keywords
    elif "keywords" in agent_response and agent_response["keywords"]:
        parts.append(_format_keywords_for_speech(agent_response["keywords"]))

    # Analytics
    elif "analytics" in agent_response and agent_response["analytics"]:
        parts.append(_format_analytics_for_speech(agent_response["analytics"]))

    # Fallback: any string value in the response
    else:
        for v in agent_response.values():
            if isinstance(v, str) and len(v) > 10:
                parts.append(_strip_markdown(v))
                break

    if not parts:
        return "I'm sorry, I could not generate a spoken response for this result."

    return " ".join(parts)


def _apply_word_cap(text: str) -> tuple[str, bool]:
    """Truncates text to _MAX_WORDS and appends a spoken notice."""
    words = text.split()
    if len(words) <= _MAX_WORDS:
        return text, False
    truncated = " ".join(words[:_MAX_WORDS])
    truncated += (
        f" ... The full response contains {len(words)} words. "
        "I have read the first part aloud. You can read the rest on screen."
    )
    return truncated, True


# ── TTS engine ────────────────────────────────────────────────────────────────

def _select_voice(engine) -> None:
    """Picks the best available English voice matching _PREFERRED_VOICE_HINT."""
    voices = engine.getProperty("voices") or []
    hint = _PREFERRED_VOICE_HINT.lower()

    # Exact match first
    for v in voices:
        if hint in str(v.id).lower():
            engine.setProperty("voice", v.id)
            logger.debug(f"[tts] voice selected (exact): {v.id}")
            return

    # Any English voice
    for v in voices:
        if "/en" in str(v.id).lower():
            engine.setProperty("voice", v.id)
            logger.debug(f"[tts] voice selected (fallback en): {v.id}")
            return

    logger.warning("[tts] no English voice found, using system default")


def _synthesise_to_audio(text: str) -> tuple[bytes, str]:
    """
    Runs pyttsx3 synthesis to WAV, then tries to compress to MP3.
    Falls back to raw WAV bytes if pydub/ffmpeg is not available.
    Returns (audio_bytes, mime_type).
    """
    import pyttsx3

    engine = pyttsx3.init()
    _select_voice(engine)
    engine.setProperty("rate",   _SPEECH_RATE)
    engine.setProperty("volume", _SPEECH_VOLUME)

    wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_tmp.close()

    try:
        engine.save_to_file(text, wav_tmp.name)
        engine.runAndWait()

        wav_size = os.path.getsize(wav_tmp.name)
        if wav_size < 100:
            raise RuntimeError("pyttsx3 produced an empty WAV file.")

        # Try to compress WAV → MP3
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_wav(wav_tmp.name)
            mp3_buf = io.BytesIO()
            audio.export(mp3_buf, format="mp3", bitrate=_MP3_BITRATE)
            return mp3_buf.getvalue(), "audio/mpeg"
        except Exception as e:
            logger.warning(f"Could not convert WAV to MP3 (likely missing ffmpeg): {e}. Returning WAV directly.")
            with open(wav_tmp.name, "rb") as f:
                return f.read(), "audio/wav"

    finally:
        if os.path.exists(wav_tmp.name):
            os.unlink(wav_tmp.name)


# ── Public API ────────────────────────────────────────────────────────────────

def synthesise_speech(text: str) -> SpeechResult:
    """
    Converts plain text to an audio clip (MP3 if ffmpeg is present, otherwise WAV).

    Parameters
    ----------
    text : plain text to speak (Markdown will be stripped internally)

    Returns
    -------
    SpeechResult with .audio_base64 (base64 MP3/WAV) on success.
    """
    if not text or not text.strip():
        return SpeechResult(
            audio_base64="",
            success=False,
            error="No text provided for speech synthesis.",
        )

    text = _strip_markdown(text)
    text, truncated = _apply_word_cap(text)
    word_count = len(text.split())

    try:
        audio_bytes, mime_type = _synthesise_to_audio(text)
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        logger.info(f"[tts] synthesised {word_count} words → {len(audio_bytes):,} bytes {mime_type}")
        return SpeechResult(
            audio_base64=audio_b64,
            mime_type=mime_type,
            success=True,
            word_count=word_count,
            truncated=truncated,
        )

    except Exception as e:
        logger.error(f"[tts] synthesis failed: {e}", exc_info=True)
        return SpeechResult(
            audio_base64="",
            success=False,
            error=f"Speech synthesis failed: {e}",
        )


def synthesise_agent_response(agent_response: dict) -> SpeechResult:
    """
    Convenience wrapper: takes a raw agent JSON response dict,
    converts it to speakable text, then synthesises it.

    This is the function called by the Flask voice route after getting
    the agent's answer so callers never have to format agent output manually.
    """
    try:
        speakable = smart_prepare_for_speech(agent_response)
        return synthesise_speech(speakable)
    except Exception as e:
        logger.error(f"[tts] agent response prep failed: {e}")
        return SpeechResult(audio_base64="", success=False, error=str(e))
