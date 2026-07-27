"""
voice_routes.py  —  Flask Blueprint: Feature 3 (Voice Assistant)

Routes registered (all under /api/voice):
  POST  /api/voice/transcribe              — audio blob → text (STT only)
  POST  /api/voice/speak                   — text / agent JSON → MP3 (TTS only)
  POST  /api/voice/command/<upload_id>     — FULL pipeline: audio → STT → agent → TTS
  GET   /api/voice/status                  — health check + model info

The /api/voice/command/<upload_id> route is the primary voice interaction:
  1. Receives audio blob from browser microphone
  2. Transcribes with Whisper (STT)
  3. Passes transcript to LangGraph agent (Feature 1) — reuses existing logic
  4. Converts agent response to speech (TTS)
  5. Returns: transcript + agent JSON result + base64 MP3 audio

Fallback (no Feature 1): if agent_routes is not registered,
the route falls back to the existing /api/ask endpoint for Q&A.

Integration steps (add to app.py):
────────────────────────────────────
    from voice_routes import voice_bp
    app.register_blueprint(voice_bp)

That's it. Works standalone or alongside Feature 1 & 2.
"""

from __future__ import annotations
import logging
from typing import Any

from flask import Blueprint, request, jsonify, current_app

from modules.stt_service import transcribe_audio, get_model_info
from modules.tts_service import synthesise_speech, synthesise_agent_response, smart_prepare_for_speech

logger = logging.getLogger(__name__)

voice_bp = Blueprint("voice", __name__, url_prefix="/api/voice")

# Max upload size for audio blobs — 25 MB (2 min at webm/opus is ~3-4 MB, so very safe)
_MAX_AUDIO_BYTES = 25 * 1024 * 1024


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_audio_from_request() -> tuple[bytes, str]:
    """
    Extracts audio bytes and content type from the current Flask request.
    Accepts both multipart form-data (file field 'audio') and raw binary body.
    Raises ValueError with a user-friendly message on bad input.
    """
    # Prefer multipart file upload (most reliable across browsers)
    if "audio" in request.files:
        f = request.files["audio"]
        data = f.read()
        ct = f.content_type or request.content_type or ""
        return data, ct

    # Fall back to raw binary body (for fetch() with binary body)
    data = request.get_data()
    ct = request.content_type or ""

    if not data:
        raise ValueError(
            "No audio data received. Send as multipart form field 'audio' "
            "or as a raw binary body."
        )

    if len(data) > _MAX_AUDIO_BYTES:
        raise ValueError(
            f"Audio file too large ({len(data) // 1024} KB). "
            f"Maximum allowed: {_MAX_AUDIO_BYTES // 1024 // 1024} MB."
        )

    return data, ct


def _run_agent(upload_id: int, text: str) -> dict:
    """
    Calls the LangGraph agent (Feature 1) if available,
    otherwise falls back to the existing Q&A route logic.
    Returns the agent response dict.
    """
    # Try Feature 1 agent first
    try:
        from modules.agent import run_agent as _run_agent_fn
        from modules.database import get_upload

        graph = current_app.agent_graph  # set by Feature 1's init_agent()
        upload = get_upload(upload_id)
        if not upload or not upload.get("processed_text"):
            return {"success": False, "error": "Document not found or has no text."}

        return _run_agent_fn(graph, upload_id, upload["processed_text"], text)

    except (AttributeError, ImportError):
        pass  # Feature 1 not installed — use fallback

    # Fallback: existing Q&A system
    try:
        from modules.database import get_upload
        from modules.qa_system import answer_question
        from modules.database import save_qa

        upload = get_upload(upload_id)
        if not upload or not upload.get("processed_text"):
            return {"success": False, "error": "Document not found or has no text."}

        answer = answer_question(upload["processed_text"], text)
        save_qa(upload_id, text, answer)
        return {"success": True, "intent": "answer_question", "answer": answer}

    except Exception as e:
        logger.error(f"[voice] agent fallback failed: {e}")
        return {"success": False, "error": str(e)}


# ── Routes ────────────────────────────────────────────────────────────────────

@voice_bp.route("/transcribe", methods=["POST"])
def transcribe_route():
    """
    POST /api/voice/transcribe

    Accepts audio blob, returns transcribed text.
    Use this if you want STT only (e.g., to fill a text field).

    Request : multipart/form-data with field 'audio'
              OR raw binary body with Content-Type: audio/webm (or similar)

    Response:
    {
      "success": true,
      "text": "Generate 5 MCQs about inheritance",
      "duration_ms": 2300,
      "model": "tiny.en"
    }
    """
    try:
        audio_bytes, content_type = _get_audio_from_request()
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    result = transcribe_audio(audio_bytes, content_type)

    if not result.success:
        return jsonify({"success": False, "error": result.error}), 422

    return jsonify({
        "success": True,
        "text": result.text,
        "duration_ms": result.duration_ms,
        "model": result.model_used,
    })


@voice_bp.route("/speak", methods=["POST"])
def speak_route():
    """
    POST /api/voice/speak

    Converts text or an agent response JSON to spoken MP3 audio.

    Request body (JSON), one of:
      {"text": "Hello, this is a test"}
      {"response": { ...agent response dict... }}   ← auto-formats MCQs, flashcards etc.

    Response:
    {
      "success": true,
      "audio_base64": "<base64 MP3>",
      "mime_type": "audio/mpeg",
      "word_count": 42,
      "truncated": false
    }
    """
    data = request.get_json(silent=True) or {}

    if "text" in data:
        text = str(data["text"]).strip()
        if not text:
            return jsonify({"success": False, "error": "Text cannot be empty."}), 400
        result = synthesise_speech(text)

    elif "response" in data:
        agent_resp = data["response"]
        if not isinstance(agent_resp, dict):
            return jsonify({"success": False, "error": "'response' must be a JSON object."}), 400
        result = synthesise_agent_response(agent_resp)

    else:
        return jsonify({
            "success": False,
            "error": "Provide either 'text' (string) or 'response' (agent JSON object)."
        }), 400

    if not result.success:
        return jsonify({"success": False, "error": result.error}), 500

    return jsonify({
        "success": True,
        "audio_base64": result.audio_base64,
        "mime_type": result.mime_type,
        "word_count": result.word_count,
        "truncated": result.truncated,
    })


@voice_bp.route("/command/<int:upload_id>", methods=["POST"])
def voice_command_route(upload_id: int):
    """
    POST /api/voice/command/<upload_id>

    PRIMARY ENDPOINT — Complete voice interaction pipeline:
      audio → STT → LangGraph agent → TTS → response

    Request : multipart/form-data with field 'audio'
    Optional form fields:
      speak=true/false   — whether to include TTS audio in response (default: true)

    Response:
    {
      "success": true,
      "transcript": "Generate 5 MCQs about polymorphism",
      "agent_response": { ...full agent result... },
      "audio_base64": "<base64 MP3>",
      "mime_type": "audio/mpeg",
      "truncated": false,
      "pipeline": {
        "stt_duration_ms": 1800,
        "whisper_model": "tiny.en"
      }
    }
    """
    # ── Step 1: Extract audio ────────────────────────────────────────────────
    try:
        audio_bytes, content_type = _get_audio_from_request()
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    # ── Step 2: STT — audio → text ────────────────────────────────────────────
    stt_result = transcribe_audio(audio_bytes, content_type)

    if not stt_result.success:
        return jsonify({
            "success": False,
            "stage": "transcription",
            "error": stt_result.error,
        }), 422

    transcript = stt_result.text
    logger.info(f"[voice] upload_id={upload_id} transcript='{transcript}'")

    # ── Step 3: Agent — text → structured response ────────────────────────────
    agent_response = _run_agent(upload_id, transcript)

    if not agent_response.get("success"):
        # Still return the transcript so the UI can show what was heard
        return jsonify({
            "success": False,
            "stage": "agent",
            "transcript": transcript,
            "error": agent_response.get("error", "Agent processing failed."),
        }), 500

    # ── Step 4: TTS — agent response → spoken audio ───────────────────────────
    speak = str(request.form.get("speak", "true")).lower() not in ("false", "0", "no")
    tts_result = None
    audio_b64 = ""
    truncated = False

    if speak:
        tts_result = synthesise_agent_response(agent_response)
        if tts_result.success:
            audio_b64 = tts_result.audio_base64
            truncated = tts_result.truncated
        else:
            logger.warning(f"[voice] TTS failed (non-fatal): {tts_result.error}")
            # TTS failure is non-fatal — still return transcript + agent result

    return jsonify({
        "success": True,
        "transcript": transcript,
        "agent_response": agent_response,
        "audio_base64": audio_b64,
        "mime_type": "audio/mpeg",
        "truncated": truncated,
        "pipeline": {
            "stt_duration_ms": stt_result.duration_ms,
            "whisper_model": stt_result.model_used,
            "tts_word_count": tts_result.word_count if tts_result else 0,
            "tts_success": tts_result.success if tts_result else False,
        },
    })


@voice_bp.route("/status", methods=["GET"])
def voice_status():
    """
    GET /api/voice/status

    Health check — verifies STT and TTS services are available.
    Also reports which Whisper model is configured.
    Does NOT download/load the Whisper model; just returns config.
    """
    model_info = get_model_info()

    # Check pyttsx3 availability
    tts_available = False
    tts_error = None
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty("voices") or []
        tts_available = len(voices) > 0
    except Exception as e:
        tts_error = str(e)

    # Check whisper availability
    stt_available = False
    try:
        import whisper
        stt_available = True
    except ImportError:
        pass

    return jsonify({
        "success": True,
        "stt": {
            "available": stt_available,
            "backend": "openai-whisper",
            "model": model_info["model"],
            "model_loaded": model_info["loaded"],
            "max_duration_seconds": model_info["max_duration_seconds"],
        },
        "tts": {
            "available": tts_available,
            "backend": "pyttsx3 + espeak-ng",
            "error": tts_error,
        },
        "agent_integrated": hasattr(current_app, "agent_graph"),
    })
