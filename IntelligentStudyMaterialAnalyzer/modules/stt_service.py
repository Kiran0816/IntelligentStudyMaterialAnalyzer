"""
modules/stt_service.py  —  Speech-to-Text via OpenAI Whisper

Architecture
────────────
Input  : raw audio bytes from browser (webm / ogg / wav / mp4)
Process: pydub normalises to 16kHz mono WAV → Whisper transcribes
Output : TranscriptionResult dataclass

Design decisions
────────────────
* Whisper model is loaded ONCE at module level on first use (lazy singleton)
  so Flask workers don't reload a 150 MB model on every request.
* "tiny.en" is the default — 39 MB, ~32x real-time on CPU, good English accuracy.
  Switch to "base.en" (74 MB) or "small.en" (244 MB) by setting
  WHISPER_MODEL env var before starting Flask.
* pydub + ffmpeg handles every format the browser MediaRecorder can produce
  (audio/webm;codecs=opus is the most common).
* Minimum audio length guard: < 0.3 s is almost certainly noise/accidental click.
* No_speech_threshold: Whisper's own hallucination guard is kept at 0.6
  (slightly stricter than default 0.45) to reduce phantom text on silence.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# Override with WHISPER_MODEL=base.en  or  small.en  for higher accuracy
_WHISPER_MODEL_NAME: str = os.environ.get("WHISPER_MODEL", "tiny.en")

# Audio constraints
_MIN_DURATION_MS  = 300    # ignore recordings shorter than 300 ms
_MAX_DURATION_MS  = 120_000  # 2-minute hard cap
_TARGET_SR        = 16_000  # Whisper's native sample rate
_TARGET_CHANNELS  = 1       # mono

# Whisper decode options tuned for command-style voice input
_WHISPER_OPTIONS: dict = {
    "language": "en",
    "task": "transcribe",
    "no_speech_threshold": 0.6,   # stricter than default 0.45
    "condition_on_previous_text": False,
    "fp16": False,                 # CPU inference
}

# ── Lazy model singleton ───────────────────────────────────────────────────────

_model       = None
_model_lock  = threading.Lock()


def _get_model():
    """
    Loads the Whisper model once and caches it.
    Thread-safe via a lock — concurrent requests will wait for the first load.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                import whisper
                logger.info(f"[stt] loading Whisper model '{_WHISPER_MODEL_NAME}'...")
                _model = whisper.load_model(_WHISPER_MODEL_NAME)
                logger.info(f"[stt] Whisper '{_WHISPER_MODEL_NAME}' loaded")
    return _model


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class TranscriptionResult:
    text: str                    # Cleaned transcript (empty string on failure)
    success: bool
    error: Optional[str] = None
    duration_ms: int = 0         # Audio duration actually processed
    model_used: str = _WHISPER_MODEL_NAME


# ── Audio normalisation ───────────────────────────────────────────────────────

def _normalise_audio(audio_bytes: bytes, content_type: str = "") -> tuple[str, int]:
    """
    Converts raw audio bytes to a 16kHz mono WAV temp file.

    Returns (temp_file_path, duration_ms).
    Raises ValueError for invalid / too-short / too-long audio.
    Caller must delete the temp file after use.
    """
    from pydub import AudioSegment

    if not audio_bytes or len(audio_bytes) < 100:
        raise ValueError("Audio data is too small to process.")

    # Detect format from content_type or let pydub/ffmpeg sniff it
    fmt = None
    ct = (content_type or "").lower()
    if "webm" in ct:
        fmt = "webm"
    elif "ogg" in ct:
        fmt = "ogg"
    elif "mp4" in ct or "m4a" in ct:
        fmt = "mp4"
    elif "wav" in ct:
        fmt = "wav"
    # fmt=None → pydub uses ffmpeg to auto-detect

    try:
        if fmt:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
        else:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    except Exception as e:
        raise ValueError(
            f"Could not decode audio. Ensure the browser is sending a supported format "
            f"(webm, ogg, wav). Detail: {e}"
        )

    duration_ms = len(audio)

    if duration_ms < _MIN_DURATION_MS:
        raise ValueError(
            f"Recording is too short ({duration_ms} ms). "
            "Please hold the microphone button while speaking."
        )

    if duration_ms > _MAX_DURATION_MS:
        logger.warning(f"[stt] audio {duration_ms}ms exceeds cap, trimming to {_MAX_DURATION_MS}ms")
        audio = audio[:_MAX_DURATION_MS]
        duration_ms = _MAX_DURATION_MS

    # Normalise to Whisper's expected format
    audio = audio.set_frame_rate(_TARGET_SR).set_channels(_TARGET_CHANNELS)

    # Write to named temp file (Whisper needs a path, not a buffer)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        audio.export(tmp.name, format="wav")
        tmp.close()
    except Exception as e:
        tmp.close()
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        raise ValueError(f"Audio conversion failed: {e}")

    return tmp.name, duration_ms


def _load_wav_bytes_natively(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    import wave
    import io
    import numpy as np
    
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
        params = wav.getparams()
        n_channels = params.nchannels
        sampwidth = params.sampwidth
        framerate = params.framerate
        n_frames = params.nframes
        
        raw_frames = wav.readframes(n_frames)
        duration_ms = int((n_frames / framerate) * 1000)
        
        if sampwidth == 2:
            data = np.frombuffer(raw_frames, dtype=np.int16)
        elif sampwidth == 1:
            data = (np.frombuffer(raw_frames, dtype=np.uint8).astype(np.int32) - 128) * 256
            data = data.astype(np.int16)
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")
            
        data = data.astype(np.float32) / 32768.0
        
        if n_channels > 1:
            data = data.reshape(-1, n_channels).mean(axis=1)
            
        if framerate != 16000:
            import scipy.interpolate
            duration = len(data) / framerate
            old_x = np.linspace(0, duration, len(data))
            new_len = int(duration * 16000)
            new_x = np.linspace(0, duration, new_len)
            f = scipy.interpolate.interp1d(old_x, data, bounds_error=False, fill_value="extrapolate")
            data = f(new_x)
            
        return data, duration_ms


# ── Public API ────────────────────────────────────────────────────────────────

def transcribe_audio(audio_bytes: bytes, content_type: str = "") -> TranscriptionResult:
    """
    Main entry point: converts raw audio bytes to text via Whisper.

    Parameters
    ----------
    audio_bytes   : raw bytes from the HTTP request (browser MediaRecorder output)
    content_type  : MIME type hint, e.g. 'audio/webm' (optional but helps pydub)

    Returns
    -------
    TranscriptionResult with success=True and .text on success,
    or success=False and .error on any failure.
    """
    tmp_path: Optional[str] = None
    audio_data = None
    duration_ms = 0

    try:
        # Check if the input is a WAV file
        is_wav = audio_bytes.startswith(b"RIFF") and b"WAVE" in audio_bytes[:16]
        
        if is_wav:
            try:
                logger.info("[stt] Inbound file is WAV, decoding natively...")
                audio_data, duration_ms = _load_wav_bytes_natively(audio_bytes)
                logger.info(f"[stt] Decoded WAV: {duration_ms}ms")
            except Exception as wav_err:
                logger.warning(f"[stt] Native WAV decode failed: {wav_err}. Falling back to pydub.")
                
        if audio_data is None:
            # Fall back to pydub normalisation
            tmp_path, duration_ms = _normalise_audio(audio_bytes, content_type)
            logger.info(f"[stt] normalised audio via pydub: {duration_ms}ms → {tmp_path}")
            audio_data = tmp_path

        # Step 2: transcribe
        model = _get_model()
        result = model.transcribe(audio_data, **_WHISPER_OPTIONS)

        text: str = (result.get("text") or "").strip()

        if not text:
            return TranscriptionResult(
                text="",
                success=False,
                error=(
                    "No speech detected. Please speak clearly and closer to the microphone, "
                    "or check that the correct microphone is selected."
                ),
                duration_ms=duration_ms,
            )

        # Remove leading/trailing punctuation noise Whisper sometimes adds
        text = text.strip(".,!? ")
        logger.info(f"[stt] transcribed ({duration_ms}ms): '{text}'")

        return TranscriptionResult(
            text=text,
            success=True,
            duration_ms=duration_ms,
            model_used=_WHISPER_MODEL_NAME,
        )

    except ValueError as e:
        logger.warning(f"[stt] validation error: {e}")
        return TranscriptionResult(text="", success=False, error=str(e))

    except Exception as e:
        logger.error(f"[stt] unexpected error: {e}", exc_info=True)
        return TranscriptionResult(
            text="",
            success=False,
            error=f"Speech recognition failed: {e}",
        )

    finally:
        # Always clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def get_model_info() -> dict:
    """Returns the currently configured Whisper model name (no download triggered)."""
    return {
        "model": _WHISPER_MODEL_NAME,
        "loaded": _model is not None,
        "target_sample_rate": _TARGET_SR,
        "max_duration_seconds": _MAX_DURATION_MS // 1000,
    }
