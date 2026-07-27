/**
 * voice_assistant.js  —  Feature 3: Voice Assistant
 *
 * Drop this file into your existing static/js/ folder.
 * Include it in your index.html AFTER your main app JS.
 *
 * What this does
 * ─────────────────────────────────────────────────────────────
 * 1. Requests microphone permission via MediaRecorder API
 * 2. Records audio while user holds the mic button (push-to-talk)
 * 3. POSTs the audio blob to /api/voice/command/<upload_id>
 * 4. Shows the transcript in the UI instantly
 * 5. Displays the agent response (same as text channel)
 * 6. Auto-plays the spoken MP3 response via Web Audio API
 * 7. Also exposes /api/voice/speak for reading any text aloud
 *    (e.g., "Read summary aloud" button)
 *
 * Browser compatibility
 * ─────────────────────────────────────────────────────────────
 * Requires: MediaRecorder API (Chrome 49+, Firefox 29+, Safari 14.1+)
 * Audio format priority: audio/webm;codecs=opus → audio/ogg;codecs=opus → audio/wav
 *
 * Usage
 * ─────────────────────────────────────────────────────────────
 * VoiceAssistant.init({
 *   uploadId: 42,                          // required: current upload_id
 *   onTranscript: (text) => { ... },       // called when STT returns
 *   onAgentResponse: (data) => { ... },    // called with full agent JSON
 *   onError: (msg) => { ... },             // called on any failure
 *   autoSpeak: true,                       // auto-play TTS response (default: true)
 *   micButtonId: 'voice-mic-btn',          // ID of your mic button element
 *   statusElementId: 'voice-status',       // ID of status text element
 * });
 *
 * VoiceAssistant.speakText("Hello world"); // TTS on demand
 * VoiceAssistant.speakAgentResponse(obj);  // TTS for agent result dict
 */

const VoiceAssistant = (() => {
  // ── State ─────────────────────────────────────────────────────────────────
  let _config          = {};
  let _isRecording     = false;
  let _currentAudio    = null;   // HTMLAudioElement for currently playing TTS
  let _stream          = null;   // MediaStream (held to stop tracks on cleanup)
  let _uploadId        = null;
  let _audioContext    = null;
  let _processor       = null;
  let _inputNode       = null;
  let _pcmBuffer       = [];
  let _recordedSampleRate = 16000;
 
  // ── UI helpers ────────────────────────────────────────────────────────────
  function _setStatus(msg, type = "info") {
    const el = document.getElementById(_config.statusElementId);
    if (!el) return;
    el.textContent = msg;
    el.className = `voice-status voice-status--${type}`;
  }
 
  function _setBtnState(state) {
    const btn = document.getElementById(_config.micButtonId);
    if (!btn) return;
    btn.dataset.voiceState = state;
    const labels = {
      idle:        { text: "🎤 Click to speak",     cls: "voice-btn--idle"        },
      recording:   { text: "⏹ Stop recording",     cls: "voice-btn--recording"   },
      processing:  { text: "⏳ Processing…",        cls: "voice-btn--processing"  },
      speaking:    { text: "🔊 Speaking…",          cls: "voice-btn--speaking"    },
      error:       { text: "🎤 Click to speak",     cls: "voice-btn--idle"        },
    };
    const s = labels[state] || labels.idle;
    btn.textContent = s.text;
    btn.className = `voice-btn ${s.cls}`;
    btn.disabled = (state === "processing");
  }
 
  // ── Audio playback ────────────────────────────────────────────────────────
  function _playBase64Audio(base64, mimeType = "audio/mpeg", onEnd) {
    if (_currentAudio) {
      _currentAudio.pause();
      _currentAudio = null;
    }
    const dataUrl = `data:${mimeType};base64,${base64}`;
    _currentAudio = new Audio(dataUrl);
    _currentAudio.onended = () => {
      _currentAudio = null;
      _setBtnState("idle");
      if (typeof onEnd === "function") onEnd();
    };
    _currentAudio.onerror = (e) => {
      console.error("[VoiceAssistant] audio playback error:", e);
      _setBtnState("idle");
    };
    _setBtnState("speaking");
    _currentAudio.play().catch((err) => {
      // Autoplay may be blocked — inform user
      console.warn("[VoiceAssistant] autoplay blocked:", err);
      _setStatus("Click anywhere to enable audio playback.", "warning");
      _setBtnState("idle");
    });
  }
 
  function stopSpeaking() {
    if (_currentAudio) {
      _currentAudio.pause();
      _currentAudio = null;
    }
    _setBtnState("idle");
  }
 
  // ── Recording ─────────────────────────────────────────────────────────────
  async function _startRecording() {
    if (_isRecording) return;
 
    // Check browser support
    if (!navigator.mediaDevices?.getUserMedia) {
      const msg = "Your browser does not support microphone access. Try Chrome or Firefox.";
      _setStatus(msg, "error");
      if (_config.onError) _config.onError(msg);
      return;
    }
 
    // Stop any playing audio first
    stopSpeaking();
 
    try {
      _stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }
      });
    } catch (err) {
      let msg;
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        msg = "Microphone permission denied. Please allow microphone access and try again.";
      } else if (err.name === "NotFoundError") {
        msg = "No microphone found. Please connect a microphone and try again.";
      } else {
        msg = `Microphone error: ${err.message}`;
      }
      _setStatus(msg, "error");
      if (_config.onError) _config.onError(msg);
      return;
    }
 
    try {
      _audioContext = new (window.AudioContext || window.webkitAudioContext)();
      _recordedSampleRate = _audioContext.sampleRate;
      if (_audioContext.state === "suspended") {
        await _audioContext.resume();
      }
      _inputNode = _audioContext.createMediaStreamSource(_stream);
      _processor = _audioContext.createScriptProcessor(4096, 1, 1);
      
      _pcmBuffer = [];
      _processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        _pcmBuffer.push(new Float32Array(inputData));
      };
 
      _inputNode.connect(_processor);
      _processor.connect(_audioContext.destination);
 
      _isRecording = true;
      _setBtnState("recording");
      _setStatus("Listening… click the button to stop and submit.", "recording");
    } catch (err) {
      const msg = `Audio initialization failed: ${err.message}`;
      _setStatus(msg, "error");
      _setBtnState("error");
      if (_config.onError) _config.onError(msg);
    }
  }
 
  function _stopRecording() {
    if (!_isRecording) return;
    _isRecording = false;
 
    if (_processor) {
      _processor.disconnect();
      _processor = null;
    }
    if (_inputNode) {
      _inputNode.disconnect();
      _inputNode = null;
    }
    if (_stream) {
      _stream.getTracks().forEach((t) => t.stop());
      _stream = null;
    }
    if (_audioContext) {
      _audioContext.close();
      _audioContext = null;
    }
 
    _setBtnState("processing");
    _setStatus("Transcribing your speech…", "info");
 
    _processRecording();
  }
 
  async function _processRecording() {
    if (_pcmBuffer.length === 0) {
      const msg = "No audio captured. Please try again.";
      _setStatus(msg, "error");
      _setBtnState("idle");
      if (_config.onError) _config.onError(msg);
      return;
    }
 
    // Flatten PCM buffer
    let totalLength = 0;
    for (const buf of _pcmBuffer) {
      totalLength += buf.length;
    }
    let pcmData = new Float32Array(totalLength);
    let offset = 0;
    for (const buf of _pcmBuffer) {
      pcmData.set(buf, offset);
      offset += buf.length;
    }
    _pcmBuffer = [];

    // Downsample from the native sample rate to 16000Hz
    const targetSampleRate = 16000;
    if (_recordedSampleRate && _recordedSampleRate !== targetSampleRate) {
      pcmData = _downsampleBuffer(pcmData, _recordedSampleRate, targetSampleRate);
    }

    if (pcmData.length < 3200) { // < 200ms
      const msg = "Recording too short. Please speak longer.";
      _setStatus(msg, "error");
      _setBtnState("idle");
      if (_config.onError) _config.onError(msg);
      return;
    }
 
    // Encode to 16-bit WAV
    const buffer = new ArrayBuffer(44 + pcmData.length * 2);
    const view = new DataView(buffer);
 
    /* RIFF identifier */
    _writeString(view, 0, 'RIFF');
    /* file length */
    view.setUint32(4, 36 + pcmData.length * 2, true);
    /* RIFF type */
    _writeString(view, 8, 'WAVE');
    /* format chunk identifier */
    _writeString(view, 12, 'fmt ');
    /* format chunk length */
    view.setUint32(16, 16, true);
    /* sample format (raw pcm = 1) */
    view.setUint16(20, 1, true);
    /* channel count (mono) */
    view.setUint16(22, 1, true);
    /* sample rate (16000) */
    view.setUint32(24, 16000, true);
    /* byte rate */
    view.setUint32(28, 16000 * 2, true);
    /* block align */
    view.setUint16(32, 2, true);
    /* bits per sample */
    view.setUint16(34, 16, true);
    /* data chunk identifier */
    _writeString(view, 36, 'data');
    /* data chunk length */
    view.setUint32(40, pcmData.length * 2, true);
 
    // Write PCM samples
    let index = 44;
    for (let i = 0; i < pcmData.length; i++) {
      let s = Math.max(-1, Math.min(1, pcmData[i]));
      view.setInt16(index, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      index += 2;
    }
 
    const blob = new Blob([view], { type: 'audio/wav' });
 
    // ── Send to backend ────────────────────────────────────────────────────
    const formData = new FormData();
    formData.append("audio", blob, "recording.wav");
    formData.append("speak", _config.autoSpeak !== false ? "true" : "false");
 
    try {
      const resp = await fetch(`/api/voice/command/${_uploadId}`, {
        method: "POST",
        body: formData,
      });
 
      const data = await resp.json();
 
      if (!resp.ok || !data.success) {
        const msg = data.error || `Server error ${resp.status}`;
        _setStatus(`Error: ${msg}`, "error");
        _setBtnState("error");
        if (_config.onError) _config.onError(msg);
        return;
      }
 
      // ── Show transcript ────────────────────────────────────────────────
      if (data.transcript) {
        _setStatus(`Heard: "${data.transcript}"`, "success");
        if (_config.onTranscript) _config.onTranscript(data.transcript);
      }
 
      // ── Pass agent response to caller ──────────────────────────────────
      if (_config.onAgentResponse && data.agent_response) {
        _config.onAgentResponse(data.agent_response);
      }
 
      // ── Play TTS audio ─────────────────────────────────────────────────
      if (data.audio_base64) {
        _playBase64Audio(data.audio_base64, data.mime_type || "audio/mpeg", () => {
          _setStatus("Ready. Click button to speak again.", "info");
        });
        if (data.truncated) {
          console.info("[VoiceAssistant] response was truncated for TTS");
        }
      } else {
        _setBtnState("idle");
        _setStatus("Ready. Click button to speak.", "info");
      }
 
    } catch (err) {
      const msg = `Network error: ${err.message}`;
      _setStatus(msg, "error");
      _setBtnState("error");
      if (_config.onError) _config.onError(msg);
    }
  }
 
  function _writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }
 
  function _downsampleBuffer(buffer, fromRate, toRate) {
    if (fromRate === toRate) return buffer;
    if (toRate > fromRate) return buffer;
    const compression = fromRate / toRate;
    const length = Math.floor(buffer.length / compression);
    const result = new Float32Array(length);
    for (let i = 0; i < length; i++) {
      result[i] = buffer[Math.round(i * compression)];
    }
    return result;
  }
 
  // ── Button event binding ──────────────────────────────────────────────────
  function _bindButton() {
    const btn = document.getElementById(_config.micButtonId);
    if (!btn) {
      console.warn(`[VoiceAssistant] button #${_config.micButtonId} not found`);
      return;
    }
 
    // Toggle click: click to record, click again to send
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      if (!_isRecording) {
        _startRecording();
      } else {
        _stopRecording();
      }
    });
  }

  // ── Public: speak any text on demand ──────────────────────────────────────
  async function speakText(text) {
    if (!text) return;
    try {
      const resp  = await fetch("/api/voice/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await resp.json();
      if (data.success && data.audio_base64) {
        _playBase64Audio(data.audio_base64, data.mime_type || "audio/mpeg");
      } else {
        console.warn("[VoiceAssistant] speakText failed:", data.error);
      }
    } catch (err) {
      console.error("[VoiceAssistant] speakText error:", err);
    }
  }

  // ── Public: speak an agent response dict ──────────────────────────────────
  async function speakAgentResponse(agentResponseObj) {
    if (!agentResponseObj) return;
    try {
      const resp = await fetch("/api/voice/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ response: agentResponseObj }),
      });
      const data = await resp.json();
      if (data.success && data.audio_base64) {
        _playBase64Audio(data.audio_base64, data.mime_type || "audio/mpeg");
      } else {
        console.warn("[VoiceAssistant] speakAgentResponse failed:", data.error);
      }
    } catch (err) {
      console.error("[VoiceAssistant] speakAgentResponse error:", err);
    }
  }

  // ── Public: initialise ────────────────────────────────────────────────────
  function init(config = {}) {
    if (!config.uploadId) {
      console.error("[VoiceAssistant] init() requires uploadId");
      return;
    }
    _config   = Object.assign({
      micButtonId: 'voice-mic-btn',
      statusElementId: 'voice-status',
      autoSpeak: true
    }, config);
    _uploadId = _config.uploadId;
    _setBtnState("idle");
    _setStatus("Ready. Hold the button to speak.", "info");
    _bindButton();
    console.info(`[VoiceAssistant] initialised for upload_id=${_uploadId}`);
  }

  // ── Public API ────────────────────────────────────────────────────────────
  return { init, speakText, speakAgentResponse, stopSpeaking };
})();
