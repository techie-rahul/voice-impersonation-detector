"""Phase 4 -- near-real-time voice analysis over a WebSocket.

Protocol (see ``docs/websocket-api.md``):

* connect to ``/ws/analyze-call/{speaker_id}``
* send one JSON ``{"type": "start", "context": {...}, "transaction_amount": N}``
* stream binary frames of 16 kHz / mono / 16-bit little-endian PCM
* optionally send ``{"type": "stop"}``

For every ~4 s of accumulated audio the server runs, **once per window**:
AASIST (``p_spoof``) -> speaker verification (``speaker_match``) ->
:func:`app.risk_engine.calculate_risk` -- the exact same call
``POST /analyze-call`` makes -- then emits an ``analysis`` message (plus an
``alert`` when ``risk_score > 70``).

Design notes
------------
* No global mutable state: every connection owns a :class:`SessionState`.
* Heavy ML runs via :func:`asyncio.to_thread`; a process-wide lock serialises
  inference so concurrent clients cannot corrupt the shared models and the
  demo box stays responsive.
* Each window is written to a temp WAV that is always deleted (``try/finally``).
  Raw audio and embeddings are never persisted or logged.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app import speaker_service
from app.risk_engine import calculate_risk
from app.schemas import CallContext

# Make the repo-root ``detector`` module importable (Phase 1), without touching it.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# --- Audio / windowing config (env-overridable) ------------------------
SAMPLE_RATE: int = int(os.environ.get("RT_SAMPLE_RATE", "16000"))
SAMPLE_WIDTH: int = 2      # 16-bit PCM
CHANNELS: int = 1          # mono
WINDOW_SECONDS: float = float(os.environ.get("RT_WINDOW_SECONDS", "4.0"))

WINDOW_BYTES: int = int(WINDOW_SECONDS * SAMPLE_RATE) * SAMPLE_WIDTH * CHANNELS
# Safety cap: if a client floods audio faster than we can analyse it, keep only
# the most recent few windows so memory stays bounded during a demo.
_MAX_BUFFER_BYTES: int = WINDOW_BYTES * 8
_KEEP_BYTES: int = WINDOW_BYTES * 4

# One window at a time across all sessions -> shared ML models stay consistent.
_INFERENCE_LOCK = threading.Lock()

HIGH_RISK_ALERT_THRESHOLD: float = 70.0  # matches risk_engine HIGH cut-off

# Opt-in debug capture: when set, every raw mic-window WAV that analyze_window
# writes is also copied here before its temp copy is deleted. Off by default;
# no effect on detection, scoring, or any threshold. Used to collect real
# browser-mic calibration samples (see analyze_window).
_DEBUG_CAPTURE_DIR: str | None = os.environ.get("RT_DEBUG_CAPTURE_DIR")

# --- Live-microphone calibration (documented MVP, not a production-grade fix) ---
# Live mic input differs from ASVspoof's studio-recorded training distribution
# (variable levels, room noise, leading/trailing silence). Mic-mode calibration:
#   * silence-trim + loudness-normalise (calibrate_mic_pcm) -- applied to the
#     SPEAKER-VERIFICATION copy only; measured to raise AASIST false positives if
#     fed to AASIST, so AASIST always sees the raw window.
#   * MIC_MODE_THRESHOLD_OFFSET -- a fixed decision-only reduction applied to the
#     synthetic score handed to the risk engine (shifts ALLOW/VERIFY/BLOCK). The
#     raw AASIST p_spoof reported to the client is never modified.
# Empirically tuned, no calibration dataset.
MIC_TRIM_TOP_DB: float = 25.0
MIC_TARGET_RMS: float = 0.05
MIC_MODE_THRESHOLD_OFFSET: float = 0.15
# The offset targets borderline false positives, not confident detections: a
# p_spoof at/above this is a strong "synthetic" call and is NOT discounted, so a
# real clone still blocks in mic mode.
MIC_OFFSET_PSPOOF_CEILING: float = 0.90

# Quick Scan has no enrolled identity to verify against. Speaker verification is
# skipped entirely; the risk engine still needs a speaker_match in [0, 1], so it
# is given a fixed identity-uncertainty prior (an unverified caller carries some
# baseline suspicion). The AASIST detector and the risk formula are unchanged;
# the analysis message reports speaker_match as null in this mode.
QUICK_SCAN_SPEAKER_MATCH: float = 0.2


class MalformedAudioError(Exception):
    """Raised when a PCM window is not 16-bit aligned / cannot be written."""


class WSError(Exception):
    """A fatal, client-facing error. Carries a stable ``code`` + safe message."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


# ======================================================================
# Per-connection session state
# ======================================================================

@dataclass
class SessionState:
    """Everything one WebSocket connection needs. Never shared between clients.

    ``speaker_id`` is ``None`` for Quick Scan (synthetic-voice detection only,
    no identity verification).
    """

    speaker_id: str | None
    context: dict
    transaction_amount: float
    mic_mode: bool = False  # live microphone input -> apply mic calibration
    started_at: float = field(default_factory=time.monotonic)
    analysis_count: int = 0
    buffer: bytearray = field(default_factory=bytearray)
    _header_checked: bool = False

    def feed(self, data: bytes) -> None:
        self.buffer.extend(data)
        if not self._header_checked and len(self.buffer) >= 12:
            self._header_checked = True
            _strip_leading_wav_header(self.buffer)
        if len(self.buffer) > _MAX_BUFFER_BYTES:
            del self.buffer[:-_KEEP_BYTES]

    def has_full_window(self) -> bool:
        return len(self.buffer) >= WINDOW_BYTES

    def take_window(self) -> bytes:
        window = bytes(self.buffer[:WINDOW_BYTES])
        del self.buffer[:WINDOW_BYTES]
        return window


def _strip_leading_wav_header(buffer: bytearray) -> None:
    """If a client sent a whole WAV file, drop its header so we keep only PCM."""
    if len(buffer) >= 12 and buffer[0:4] == b"RIFF" and buffer[8:12] == b"WAVE":
        data_idx = buffer.find(b"data")
        if data_idx != -1 and len(buffer) >= data_idx + 8:
            del buffer[: data_idx + 8]
        else:
            del buffer[:44]  # standard PCM header length


# ======================================================================
# Blocking work (runs in a worker thread)
# ======================================================================

def run_aasist(wav_path: str) -> float:
    """Existing AASIST detector -> ``p_spoof`` in ``[0, 1]``. Not reimplemented."""
    from detector import detect_synthetic  # lazy: loads the model on first call

    return float(detect_synthetic(wav_path)["p_spoof"])


def run_speaker_verification(speaker_id: str, wav_path: str) -> float:
    """Phase 3 speaker verification -> the raw ``speaker_match`` similarity."""
    result = speaker_service.verify_against_speaker(speaker_id, wav_path)
    return float(result["speaker_match"])


def calibrate_mic_pcm(pcm: bytes) -> bytes:
    """Documented MVP calibration for live-mic conditions (not production-grade).

    Trims leading/trailing silence (``librosa.effects.trim``, ``top_db=25``) and
    RMS-normalises loudness to ``target_rms=0.05`` so a microphone window sits
    closer to AASIST's ASVspoof studio-recorded training distribution. Returns
    16-bit little-endian PCM bytes. Falls back to the input on empty/near-silent
    audio.
    """
    import librosa
    import numpy as np

    y = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    if y.size == 0:
        return pcm

    trimmed, _ = librosa.effects.trim(y, top_db=MIC_TRIM_TOP_DB)
    if trimmed.size >= SAMPLE_RATE // 10:  # keep >= 0.1 s; else it was silence
        y = trimmed

    rms = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0
    if rms > 1e-6:
        # Clamp the gain: a window that is still mostly silence would otherwise
        # be over-amplified and *raise* false-positive synthetic detections.
        gain = float(np.clip(MIC_TARGET_RMS / rms, 0.25, 4.0))
        y = y * gain

    y = np.clip(y, -1.0, 1.0)
    return (y * 32767.0).astype("<i2").tobytes()


def _write_temp_wav(pcm: bytes) -> str:
    if len(pcm) % SAMPLE_WIDTH != 0:
        raise MalformedAudioError("PCM window is not 16-bit aligned")
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        with wave.open(path, "wb") as wav:
            wav.setnchannels(CHANNELS)
            wav.setsampwidth(SAMPLE_WIDTH)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(pcm)
    except wave.Error as exc:
        with contextlib.suppress(OSError):
            os.unlink(path)
        raise MalformedAudioError(f"could not build WAV from PCM: {exc}") from exc
    return path


def analyze_window(pcm: bytes, session: SessionState) -> dict:
    """Analyse one ~4 s window. Returns ``{"analysis": {...}, "alert": {...}|None}``.

    Runs in a worker thread. All heavy calls are serialised by ``_INFERENCE_LOCK``.
    The temp WAV is deleted in ``finally``.
    """
    quick_scan = session.speaker_id is None
    raw_wav = _write_temp_wav(pcm)
    if _DEBUG_CAPTURE_DIR:
        with contextlib.suppress(OSError):
            os.makedirs(_DEBUG_CAPTURE_DIR, exist_ok=True)
            dest = os.path.join(
                _DEBUG_CAPTURE_DIR, f"mic_{time.time_ns()}.wav"
            )
            shutil.copyfile(raw_wav, dest)
    # Mic calibration (trim silence + loudness-normalise) is applied to the
    # SPEAKER-VERIFICATION copy only. Measured on the pretrained ASVspoof model,
    # feeding trimmed/normalised audio to AASIST *raises* false positives badly
    # (AASIST tiles a short trimmed window to 4 s and the repeat seam reads as
    # synthetic), so AASIST always sees the raw window; mic calibration for the
    # synthetic score is the decision-only offset below. Documented MVP, not a
    # production-grade fix.
    calibrated_wav = (
        _write_temp_wav(calibrate_mic_pcm(pcm))
        if (session.mic_mode and not quick_scan)
        else None
    )
    try:
        with _INFERENCE_LOCK:
            # Raw AASIST output -- reported to the client unchanged.
            p_spoof = _clamp01(run_aasist(raw_wav))
            if quick_scan:
                speaker_match: float | None = None
                engine_match = QUICK_SCAN_SPEAKER_MATCH
            else:
                speaker_match = run_speaker_verification(
                    session.speaker_id, calibrated_wav or raw_wav
                )
                engine_match = _clamp01(speaker_match)  # engine needs [0, 1]

            # Decision-only calibration: for live mic, feed a lowered synthetic
            # score to the risk engine (shifts ALLOW/VERIFY/BLOCK), while the
            # reported p_spoof stays raw. Only borderline scores are discounted
            # (a confident detection is left alone so real clones still block).
            # MVP adjustment for mic vs studio audio.
            engine_synthetic = p_spoof
            if session.mic_mode and p_spoof < MIC_OFFSET_PSPOOF_CEILING:
                engine_synthetic = _clamp01(p_spoof - MIC_MODE_THRESHOLD_OFFSET)
            assessment = calculate_risk(
                synthetic_score=engine_synthetic,
                speaker_match=engine_match,
                context=session.context,
                transaction_amount=session.transaction_amount,
            )
    finally:
        for p in (raw_wav, calibrated_wav):
            if p:
                with contextlib.suppress(OSError):
                    os.unlink(p)

    factors = assessment.risk_factors
    if quick_scan:
        # No identity was verified, so don't surface an identity-mismatch factor
        # (it stems only from the fixed prior). The score itself is unchanged.
        factors = [f for f in factors if f != "Speaker identity mismatch"]

    analysis = {
        "type": "analysis",
        "synthetic_score": p_spoof,
        "speaker_match": None if speaker_match is None else float(speaker_match),
        "risk_score": assessment.risk_score,
        "risk_level": assessment.risk_level,
        "decision": assessment.decision,
        "risk_factors": factors,
    }
    alert = None
    if assessment.risk_score > HIGH_RISK_ALERT_THRESHOLD:
        alert = {
            "type": "alert",
            "severity": "HIGH",
            "message": "Potential voice impersonation detected",
            "risk_score": assessment.risk_score,
        }
    return {"analysis": analysis, "alert": alert}


# ======================================================================
# WebSocket handler
# ======================================================================

async def _send_error(websocket: WebSocket, code: str, message: str) -> None:
    with contextlib.suppress(Exception):
        await websocket.send_json({"type": "error", "code": code, "message": message})


def _parse_start_message(payload: object) -> tuple[dict, float, bool]:
    """Validate the opening ``start`` message.

    Returns ``(context dict, transaction_amount, mic_mode)``. ``mic_mode`` is an
    optional client hint that this session's audio is a live microphone (vs a
    demo-file upload); it enables the mic calibration in :func:`analyze_window`.
    """
    if not isinstance(payload, dict) or payload.get("type") != "start":
        raise WSError("INVALID_START", "First message must be JSON with type 'start'")

    raw_context = payload.get("context", {})
    if raw_context is None:
        raw_context = {}
    if not isinstance(raw_context, dict):
        raise WSError("INVALID_CONTEXT", "context must be an object of boolean flags")
    try:
        context = CallContext.model_validate(raw_context).model_dump()
    except ValidationError as exc:
        raise WSError("INVALID_CONTEXT", "context flags must be booleans") from exc

    raw_amount = payload.get("transaction_amount", 0)
    try:
        amount = float(raw_amount)
    except (TypeError, ValueError) as exc:
        raise WSError(
            "INVALID_TRANSACTION_AMOUNT", "transaction_amount must be a number >= 0"
        ) from exc
    if not math.isfinite(amount) or amount < 0:
        raise WSError(
            "INVALID_TRANSACTION_AMOUNT", "transaction_amount must be a number >= 0"
        )

    mic_mode = bool(payload.get("mic_mode", False))
    return context, amount, mic_mode


async def _drain_windows(websocket: WebSocket, session: SessionState) -> bool:
    """Analyse every complete buffered window. Returns ``True`` if the session
    must end (fatal error)."""
    while session.has_full_window():
        pcm = session.take_window()
        try:
            result = await asyncio.to_thread(analyze_window, pcm, session)
        except MalformedAudioError as exc:
            await _send_error(websocket, "MALFORMED_AUDIO", str(exc))
            return True
        except speaker_service.SpeakerNotFoundError:
            await _send_error(
                websocket,
                "SPEAKER_NOT_FOUND",
                f"Speaker '{session.speaker_id}' is not enrolled",
            )
            return True
        except Exception:  # noqa: BLE001 - never leak a traceback to the client
            await _send_error(
                websocket, "INFERENCE_ERROR", "Failed to analyse the audio window"
            )
            continue

        session.analysis_count += 1
        await websocket.send_json(result["analysis"])
        if result["alert"] is not None:
            await websocket.send_json(result["alert"])
    return False


async def handle_analyze_call_ws(
    websocket: WebSocket, speaker_id: str | None
) -> None:
    """Entry point for both WebSocket routes.

    ``speaker_id`` is a string for Identity Protection
    (``/ws/analyze-call/{speaker_id}``) and ``None`` for Quick Scan
    (``/ws/analyze-call``), which skips speaker verification.
    """
    await websocket.accept()
    try:
        # 1. Identity Protection only: the speaker must be enrolled (Phase 3).
        if speaker_id is not None:
            try:
                await asyncio.to_thread(speaker_service.get_speaker, speaker_id)
            except speaker_service.SpeakerNotFoundError:
                await _send_error(
                    websocket,
                    "SPEAKER_NOT_FOUND",
                    f"Speaker '{speaker_id}' is not enrolled",
                )
                return

        # 2. first frame must be a valid JSON 'start'.
        try:
            first = await websocket.receive()
        except WebSocketDisconnect:
            return
        if first.get("type") == "websocket.disconnect":
            return
        if first.get("text") is None:
            await _send_error(
                websocket, "INVALID_START", "Expected a JSON 'start' message before audio"
            )
            return
        try:
            payload = json.loads(first["text"])
        except json.JSONDecodeError:
            await _send_error(websocket, "INVALID_START", "start message is not valid JSON")
            return
        try:
            context, amount, mic_mode = _parse_start_message(payload)
        except WSError as exc:
            await _send_error(websocket, exc.code, exc.message)
            return

        session = SessionState(
            speaker_id=speaker_id,
            context=context,
            transaction_amount=amount,
            mic_mode=mic_mode,
        )
        await websocket.send_json(
            {
                "type": "started",
                "mode": "identity" if speaker_id is not None else "quick-scan",
                "speaker_id": speaker_id,
                "mic_calibrated": mic_mode,
                "window_seconds": WINDOW_SECONDS,
                "sample_rate": SAMPLE_RATE,
            }
        )

        # 3. stream loop.
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                break
            if message.get("type") == "websocket.disconnect":
                break

            text = message.get("text")
            if text is not None:
                try:
                    control = json.loads(text)
                except json.JSONDecodeError:
                    await _send_error(websocket, "INVALID_MESSAGE", "Message is not valid JSON")
                    continue
                if isinstance(control, dict) and control.get("type") == "stop":
                    await websocket.send_json(
                        {"type": "stopped", "analyses": session.analysis_count}
                    )
                    break
                await _send_error(
                    websocket,
                    "INVALID_MESSAGE",
                    "Expected binary audio or a {'type': 'stop'} message",
                )
                continue

            data = message.get("bytes")
            if data is None:
                continue
            if len(data) == 0:
                await _send_error(
                    websocket, "EMPTY_AUDIO", "Received an empty binary audio frame"
                )
                continue

            session.feed(data)
            if await _drain_windows(websocket, session):
                break
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()
