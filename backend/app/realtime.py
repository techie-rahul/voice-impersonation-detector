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
    """Everything one WebSocket connection needs. Never shared between clients."""

    speaker_id: str
    context: dict
    transaction_amount: float
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
    wav_path = _write_temp_wav(pcm)
    try:
        with _INFERENCE_LOCK:
            p_spoof = _clamp01(run_aasist(wav_path))
            speaker_match = run_speaker_verification(session.speaker_id, wav_path)
            assessment = calculate_risk(
                synthetic_score=p_spoof,
                # response reports the raw score; the engine needs [0, 1].
                speaker_match=_clamp01(speaker_match),
                context=session.context,
                transaction_amount=session.transaction_amount,
            )
    finally:
        with contextlib.suppress(OSError):
            os.unlink(wav_path)

    analysis = {
        "type": "analysis",
        "synthetic_score": p_spoof,
        "speaker_match": float(speaker_match),
        "risk_score": assessment.risk_score,
        "risk_level": assessment.risk_level,
        "decision": assessment.decision,
        "risk_factors": assessment.risk_factors,
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


def _parse_start_message(payload: object) -> tuple[dict, float]:
    """Validate the opening ``start`` message -> (context dict, transaction_amount)."""
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

    return context, amount


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


async def handle_analyze_call_ws(websocket: WebSocket, speaker_id: str) -> None:
    """Entry point wired to ``@app.websocket('/ws/analyze-call/{speaker_id}')``."""
    await websocket.accept()
    try:
        # 1. speaker must be enrolled (Phase 3 store).
        try:
            await asyncio.to_thread(speaker_service.get_speaker, speaker_id)
        except speaker_service.SpeakerNotFoundError:
            await _send_error(
                websocket, "SPEAKER_NOT_FOUND", f"Speaker '{speaker_id}' is not enrolled"
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
            context, amount = _parse_start_message(payload)
        except WSError as exc:
            await _send_error(websocket, exc.code, exc.message)
            return

        session = SessionState(
            speaker_id=speaker_id, context=context, transaction_amount=amount
        )
        await websocket.send_json(
            {
                "type": "started",
                "speaker_id": speaker_id,
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
