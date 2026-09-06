"""FastAPI application for SIH26104.

Phase 2 -- pure-Python risk scoring over precomputed ML scores:
    GET  /health
    POST /analyze-call        (unchanged risk formula)

Phase 3 -- speaker enrollment + persistent profiles (SQLite):
    POST   /enroll                    multipart: speaker_id, name, audio
    GET    /speakers
    POST   /verify-speaker            multipart: speaker_id, audio
    DELETE /speakers/{speaker_id}

Phase 4 -- near-real-time analysis over WebSocket (see app.realtime):
    WS     /ws/analyze-call/{speaker_id}

No ML inference happens for ``/analyze-call``. ``/enroll`` and
``/verify-speaker`` reuse the existing Phase 1 Resemblyzer functions via
``app.speaker_service``; uploaded audio is processed in a temp file that is
always deleted afterwards -- raw recordings are never persisted.

Run locally (from the ``backend/`` directory)::

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import JSONResponse

from app import database, realtime, speaker_service
from app.risk_engine import calculate_risk
from app.schemas import (
    AnalyzeCallRequest,
    AnalyzeCallResponse,
    DeleteSpeakerResponse,
    EnrollResponse,
    SpeakerListResponse,
    SpeakerProfile,
    VerifySpeakerResponse,
    validate_name,
    validate_speaker_id,
)

# Audio container types we accept for upload. Unknown extensions -> HTTP 415.
_ALLOWED_AUDIO_SUFFIXES = {
    ".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".webm",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure the SQLite schema exists before the first request.
    database.init_db()
    yield


app = FastAPI(
    title="SIH26104 Voice-Cloning Risk Engine",
    version="0.3.0",
    description="Risk scoring (Phase 2) + speaker enrollment & verification (Phase 3).",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Exception handlers -- map service errors to HTTP status codes.
# Stack traces are never sent to the client.
# --------------------------------------------------------------------------

@app.exception_handler(speaker_service.SpeakerNotFoundError)
async def _handle_speaker_not_found(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": f"Speaker '{exc}' not found"})


@app.exception_handler(speaker_service.DuplicateSpeakerError)
async def _handle_duplicate_speaker(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=409, content={"detail": f"Speaker '{exc}' is already enrolled"}
    )


@app.exception_handler(speaker_service.EmbeddingError)
async def _handle_embedding_error(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=400, content={"detail": f"Could not process audio: {exc}"}
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _validated_speaker_id(raw: str) -> str:
    try:
        return validate_speaker_id(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _validated_name(raw: str) -> str:
    try:
        return validate_name(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def _save_upload_to_tempfile(audio: UploadFile) -> str:
    """Persist an upload to a temp file and return its path. Caller must delete it."""
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix and suffix not in _ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"Unsupported audio type '{suffix}'")

    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

    fd, path = tempfile.mkstemp(suffix=suffix or ".wav")
    with os.fdopen(fd, "wb") as handle:
        handle.write(content)
    return path


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


# --------------------------------------------------------------------------
# Phase 2 endpoints (unchanged behaviour)
# --------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/analyze-call", response_model=AnalyzeCallResponse)
def analyze_call(request: AnalyzeCallRequest) -> AnalyzeCallResponse:
    """Score a call from precomputed ML outputs + call context.

    Pydantic validates the request (invalid values -> HTTP 422). The validated
    values are passed straight to the risk engine; nothing is recomputed.
    """
    assessment = calculate_risk(
        synthetic_score=request.synthetic_score,
        speaker_match=request.speaker_match,
        context=request.context.model_dump(),
        transaction_amount=request.transaction_amount,
    )
    return AnalyzeCallResponse(
        synthetic_score=request.synthetic_score,
        speaker_match=request.speaker_match,
        risk_score=assessment.risk_score,
        risk_level=assessment.risk_level,
        decision=assessment.decision,
        risk_factors=assessment.risk_factors,
    )


# --------------------------------------------------------------------------
# Phase 3 endpoints -- speaker enrollment / profiles
# --------------------------------------------------------------------------

@app.post("/enroll", response_model=EnrollResponse)
async def enroll(
    speaker_id: str = Form(...),
    name: str = Form(...),
    audio: UploadFile = File(...),
) -> EnrollResponse:
    """Enroll a speaker: embed the uploaded audio and store the profile.

    The uploaded file is written to a temp path, embedded with the existing
    Phase 1 ``get_embedding``, then deleted. Only the embedding is persisted.
    """
    sid = _validated_speaker_id(speaker_id)
    display_name = _validated_name(name)

    tmp_path = await _save_upload_to_tempfile(audio)
    try:
        record = speaker_service.create_speaker(sid, display_name, tmp_path)
    finally:
        _safe_unlink(tmp_path)

    return EnrollResponse(
        success=True,
        speaker_id=record["speaker_id"],
        name=record["name"],
        message="Speaker enrolled successfully",
    )


@app.get("/speakers", response_model=SpeakerListResponse)
def list_speakers() -> SpeakerListResponse:
    """List enrolled speaker profiles (never includes the embedding)."""
    return SpeakerListResponse(
        speakers=[SpeakerProfile(**profile) for profile in speaker_service.list_speakers()]
    )


@app.post("/verify-speaker", response_model=VerifySpeakerResponse)
async def verify_speaker_endpoint(
    speaker_id: str = Form(...),
    audio: UploadFile = File(...),
) -> VerifySpeakerResponse:
    """Compare uploaded audio against an enrolled speaker's stored embedding."""
    sid = _validated_speaker_id(speaker_id)

    tmp_path = await _save_upload_to_tempfile(audio)
    try:
        result = speaker_service.verify_against_speaker(sid, tmp_path)
    finally:
        _safe_unlink(tmp_path)

    return VerifySpeakerResponse(**result)


@app.delete("/speakers/{speaker_id}", response_model=DeleteSpeakerResponse)
def delete_speaker(speaker_id: str) -> DeleteSpeakerResponse:
    """Delete an enrolled speaker profile (404 if it does not exist)."""
    sid = _validated_speaker_id(speaker_id)
    speaker_service.delete_speaker(sid)  # raises SpeakerNotFoundError -> 404
    return DeleteSpeakerResponse(
        success=True, speaker_id=sid, message="Speaker deleted successfully"
    )


# --------------------------------------------------------------------------
# Phase 4 endpoint -- near-real-time analysis over WebSocket
# --------------------------------------------------------------------------

@app.websocket("/ws/analyze-call/{speaker_id}")
async def ws_analyze_call(websocket: WebSocket, speaker_id: str) -> None:
    """Stream audio; receive per-window risk analysis. Logic lives in app.realtime."""
    await realtime.handle_analyze_call_ws(websocket, speaker_id)
