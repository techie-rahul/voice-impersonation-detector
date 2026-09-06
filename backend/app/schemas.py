"""Pydantic request/response models for the API.

Validation lives here: out-of-range numeric fields raise ``ValidationError``,
which FastAPI turns into an HTTP 422 response.

Phase 2 models: ``CallContext``, ``AnalyzeCallRequest``, ``AnalyzeCallResponse``.
Phase 3 models: speaker enrollment / listing / verification / deletion.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


class CallContext(BaseModel):
    """Boolean call-context signals. All default to ``False`` when omitted."""

    transfer: bool = False
    otp: bool = False
    urgent: bool = False
    unknown_caller: bool = False


class AnalyzeCallRequest(BaseModel):
    """Input to ``POST /analyze-call`` -- scores are precomputed upstream."""

    synthetic_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="AASIST spoof probability (p_spoof), 0..1; higher = more synthetic.",
    )
    speaker_match: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Speaker-verification similarity, 0..1; higher = same speaker.",
    )
    context: CallContext = Field(
        default_factory=CallContext,
        description="Boolean call-context signals.",
    )
    transaction_amount: float = Field(
        ...,
        ge=0.0,
        description="Transaction value in INR, must be >= 0.",
    )


class AnalyzeCallResponse(BaseModel):
    """Result of ``POST /analyze-call``."""

    synthetic_score: float
    speaker_match: float
    risk_score: float
    risk_level: str
    decision: str
    risk_factors: list[str]


# ==========================================================================
# Phase 3 -- speaker enrollment / profiles
# ==========================================================================

# ``speaker_id`` is a URL path segment and a DB key: keep it simple/safe.
SPEAKER_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
MAX_NAME_LENGTH = 128


def validate_speaker_id(value: str) -> str:
    """Return a cleaned ``speaker_id`` or raise ``ValueError``."""
    cleaned = (value or "").strip()
    if not SPEAKER_ID_PATTERN.fullmatch(cleaned):
        raise ValueError(
            "speaker_id must be 1-64 chars of letters, digits, '.', '_' or '-'"
        )
    return cleaned


def validate_name(value: str) -> str:
    """Return a cleaned display ``name`` or raise ``ValueError``."""
    cleaned = (value or "").strip()
    if not cleaned:
        raise ValueError("name must not be empty")
    if len(cleaned) > MAX_NAME_LENGTH:
        raise ValueError(f"name must be at most {MAX_NAME_LENGTH} characters")
    return cleaned


class EnrollResponse(BaseModel):
    """Result of ``POST /enroll``."""

    success: bool
    speaker_id: str
    name: str
    message: str


class SpeakerProfile(BaseModel):
    """A stored speaker profile as exposed by the API (never the embedding)."""

    speaker_id: str
    name: str
    created_at: str


class SpeakerListResponse(BaseModel):
    """Result of ``GET /speakers``."""

    speakers: list[SpeakerProfile]


class VerifySpeakerResponse(BaseModel):
    """Result of ``POST /verify-speaker``."""

    speaker_id: str
    speaker_match: float
    verified: bool


class DeleteSpeakerResponse(BaseModel):
    """Result of ``DELETE /speakers/{speaker_id}``."""

    success: bool
    speaker_id: str
    message: str
