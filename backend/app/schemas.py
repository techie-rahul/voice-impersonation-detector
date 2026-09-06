"""Pydantic request/response models for the Phase 2 API.

Validation lives here: out-of-range numeric fields raise ``ValidationError``,
which FastAPI turns into an HTTP 422 response.
"""

from __future__ import annotations

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
