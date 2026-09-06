"""Phase 2 FastAPI application.

Exposes the pure-Python risk engine over HTTP. **No ML inference happens here.**
The client supplies already-computed scores (``synthetic_score`` from AASIST,
``speaker_match`` from speaker verification) plus call context; this service
only blends them via :func:`calculate_risk`.

Run locally (from the ``backend/`` directory)::

    uvicorn app.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI

from app.risk_engine import calculate_risk
from app.schemas import AnalyzeCallRequest, AnalyzeCallResponse

app = FastAPI(
    title="SIH26104 Voice-Cloning Risk Engine",
    version="0.1.0",
    description="Phase 2: risk scoring API over precomputed ML scores.",
)


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
