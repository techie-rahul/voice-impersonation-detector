"""API tests for the Phase 2 FastAPI app, exercised via ``TestClient``.

These tests never touch AASIST or Resemblyzer -- the API consumes precomputed
scores only.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.risk_engine import calculate_risk

client = TestClient(app)


def valid_payload(**overrides) -> dict:
    payload = {
        "synthetic_score": 0.5,
        "speaker_match": 0.5,
        "context": {
            "transfer": False,
            "otp": False,
            "urgent": False,
            "unknown_caller": False,
        },
        "transaction_amount": 1000,
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------
# 1. GET /health
# --------------------------------------------------------------------------

def test_health_returns_200_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --------------------------------------------------------------------------
# 2. POST /analyze-call -- valid request
# --------------------------------------------------------------------------

def test_analyze_call_valid_request_returns_200():
    response = client.post("/analyze-call", json=valid_payload())
    assert response.status_code == 200

    body = response.json()
    assert set(body) == {
        "synthetic_score",
        "speaker_match",
        "risk_score",
        "risk_level",
        "decision",
        "risk_factors",
    }
    # scores are echoed straight back
    assert body["synthetic_score"] == 0.5
    assert body["speaker_match"] == 0.5
    assert isinstance(body["risk_factors"], list)


def test_analyze_call_matches_engine_output():
    payload = valid_payload(
        synthetic_score=0.8,
        speaker_match=0.2,
        context={"transfer": True, "otp": False, "urgent": True, "unknown_caller": False},
        transaction_amount=150_000,
    )
    expected = calculate_risk(
        synthetic_score=payload["synthetic_score"],
        speaker_match=payload["speaker_match"],
        context=payload["context"],
        transaction_amount=payload["transaction_amount"],
    )

    body = client.post("/analyze-call", json=payload).json()
    assert body["risk_score"] == expected.risk_score
    assert body["risk_level"] == expected.risk_level
    assert body["decision"] == expected.decision
    assert body["risk_factors"] == expected.risk_factors


def test_analyze_call_high_risk_blocks():
    payload = valid_payload(
        synthetic_score=0.97,
        speaker_match=0.02,
        context={"transfer": True, "otp": True, "urgent": True, "unknown_caller": True},
        transaction_amount=500_000,
    )
    body = client.post("/analyze-call", json=payload).json()
    assert body["risk_level"] == "HIGH"
    assert body["decision"] == "BLOCK"
    assert body["risk_score"] > 70


def test_analyze_call_low_risk_allows():
    payload = valid_payload(synthetic_score=0.01, speaker_match=0.99, transaction_amount=10)
    body = client.post("/analyze-call", json=payload).json()
    assert body["risk_level"] == "LOW"
    assert body["decision"] == "ALLOW"


def test_analyze_call_context_defaults_when_omitted():
    payload = valid_payload()
    payload.pop("context")
    response = client.post("/analyze-call", json=payload)
    assert response.status_code == 200


# --------------------------------------------------------------------------
# 3-5. Invalid requests -> HTTP 422
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [1.5, -0.1, 2.0])
def test_analyze_call_invalid_synthetic_score_returns_422(bad):
    response = client.post("/analyze-call", json=valid_payload(synthetic_score=bad))
    assert response.status_code == 422


@pytest.mark.parametrize("bad", [1.5, -0.1, 99.0])
def test_analyze_call_invalid_speaker_match_returns_422(bad):
    response = client.post("/analyze-call", json=valid_payload(speaker_match=bad))
    assert response.status_code == 422


@pytest.mark.parametrize("bad", [-1, -0.01, -50000])
def test_analyze_call_negative_transaction_amount_returns_422(bad):
    response = client.post("/analyze-call", json=valid_payload(transaction_amount=bad))
    assert response.status_code == 422


def test_analyze_call_missing_required_field_returns_422():
    payload = valid_payload()
    payload.pop("synthetic_score")
    response = client.post("/analyze-call", json=payload)
    assert response.status_code == 422


def test_analyze_call_non_numeric_score_returns_422():
    response = client.post("/analyze-call", json=valid_payload(synthetic_score="high"))
    assert response.status_code == 422
