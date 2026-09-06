"""API tests for the Phase 3 speaker endpoints (ML functions stubbed via ``mock_ml``).

Also re-checks that the Phase 2 endpoints still behave exactly as before.
"""

from __future__ import annotations

import pytest


def enroll(client, wav_bytes, speaker_id="rahul", name="Rahul", filename="sample.wav"):
    return client.post(
        "/enroll",
        data={"speaker_id": speaker_id, "name": name},
        files={"audio": (filename, wav_bytes, "audio/wav")},
    )


# --------------------------------------------------------------------------
# 1. POST /enroll works
# --------------------------------------------------------------------------

def test_enroll_success(client, mock_ml, wav_bytes):
    response = enroll(client, wav_bytes)
    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "speaker_id": "rahul",
        "name": "Rahul",
        "message": "Speaker enrolled successfully",
    }


# --------------------------------------------------------------------------
# 2. Duplicate enrollment returns 409
# --------------------------------------------------------------------------

def test_enroll_duplicate_returns_409(client, mock_ml, wav_bytes):
    assert enroll(client, wav_bytes).status_code == 200
    dup = enroll(client, wav_bytes)
    assert dup.status_code == 409
    assert "already enrolled" in dup.json()["detail"]


# --------------------------------------------------------------------------
# 3. GET /speakers works (and never leaks embeddings)
# --------------------------------------------------------------------------

def test_list_speakers(client, mock_ml, wav_bytes):
    enroll(client, wav_bytes, speaker_id="rahul", name="Rahul")
    enroll(client, wav_bytes, speaker_id="saanvi", name="Saanvi")

    response = client.get("/speakers")
    assert response.status_code == 200
    speakers = response.json()["speakers"]
    assert {s["speaker_id"] for s in speakers} == {"rahul", "saanvi"}
    for s in speakers:
        assert set(s) == {"speaker_id", "name", "created_at"}
        assert "embedding" not in s


def test_list_speakers_empty(client, mock_ml):
    assert client.get("/speakers").json() == {"speakers": []}


# --------------------------------------------------------------------------
# 4. POST /verify-speaker works  +  7/8/9 threshold behaviour
# --------------------------------------------------------------------------

def test_verify_speaker_success_returns_score(client, mock_ml, wav_bytes):
    enroll(client, wav_bytes)
    mock_ml.score = 0.83

    response = client.post(
        "/verify-speaker",
        data={"speaker_id": "rahul"},
        files={"audio": ("probe.wav", wav_bytes, "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["speaker_id"] == "rahul"
    assert body["speaker_match"] == pytest.approx(0.83)
    assert isinstance(body["speaker_match"], float)
    assert body["verified"] is True


@pytest.mark.parametrize(
    "score, verified",
    [(0.90, True), (0.70, True), (0.699, False), (0.20, False)],
)
def test_verify_speaker_threshold(client, mock_ml, wav_bytes, score, verified):
    enroll(client, wav_bytes)
    mock_ml.score = score
    body = client.post(
        "/verify-speaker",
        data={"speaker_id": "rahul"},
        files={"audio": ("probe.wav", wav_bytes, "audio/wav")},
    ).json()
    assert body["speaker_match"] == pytest.approx(score)
    assert body["verified"] is verified


# --------------------------------------------------------------------------
# 5. Unknown speaker returns 404
# --------------------------------------------------------------------------

def test_verify_unknown_speaker_returns_404(client, mock_ml, wav_bytes):
    response = client.post(
        "/verify-speaker",
        data={"speaker_id": "ghost"},
        files={"audio": ("probe.wav", wav_bytes, "audio/wav")},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_delete_unknown_speaker_returns_404(client, mock_ml):
    response = client.delete("/speakers/ghost")
    assert response.status_code == 404


# --------------------------------------------------------------------------
# 6. DELETE /speakers/{speaker_id} works
# --------------------------------------------------------------------------

def test_delete_speaker(client, mock_ml, wav_bytes):
    enroll(client, wav_bytes)
    deleted = client.delete("/speakers/rahul")
    assert deleted.status_code == 200
    assert deleted.json()["success"] is True

    # gone now
    assert client.get("/speakers").json() == {"speakers": []}
    assert client.delete("/speakers/rahul").status_code == 404


# --------------------------------------------------------------------------
# 8. Validation / error handling
# --------------------------------------------------------------------------

def test_enroll_missing_audio_returns_422(client, mock_ml):
    response = client.post("/enroll", data={"speaker_id": "rahul", "name": "Rahul"})
    assert response.status_code == 422


def test_enroll_empty_audio_returns_400(client, mock_ml):
    response = client.post(
        "/enroll",
        data={"speaker_id": "rahul", "name": "Rahul"},
        files={"audio": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400


def test_enroll_unsupported_audio_type_returns_415(client, mock_ml, wav_bytes):
    response = client.post(
        "/enroll",
        data={"speaker_id": "rahul", "name": "Rahul"},
        files={"audio": ("notes.txt", wav_bytes, "text/plain")},
    )
    assert response.status_code == 415


@pytest.mark.parametrize("bad_id", ["", "  ", "has space", "bad/slash", "x" * 65, "drop;table"])
def test_enroll_invalid_speaker_id_returns_422(client, mock_ml, wav_bytes, bad_id):
    assert enroll(client, wav_bytes, speaker_id=bad_id).status_code == 422


@pytest.mark.parametrize("bad_name", ["", "   "])
def test_enroll_invalid_name_returns_422(client, mock_ml, wav_bytes, bad_name):
    assert enroll(client, wav_bytes, name=bad_name).status_code == 422


def test_enroll_embedding_failure_returns_400(client, wav_bytes, monkeypatch):
    from app import speaker_service

    def boom(_path):
        raise RuntimeError("bad audio content")

    monkeypatch.setattr(speaker_service, "get_embedding", boom)
    response = enroll(client, wav_bytes)
    assert response.status_code == 400
    assert "Could not process audio" in response.json()["detail"]
    # nothing persisted
    assert client.get("/speakers").json() == {"speakers": []}


def test_error_responses_have_no_traceback(client, mock_ml, wav_bytes):
    body = client.post(
        "/verify-speaker",
        data={"speaker_id": "ghost"},
        files={"audio": ("p.wav", wav_bytes, "audio/wav")},
    ).json()
    assert set(body) == {"detail"}
    assert "Traceback" not in body["detail"]


# --------------------------------------------------------------------------
# 13. Backward compatibility -- Phase 2 endpoints unchanged
# --------------------------------------------------------------------------

def test_health_still_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_call_still_works_unchanged(client):
    payload = {
        "synthetic_score": 0.8,
        "speaker_match": 0.2,
        "context": {"transfer": True, "otp": False, "urgent": True, "unknown_caller": False},
        "transaction_amount": 150_000,
    }
    response = client.post("/analyze-call", json=payload)
    assert response.status_code == 200
    body = response.json()
    # 0.8*0.5 + 0.8*0.3 + (0.25+0.15+0.20)/1.0 * 0.2  -> 76.0
    assert body["risk_score"] == pytest.approx(76.0)
    assert body["risk_level"] == "HIGH"
    assert body["decision"] == "BLOCK"


def test_analyze_call_still_rejects_invalid_input(client):
    bad = {"synthetic_score": 1.5, "speaker_match": 0.5, "transaction_amount": 0}
    assert client.post("/analyze-call", json=bad).status_code == 422
