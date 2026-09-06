"""Unit tests for ``app.speaker_service`` with the Phase 1 ML functions stubbed.

``mock_ml`` (conftest) replaces ``speaker_service.get_embedding`` /
``verify_speaker`` so no Resemblyzer model is loaded here.
"""

from __future__ import annotations

import numpy as np
import pytest

from app import speaker_service


def test_threshold_constant_is_070():
    assert speaker_service.SPEAKER_MATCH_THRESHOLD == 0.70


# --------------------------------------------------------------------------
# create / get / list / delete
# --------------------------------------------------------------------------

def test_create_speaker_persists_profile(mock_ml):
    record = speaker_service.create_speaker("rahul", "Rahul", "/tmp/whatever.wav")
    assert record == {
        "speaker_id": "rahul",
        "name": "Rahul",
        "created_at": record["created_at"],
    }
    assert speaker_service.get_speaker("rahul")["name"] == "Rahul"


def test_create_speaker_duplicate_raises(mock_ml):
    speaker_service.create_speaker("rahul", "Rahul", "/tmp/a.wav")
    with pytest.raises(speaker_service.DuplicateSpeakerError):
        speaker_service.create_speaker("rahul", "Rahul Again", "/tmp/b.wav")


def test_get_speaker_unknown_raises(mock_ml):
    with pytest.raises(speaker_service.SpeakerNotFoundError):
        speaker_service.get_speaker("missing")


def test_list_speakers_returns_profiles_without_embeddings(mock_ml):
    speaker_service.create_speaker("a", "A", "/tmp/a.wav")
    speaker_service.create_speaker("b", "B", "/tmp/b.wav")
    profiles = speaker_service.list_speakers()
    assert {p["speaker_id"] for p in profiles} == {"a", "b"}
    assert all(set(p) == {"speaker_id", "name", "created_at"} for p in profiles)


def test_delete_speaker_unknown_raises(mock_ml):
    with pytest.raises(speaker_service.SpeakerNotFoundError):
        speaker_service.delete_speaker("missing")


def test_delete_speaker_removes_profile(mock_ml):
    speaker_service.create_speaker("rahul", "Rahul", "/tmp/a.wav")
    speaker_service.delete_speaker("rahul")
    with pytest.raises(speaker_service.SpeakerNotFoundError):
        speaker_service.get_speaker("rahul")


# --------------------------------------------------------------------------
# embedding-generation failure surfaces as EmbeddingError
# --------------------------------------------------------------------------

def test_create_speaker_wraps_embedding_failure(monkeypatch):
    def boom(_path):
        raise RuntimeError("corrupt audio")

    monkeypatch.setattr(speaker_service, "get_embedding", boom)
    with pytest.raises(speaker_service.EmbeddingError):
        speaker_service.create_speaker("x", "X", "/tmp/bad.wav")


def test_create_speaker_propagates_missing_file(monkeypatch):
    def missing(_path):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(speaker_service, "get_embedding", missing)
    with pytest.raises(FileNotFoundError):
        speaker_service.create_speaker("x", "X", "/tmp/missing.wav")


# --------------------------------------------------------------------------
# verify_against_speaker: score passthrough + threshold -> verified
# --------------------------------------------------------------------------

def test_verify_returns_score_and_unknown_speaker_raises(mock_ml):
    with pytest.raises(speaker_service.SpeakerNotFoundError):
        speaker_service.verify_against_speaker("nobody", "/tmp/a.wav")


def test_verify_score_is_passed_through_unmodified(mock_ml):
    speaker_service.create_speaker("rahul", "Rahul", "/tmp/a.wav")
    mock_ml.score = 0.834219
    result = speaker_service.verify_against_speaker("rahul", "/tmp/probe.wav")
    assert result["speaker_id"] == "rahul"
    assert result["speaker_match"] == pytest.approx(0.834219)


@pytest.mark.parametrize(
    "score, expected_verified",
    [
        (0.95, True),
        (0.70, True),    # threshold is inclusive
        (0.6999, False),
        (0.10, False),
    ],
)
def test_verify_threshold_decides_verified_flag(mock_ml, score, expected_verified):
    speaker_service.create_speaker("rahul", "Rahul", "/tmp/a.wav")
    mock_ml.score = score
    result = speaker_service.verify_against_speaker("rahul", "/tmp/probe.wav")
    assert result["speaker_match"] == pytest.approx(score)
    assert result["verified"] is expected_verified


def test_verify_wraps_ml_failure(mock_ml, monkeypatch):
    speaker_service.create_speaker("rahul", "Rahul", "/tmp/a.wav")

    def boom(_path, _emb):
        raise RuntimeError("resemblyzer blew up")

    monkeypatch.setattr(speaker_service, "verify_speaker", boom)
    with pytest.raises(speaker_service.EmbeddingError):
        speaker_service.verify_against_speaker("rahul", "/tmp/probe.wav")


def test_stored_embedding_is_float32_1d(mock_ml):
    speaker_service.create_speaker("rahul", "Rahul", "/tmp/a.wav")
    from app import database

    restored = database.blob_to_embedding(database.get_speaker("rahul")["embedding"])
    assert restored.dtype == np.float32
    assert restored.ndim == 1
