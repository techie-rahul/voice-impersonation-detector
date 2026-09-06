"""Pytest bootstrap + shared fixtures for the backend tests.

* Adds ``backend/`` to ``sys.path`` so ``import app.*`` works from any CWD.
* Redirects the speaker SQLite DB to a throwaway file for every test, so no
  test ever touches ``backend/data/speakers.db``.
* Provides a ``client`` fixture and an ``mock_ml`` fixture that stubs the
  Phase 1 embedding / verification functions (no Resemblyzer load).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

REPO_ROOT = _BACKEND_DIR.parent

# One real ASVspoof speaker (LA_0079) with two genuine clips, used by the
# single integration test. Absent on machines without the dataset -> skipped.
ASVSPOOF_FLAC_DIR = (
    REPO_ROOT / "data" / "asvspoof2019" / "LA" / "LA"
    / "ASVspoof2019_LA_train" / "flac"
)
SAME_SPEAKER_CLIP_A = ASVSPOOF_FLAC_DIR / "LA_T_1138215.flac"  # LA_0079
SAME_SPEAKER_CLIP_B = ASVSPOOF_FLAC_DIR / "LA_T_1271820.flac"  # LA_0079
OTHER_SPEAKER_CLIP = ASVSPOOF_FLAC_DIR / "LA_T_1078395.flac"   # LA_0080


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "integration: exercises the real Resemblyzer verification module"
    )


@pytest.fixture(autouse=True)
def isolated_speaker_db(tmp_path, monkeypatch):
    """Point the DB module at a fresh temp file for the duration of each test."""
    from app import database

    db_file = tmp_path / "speakers.db"
    monkeypatch.setattr(database, "DB_PATH", db_file)
    database.init_db()
    yield db_file


@pytest.fixture
def client(isolated_speaker_db):
    """FastAPI TestClient with lifespan run against the isolated DB."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


class _MLStub:
    """Controllable stand-in for the Phase 1 ML functions."""

    def __init__(self) -> None:
        # Deterministic 256-D embedding; value irrelevant because verify is stubbed.
        self.embedding = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        self.score = 0.9  # returned by verify_speaker; tests override per case

    def get_embedding(self, audio_path: str) -> np.ndarray:
        return self.embedding.copy()

    def verify_speaker(self, audio_path: str, enrolled_embedding: np.ndarray) -> float:
        return self.score


@pytest.fixture
def mock_ml(monkeypatch) -> _MLStub:
    """Replace ``speaker_service.get_embedding`` / ``verify_speaker`` with stubs."""
    from app import speaker_service

    stub = _MLStub()
    monkeypatch.setattr(speaker_service, "get_embedding", stub.get_embedding)
    monkeypatch.setattr(speaker_service, "verify_speaker", stub.verify_speaker)
    return stub


@pytest.fixture
def wav_bytes() -> bytes:
    """A minimal, non-empty RIFF/WAVE byte blob for upload tests (never embedded)."""
    return b"RIFF" + (2048).to_bytes(4, "little") + b"WAVE" + b"\x00" * 2048
