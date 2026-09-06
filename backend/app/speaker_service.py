"""Speaker enrollment + verification service (Phase 3).

Bridges the SQLite store (:mod:`app.database`) and the EXISTING Phase 1
Resemblyzer functions in ``verification/verifier.py``. No ML logic is
reimplemented here -- ``get_embedding`` / ``verify_speaker`` below are thin,
lazily-importing shims around the Phase 1 module so that:

* importing this module (and app startup) does not pay the Resemblyzer
  model-load cost until an embedding is actually needed, and
* unit tests can monkeypatch these two names without loading Resemblyzer.

Verification threshold
----------------------
``SPEAKER_MATCH_THRESHOLD`` decides only the boolean ``verified`` field. It does
NOT alter the similarity score returned by the Phase 1 ``verify_speaker``.
It is an MVP heuristic, not a scientifically calibrated production threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from app import database

# Make ``verification/verifier.py`` importable without changing Phase 1 layout.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_VERIFICATION_DIR = _REPO_ROOT / "verification"
if str(_VERIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VERIFICATION_DIR))

# MVP heuristic. speaker_match >= 0.70 -> verified = True; below -> False.
SPEAKER_MATCH_THRESHOLD: float = 0.70


class SpeakerNotFoundError(Exception):
    """Raised when no profile exists for a given ``speaker_id``."""


class DuplicateSpeakerError(Exception):
    """Raised when enrolling a ``speaker_id`` that is already enrolled."""


class EmbeddingError(Exception):
    """Raised when an embedding cannot be produced from the supplied audio."""


# --- Phase 1 ML shims (lazy import; patch-friendly) --------------------

def get_embedding(audio_path: str) -> np.ndarray:
    """Delegate to the existing Phase 1 ``verifier.get_embedding``."""
    from verifier import get_embedding as _phase1_get_embedding

    return _phase1_get_embedding(audio_path)


def verify_speaker(audio_path: str, enrolled_embedding: np.ndarray) -> float:
    """Delegate to the existing Phase 1 ``verifier.verify_speaker``."""
    from verifier import verify_speaker as _phase1_verify_speaker

    return _phase1_verify_speaker(audio_path, enrolled_embedding)


# --- helpers -----------------------------------------------------------

def _embed(audio_path: str) -> np.ndarray:
    """Run Phase 1 embedding and normalise the result to 1-D float32."""
    try:
        raw = get_embedding(audio_path)
    except FileNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any ML/IO failure uniformly
        raise EmbeddingError(str(exc)) from exc

    embedding = np.asarray(raw, dtype=np.float32).ravel()
    if embedding.size == 0 or not np.all(np.isfinite(embedding)):
        raise EmbeddingError("audio did not yield a valid speaker embedding")
    return embedding


def _row_to_profile(row) -> dict:
    return {
        "speaker_id": row["speaker_id"],
        "name": row["name"],
        "created_at": row["created_at"],
    }


# --- public API ------------------------------------------------------

def create_speaker(speaker_id: str, name: str, audio_path: str) -> dict:
    """Generate an embedding from ``audio_path`` and store a new profile.

    Returns ``{"speaker_id", "name", "created_at"}``.
    Raises :class:`DuplicateSpeakerError` / :class:`EmbeddingError` /
    ``FileNotFoundError``.
    """
    if database.get_speaker(speaker_id) is not None:
        raise DuplicateSpeakerError(speaker_id)

    embedding = _embed(audio_path)

    try:
        return database.insert_speaker(speaker_id, name, embedding)
    except database.DuplicateSpeakerError as exc:  # race: inserted meanwhile
        raise DuplicateSpeakerError(speaker_id) from exc


def get_speaker(speaker_id: str) -> dict:
    """Return a single profile (no embedding). Raises :class:`SpeakerNotFoundError`."""
    row = database.get_speaker(speaker_id)
    if row is None:
        raise SpeakerNotFoundError(speaker_id)
    return _row_to_profile(row)


def list_speakers() -> list[dict]:
    """Return all enrolled profiles (no embeddings)."""
    return [_row_to_profile(row) for row in database.list_speakers()]


def delete_speaker(speaker_id: str) -> None:
    """Remove a profile. Raises :class:`SpeakerNotFoundError` if absent."""
    if not database.delete_speaker(speaker_id):
        raise SpeakerNotFoundError(speaker_id)


def verify_against_speaker(speaker_id: str, audio_path: str) -> dict:
    """Compare ``audio_path`` against an enrolled speaker's stored embedding.

    Returns ``{"speaker_id", "speaker_match", "verified"}`` where
    ``speaker_match`` is the unmodified Phase 1 similarity score and
    ``verified`` is ``speaker_match >= SPEAKER_MATCH_THRESHOLD``.
    """
    row = database.get_speaker(speaker_id)
    if row is None:
        raise SpeakerNotFoundError(speaker_id)

    enrolled_embedding = database.blob_to_embedding(row["embedding"])

    try:
        score = verify_speaker(audio_path, enrolled_embedding)
    except FileNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise EmbeddingError(str(exc)) from exc

    score = float(score)
    return {
        "speaker_id": speaker_id,
        "speaker_match": score,
        "verified": score >= SPEAKER_MATCH_THRESHOLD,
    }
