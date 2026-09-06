"""Unit tests for the SQLite speaker store (``app.database``).

The ``isolated_speaker_db`` autouse fixture (conftest) redirects ``DB_PATH`` to a
throwaway file, so these never touch the real ``backend/data/speakers.db``.
"""

from __future__ import annotations

import numpy as np
import pytest

from app import database


def make_embedding(seed: float = 0.0) -> np.ndarray:
    return (np.arange(database.EMBEDDING_DIM, dtype=np.float32) + seed) / 100.0


# --------------------------------------------------------------------------
# 1. Database initialises correctly
# --------------------------------------------------------------------------

def test_init_db_creates_speakers_table():
    database.init_db()
    with database._connection() as conn:
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='speakers'"
        ).fetchone()
        assert table is not None
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(speakers)")}
    assert columns == {"id", "speaker_id", "name", "embedding", "created_at"}


def test_init_db_is_idempotent():
    database.init_db()
    database.init_db()  # must not raise
    assert database.list_speakers() == []


# --------------------------------------------------------------------------
# 2. Speaker can be inserted
# --------------------------------------------------------------------------

def test_insert_speaker_returns_metadata():
    record = database.insert_speaker("spk1", "Speaker One", make_embedding())
    assert record["speaker_id"] == "spk1"
    assert record["name"] == "Speaker One"
    assert isinstance(record["created_at"], str) and record["created_at"]
    assert len(database.list_speakers()) == 1


# --------------------------------------------------------------------------
# 3. Speaker can be retrieved (with embedding round-trip)
# --------------------------------------------------------------------------

def test_get_speaker_round_trips_embedding():
    original = make_embedding(seed=7.0)
    database.insert_speaker("spk1", "Speaker One", original)

    row = database.get_speaker("spk1")
    assert row is not None
    assert row["speaker_id"] == "spk1"
    assert row["name"] == "Speaker One"

    restored = database.blob_to_embedding(row["embedding"])
    assert restored.dtype == np.float32
    assert restored.shape == (database.EMBEDDING_DIM,)
    np.testing.assert_array_equal(restored, original)


def test_get_speaker_returns_none_when_absent():
    assert database.get_speaker("nope") is None


def test_embedding_blob_round_trip_is_lossless_for_float32():
    emb = np.random.default_rng(0).standard_normal(256).astype(np.float32)
    restored = database.blob_to_embedding(database.embedding_to_blob(emb))
    np.testing.assert_array_equal(restored, emb)


def test_list_speakers_excludes_embedding_and_is_ordered():
    database.insert_speaker("a", "A", make_embedding())
    database.insert_speaker("b", "B", make_embedding())
    rows = database.list_speakers()
    assert [r["speaker_id"] for r in rows] == ["a", "b"]
    assert "embedding" not in rows[0].keys()


# --------------------------------------------------------------------------
# 4. Speaker can be deleted
# --------------------------------------------------------------------------

def test_delete_speaker_removes_row():
    database.insert_speaker("spk1", "Speaker One", make_embedding())
    assert database.delete_speaker("spk1") is True
    assert database.get_speaker("spk1") is None


def test_delete_missing_speaker_returns_false():
    assert database.delete_speaker("ghost") is False


# --------------------------------------------------------------------------
# 5. Duplicate speaker_id is rejected
# --------------------------------------------------------------------------

def test_duplicate_speaker_id_raises():
    database.insert_speaker("dup", "First", make_embedding())
    with pytest.raises(database.DuplicateSpeakerError):
        database.insert_speaker("dup", "Second", make_embedding(seed=1.0))
    # the original row is untouched
    assert database.get_speaker("dup")["name"] == "First"
