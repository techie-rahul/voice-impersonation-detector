"""SQLite persistence for speaker profiles (Phase 3).

Stores only the 256-dimensional speaker embedding plus metadata -- never raw
audio. This module is pure database access: no ML, no FastAPI imports.

The database lives at ``backend/data/speakers.db`` by default; set the
``SPEAKERS_DB_PATH`` environment variable to override it (used by tests).
The schema is created on demand, so callers never have to check first.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

# --- Location ---------------------------------------------------------------
_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "speakers.db"
DB_PATH: Path = Path(os.environ.get("SPEAKERS_DB_PATH", str(_DEFAULT_DB_PATH)))

# --- Embedding serialisation --------------------------------------------
# Embeddings are persisted as raw little-endian float32 bytes in a BLOB column.
EMBEDDING_DTYPE = np.float32
EMBEDDING_DIM = 256

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS speakers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id  TEXT    NOT NULL UNIQUE,
    name        TEXT    NOT NULL,
    embedding   BLOB    NOT NULL,
    created_at  TEXT    NOT NULL
)
"""


class DuplicateSpeakerError(Exception):
    """Raised when inserting a ``speaker_id`` that already exists."""


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    """Yield a connection with the schema ensured; commit/rollback and close."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(_CREATE_TABLE_SQL)  # idempotent, cheap
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Explicitly create the database file and schema. Safe to call repeatedly."""
    with _connection():
        pass


# --- Embedding <-> BLOB ------------------------------------------------

def embedding_to_blob(embedding: np.ndarray) -> bytes:
    """Serialise a numeric embedding to float32 bytes for storage."""
    return np.asarray(embedding, dtype=EMBEDDING_DTYPE).ravel().tobytes()


def blob_to_embedding(blob: bytes) -> np.ndarray:
    """Deserialise stored bytes back into a 1-D float32 numpy array."""
    # np.frombuffer returns a read-only view over the bytes; copy so callers
    # (and scipy) get a writable, independent array.
    return np.frombuffer(blob, dtype=EMBEDDING_DTYPE).copy()


# --- CRUD ----------------------------------------------------------------

def insert_speaker(speaker_id: str, name: str, embedding: np.ndarray) -> dict:
    """Persist a new speaker profile.

    Returns a dict with ``speaker_id``, ``name`` and ``created_at``.
    Raises :class:`DuplicateSpeakerError` if ``speaker_id`` is already taken.
    """
    created_at = datetime.now(timezone.utc).isoformat()
    try:
        with _connection() as conn:
            conn.execute(
                "INSERT INTO speakers (speaker_id, name, embedding, created_at) "
                "VALUES (?, ?, ?, ?)",
                (speaker_id, name, embedding_to_blob(embedding), created_at),
            )
    except sqlite3.IntegrityError as exc:
        raise DuplicateSpeakerError(speaker_id) from exc
    return {"speaker_id": speaker_id, "name": name, "created_at": created_at}


def get_speaker(speaker_id: str) -> Optional[sqlite3.Row]:
    """Return the full row (including the embedding BLOB) or ``None``."""
    with _connection() as conn:
        cur = conn.execute(
            "SELECT id, speaker_id, name, embedding, created_at "
            "FROM speakers WHERE speaker_id = ?",
            (speaker_id,),
        )
        return cur.fetchone()


def list_speakers() -> list[sqlite3.Row]:
    """Return all profiles (no embedding column) ordered by creation time."""
    with _connection() as conn:
        cur = conn.execute(
            "SELECT id, speaker_id, name, created_at "
            "FROM speakers ORDER BY created_at, id"
        )
        return cur.fetchall()


def delete_speaker(speaker_id: str) -> bool:
    """Delete a profile. Returns ``True`` if a row was removed, else ``False``."""
    with _connection() as conn:
        cur = conn.execute(
            "DELETE FROM speakers WHERE speaker_id = ?", (speaker_id,)
        )
        return cur.rowcount > 0
