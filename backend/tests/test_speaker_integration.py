"""One integration-style test that exercises the REAL Resemblyzer module.

Enrolls a genuine ASVspoof speaker from one clip and verifies against another
clip from the same speaker (should match) and a different speaker (should score
lower). Skipped automatically when the ASVspoof dataset is not present.

Run just this test with:  pytest -m integration
Skip it with:             pytest -m "not integration"
"""

from __future__ import annotations

import pytest

from tests.conftest import (
    OTHER_SPEAKER_CLIP,
    SAME_SPEAKER_CLIP_A,
    SAME_SPEAKER_CLIP_B,
)

pytestmark = pytest.mark.integration

_CLIPS = [SAME_SPEAKER_CLIP_A, SAME_SPEAKER_CLIP_B, OTHER_SPEAKER_CLIP]


def _post_audio(client, url: str, speaker_id: str, clip):
    with open(clip, "rb") as fh:
        return client.post(
            url,
            data={"speaker_id": speaker_id},
            files={"audio": (clip.name, fh.read(), "audio/flac")},
        )


@pytest.mark.skipif(
    not all(c.exists() for c in _CLIPS),
    reason="ASVspoof2019 LA sample clips not available under ./data",
)
def test_enroll_then_verify_with_real_resemblyzer(client):
    # --- enroll speaker LA_0079 from clip A -------------------------------
    with open(SAME_SPEAKER_CLIP_A, "rb") as fh:
        enrolled = client.post(
            "/enroll",
            data={"speaker_id": "spk_la0079", "name": "LA_0079"},
            files={"audio": (SAME_SPEAKER_CLIP_A.name, fh.read(), "audio/flac")},
        )
    assert enrolled.status_code == 200, enrolled.text

    # --- same speaker, different clip -> high similarity ----------------
    same = _post_audio(client, "/verify-speaker", "spk_la0079", SAME_SPEAKER_CLIP_B)
    assert same.status_code == 200, same.text
    same_body = same.json()

    # --- different speaker -> lower similarity ------------------------
    other = _post_audio(client, "/verify-speaker", "spk_la0079", OTHER_SPEAKER_CLIP)
    assert other.status_code == 200, other.text
    other_body = other.json()

    same_score = same_body["speaker_match"]
    other_score = other_body["speaker_match"]

    # scores are real cosine similarities in a sane range
    assert 0.0 <= other_score <= 1.0
    assert 0.0 <= same_score <= 1.0

    # qualitative expectation: same speaker scores clearly higher
    assert same_score > other_score
    assert same_score >= 0.70

    # verified flag is consistent with the documented MVP threshold
    from app.speaker_service import SPEAKER_MATCH_THRESHOLD

    assert same_body["verified"] is (same_score >= SPEAKER_MATCH_THRESHOLD)
    assert other_body["verified"] is (other_score >= SPEAKER_MATCH_THRESHOLD)
    assert same_body["verified"] is True
