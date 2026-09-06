"""One real end-to-end WebSocket test: real enrollment, real AASIST, real
Resemblyzer, real risk engine. Skipped when the ASVspoof clips are absent.

Run only this:   pytest -m integration
Skip it:         pytest -m "not integration"
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from starlette.websockets import WebSocketDisconnect

from tests.conftest import SAME_SPEAKER_CLIP_A, SAME_SPEAKER_CLIP_B

pytestmark = pytest.mark.integration

_CLIPS = [SAME_SPEAKER_CLIP_A, SAME_SPEAKER_CLIP_B]


def _pcm16_mono_16k(path) -> bytes:
    audio, sr = sf.read(str(path))
    assert sr == 16000 and audio.ndim == 1, "expected 16 kHz mono clip"
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


@pytest.mark.skipif(
    not all(c.exists() for c in _CLIPS),
    reason="ASVspoof2019 LA sample clips not available under ./data",
)
def test_realtime_ws_full_pipeline(client):
    from app import speaker_service
    from app.risk_engine import calculate_risk

    # real enrollment from clip A (speaker LA_0079)
    speaker_service.create_speaker("spk_la0079", "LA_0079", str(SAME_SPEAKER_CLIP_A))

    pcm = _pcm16_mono_16k(SAME_SPEAKER_CLIP_B)  # ~4.4 s -> one full 4 s window
    context = {"transfer": True, "otp": False, "urgent": False, "unknown_caller": False}
    amount = 250_000

    with client.websocket_connect("/ws/analyze-call/spk_la0079") as ws:
        ws.send_json({"type": "start", "context": context, "transaction_amount": amount})
        assert ws.receive_json()["type"] == "started"

        half = len(pcm) // 2
        ws.send_bytes(pcm[:half])
        ws.send_bytes(pcm[half:])

        analysis = ws.receive_json()
        ws.send_json({"type": "stop"})
        stopped = ws.receive_json()
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()

    assert analysis["type"] == "analysis"
    assert stopped == {"type": "stopped", "analyses": 1}

    # real AASIST probability in range
    assert 0.0 <= analysis["synthetic_score"] <= 1.0
    # same speaker -> similarity clearly positive (measured ~0.83 for this pair)
    assert 0.0 <= analysis["speaker_match"] <= 1.0
    assert analysis["speaker_match"] > 0.5

    # the WS result must equal a direct calculate_risk() call on the same inputs
    expected = calculate_risk(
        synthetic_score=analysis["synthetic_score"],
        speaker_match=analysis["speaker_match"],
        context=context,
        transaction_amount=amount,
    )
    assert analysis["risk_score"] == expected.risk_score
    assert analysis["risk_level"] == expected.risk_level
    assert analysis["decision"] == expected.decision
    assert analysis["risk_factors"] == expected.risk_factors
