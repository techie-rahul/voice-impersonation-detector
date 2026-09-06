"""WebSocket tests for Phase 4 real-time analysis.

AASIST and Resemblyzer are stubbed via the ``ws_ml`` fixture (which patches the
``app.realtime.run_aasist`` / ``run_speaker_verification`` seams), so no heavy
model is loaded here. The real pipeline is covered once in
``test_realtime_ws_integration.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from starlette.websockets import WebSocketDisconnect

from app import realtime

WS_URL = "/ws/analyze-call/{sid}"
ONE_WINDOW = b"\x00" * realtime.WINDOW_BYTES


@pytest.fixture
def enrolled(isolated_speaker_db) -> str:
    """Insert a speaker row directly (no ML needed)."""
    from app import database

    database.insert_speaker("rahul", "Rahul", np.zeros(256, dtype=np.float32))
    return "rahul"


class _WSStub:
    def __init__(self) -> None:
        self.p_spoof = 0.5
        self.speaker_match = 0.9
        self.aasist_calls = 0
        self.verify_calls = 0

    def run_aasist(self, wav_path: str) -> float:
        self.aasist_calls += 1
        return self.p_spoof

    def run_speaker_verification(self, speaker_id: str, wav_path: str) -> float:
        self.verify_calls += 1
        return self.speaker_match


@pytest.fixture
def ws_ml(monkeypatch) -> _WSStub:
    stub = _WSStub()
    monkeypatch.setattr(realtime, "run_aasist", stub.run_aasist)
    monkeypatch.setattr(realtime, "run_speaker_verification", stub.run_speaker_verification)
    return stub


def start_msg(**overrides) -> dict:
    msg = {
        "type": "start",
        "context": {"transfer": False, "otp": False, "urgent": False, "unknown_caller": False},
        "transaction_amount": 1000,
    }
    msg.update(overrides)
    return msg


# --------------------------------------------------------------------------
# 1 & 3. Connection + valid start for an enrolled speaker
# --------------------------------------------------------------------------

def test_connect_and_start_for_enrolled_speaker(client, enrolled, ws_ml):
    with client.websocket_connect(WS_URL.format(sid=enrolled)) as ws:
        ws.send_json(start_msg(context={"transfer": True}, transaction_amount=5000))
        ack = ws.receive_json()
    assert ack["type"] == "started"
    assert ack["speaker_id"] == "rahul"
    assert ack["window_seconds"] == realtime.WINDOW_SECONDS
    assert ack["sample_rate"] == realtime.SAMPLE_RATE


# --------------------------------------------------------------------------
# 2. Unknown speaker
# --------------------------------------------------------------------------

def test_unknown_speaker_is_rejected(client, isolated_speaker_db, ws_ml):
    with client.websocket_connect(WS_URL.format(sid="ghost")) as ws:
        err = ws.receive_json()
        assert err == {
            "type": "error",
            "code": "SPEAKER_NOT_FOUND",
            "message": "Speaker 'ghost' is not enrolled",
        }
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


# --------------------------------------------------------------------------
# 5 & 6. Binary audio processing + analysis response structure
# --------------------------------------------------------------------------

def test_full_window_produces_analysis(client, enrolled, ws_ml):
    ws_ml.p_spoof = 0.2
    ws_ml.speaker_match = 0.95
    with client.websocket_connect(WS_URL.format(sid=enrolled)) as ws:
        ws.send_json(start_msg())
        assert ws.receive_json()["type"] == "started"

        ws.send_bytes(ONE_WINDOW)
        analysis = ws.receive_json()

    assert set(analysis) == {
        "type", "synthetic_score", "speaker_match",
        "risk_score", "risk_level", "decision", "risk_factors",
    }
    assert analysis["type"] == "analysis"
    assert analysis["synthetic_score"] == pytest.approx(0.2)
    assert analysis["speaker_match"] == pytest.approx(0.95)
    assert 0.0 <= analysis["risk_score"] <= 100.0
    assert analysis["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert analysis["decision"] in {"ALLOW", "VERIFY", "BLOCK"}
    assert ws_ml.aasist_calls == 1 and ws_ml.verify_calls == 1


def test_partial_window_yields_no_analysis(client, enrolled, ws_ml):
    with client.websocket_connect(WS_URL.format(sid=enrolled)) as ws:
        ws.send_json(start_msg())
        ws.receive_json()  # started
        ws.send_bytes(b"\x00" * (realtime.WINDOW_BYTES // 2))
        ws.send_json({"type": "stop"})
        stopped = ws.receive_json()
    assert stopped == {"type": "stopped", "analyses": 0}
    assert ws_ml.aasist_calls == 0


def test_one_buffer_can_yield_multiple_windows(client, enrolled, ws_ml):
    with client.websocket_connect(WS_URL.format(sid=enrolled)) as ws:
        ws.send_json(start_msg())
        ws.receive_json()
        ws.send_bytes(ONE_WINDOW * 2)
        first = ws.receive_json()
        second = ws.receive_json()
    assert first["type"] == "analysis" and second["type"] == "analysis"
    assert ws_ml.aasist_calls == 2


def test_wav_header_is_stripped_from_stream(client, enrolled, ws_ml):
    # 44-byte RIFF/WAVE header + exactly one window of PCM -> still one analysis.
    header = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"fmt " + b"\x00" * 8 + b"data" + b"\x00" * 4
    with client.websocket_connect(WS_URL.format(sid=enrolled)) as ws:
        ws.send_json(start_msg())
        ws.receive_json()
        ws.send_bytes(header + ONE_WINDOW)
        msg = ws.receive_json()
    assert msg["type"] == "analysis"


# --------------------------------------------------------------------------
# 4. Context persistence across windows (same risk_score each window)
# --------------------------------------------------------------------------

def test_context_and_amount_persist_across_windows(client, enrolled, ws_ml):
    ws_ml.p_spoof = 0.3
    ws_ml.speaker_match = 0.6
    with client.websocket_connect(WS_URL.format(sid=enrolled)) as ws:
        ws.send_json(start_msg(context={"transfer": True, "urgent": True}, transaction_amount=150_000))
        ws.receive_json()

        ws.send_bytes(ONE_WINDOW)
        a1 = ws.receive_json()
        ws.send_bytes(ONE_WINDOW)
        a2 = ws.receive_json()

    assert a1["risk_score"] == a2["risk_score"]
    assert a1["risk_factors"] == a2["risk_factors"]
    assert "Money transfer requested" in a1["risk_factors"]
    assert "Urgent request detected" in a1["risk_factors"]
    assert "High-value transaction" in a1["risk_factors"]


def test_ws_uses_same_risk_engine_as_analyze_call(client, enrolled, ws_ml):
    ws_ml.p_spoof = 0.8
    ws_ml.speaker_match = 0.2
    ctx = {"transfer": True, "otp": False, "urgent": True, "unknown_caller": False}

    http = client.post(
        "/analyze-call",
        json={
            "synthetic_score": 0.8,
            "speaker_match": 0.2,
            "context": ctx,
            "transaction_amount": 150_000,
        },
    ).json()

    with client.websocket_connect(WS_URL.format(sid=enrolled)) as ws:
        ws.send_json(start_msg(context=ctx, transaction_amount=150_000))
        ws.receive_json()
        ws.send_bytes(ONE_WINDOW)
        analysis = ws.receive_json()

    assert analysis["risk_score"] == http["risk_score"]
    assert analysis["risk_level"] == http["risk_level"]
    assert analysis["decision"] == http["decision"]
    assert analysis["risk_factors"] == http["risk_factors"]


# --------------------------------------------------------------------------
# 7. High-risk alert
# --------------------------------------------------------------------------

def test_high_risk_emits_alert(client, enrolled, ws_ml):
    ws_ml.p_spoof = 0.97
    ws_ml.speaker_match = 0.05
    with client.websocket_connect(WS_URL.format(sid=enrolled)) as ws:
        ws.send_json(start_msg(
            context={"transfer": True, "otp": True, "urgent": True, "unknown_caller": True},
            transaction_amount=500_000,
        ))
        ws.receive_json()
        ws.send_bytes(ONE_WINDOW)
        analysis = ws.receive_json()
        alert = ws.receive_json()

    assert analysis["type"] == "analysis"
    assert analysis["risk_score"] > 70
    assert alert == {
        "type": "alert",
        "severity": "HIGH",
        "message": "Potential voice impersonation detected",
        "risk_score": analysis["risk_score"],
    }


def test_low_risk_emits_no_alert(client, enrolled, ws_ml):
    ws_ml.p_spoof = 0.01
    ws_ml.speaker_match = 0.99
    with client.websocket_connect(WS_URL.format(sid=enrolled)) as ws:
        ws.send_json(start_msg())
        ws.receive_json()
        ws.send_bytes(ONE_WINDOW)
        analysis = ws.receive_json()
        ws.send_json({"type": "stop"})
        stopped = ws.receive_json()

    assert analysis["risk_score"] <= 70
    assert stopped["type"] == "stopped"  # next message is 'stopped', not 'alert'


# --------------------------------------------------------------------------
# 8. Stop message
# --------------------------------------------------------------------------

def test_stop_message_closes_session_with_count(client, enrolled, ws_ml):
    with client.websocket_connect(WS_URL.format(sid=enrolled)) as ws:
        ws.send_json(start_msg())
        ws.receive_json()
        ws.send_bytes(ONE_WINDOW)
        assert ws.receive_json()["type"] == "analysis"

        ws.send_json({"type": "stop"})
        stopped = ws.receive_json()
        assert stopped == {"type": "stopped", "analyses": 1}
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


# --------------------------------------------------------------------------
# 9. Invalid input
# --------------------------------------------------------------------------

def test_binary_before_start_is_invalid(client, enrolled, ws_ml):
    with client.websocket_connect(WS_URL.format(sid=enrolled)) as ws:
        ws.send_bytes(b"\x00" * 256)
        err = ws.receive_json()
    assert err["type"] == "error"
    assert err["code"] == "INVALID_START"


def test_start_not_json_is_invalid(client, enrolled, ws_ml):
    with client.websocket_connect(WS_URL.format(sid=enrolled)) as ws:
        ws.send_text("not json at all")
        err = ws.receive_json()
    assert err["code"] == "INVALID_START"


@pytest.mark.parametrize("bad_amount", [-1, -0.01, "lots", None, float("nan")])
def test_invalid_transaction_amount(client, enrolled, ws_ml, bad_amount):
    with client.websocket_connect(WS_URL.format(sid=enrolled)) as ws:
        ws.send_json({"type": "start", "context": {}, "transaction_amount": bad_amount})
        err = ws.receive_json()
    assert err["code"] == "INVALID_TRANSACTION_AMOUNT"


@pytest.mark.parametrize("bad_context", ["nope", 5, [1, 2], {"transfer": "banana"}])
def test_invalid_context(client, enrolled, ws_ml, bad_context):
    with client.websocket_connect(WS_URL.format(sid=enrolled)) as ws:
        ws.send_json({"type": "start", "context": bad_context, "transaction_amount": 0})
        err = ws.receive_json()
    assert err["code"] == "INVALID_CONTEXT"


def test_context_accepts_lenient_bools(client, enrolled, ws_ml):
    # pydantic coerces "true"/"yes"/1 -> True; that is acceptable input.
    with client.websocket_connect(WS_URL.format(sid=enrolled)) as ws:
        ws.send_json({"type": "start", "context": {"transfer": "true", "otp": 1}, "transaction_amount": 0})
        assert ws.receive_json()["type"] == "started"


def test_wrong_first_message_type(client, enrolled, ws_ml):
    with client.websocket_connect(WS_URL.format(sid=enrolled)) as ws:
        ws.send_json({"type": "stop"})
        err = ws.receive_json()
    assert err["code"] == "INVALID_START"


def test_empty_audio_frame_is_reported_but_session_continues(client, enrolled, ws_ml):
    with client.websocket_connect(WS_URL.format(sid=enrolled)) as ws:
        ws.send_json(start_msg())
        ws.receive_json()
        ws.send_bytes(b"")
        err = ws.receive_json()
        assert err == {
            "type": "error",
            "code": "EMPTY_AUDIO",
            "message": "Received an empty binary audio frame",
        }
        # session still usable
        ws.send_bytes(ONE_WINDOW)
        assert ws.receive_json()["type"] == "analysis"


def test_unexpected_text_frame_is_reported_but_session_continues(client, enrolled, ws_ml):
    with client.websocket_connect(WS_URL.format(sid=enrolled)) as ws:
        ws.send_json(start_msg())
        ws.receive_json()
        ws.send_json({"type": "whatever"})
        err = ws.receive_json()
        assert err["code"] == "INVALID_MESSAGE"
        ws.send_bytes(ONE_WINDOW)
        assert ws.receive_json()["type"] == "analysis"


def test_inference_error_is_reported_without_traceback(client, enrolled, monkeypatch):
    def boom(_path):
        raise RuntimeError("model exploded\n  File ...")

    monkeypatch.setattr(realtime, "run_aasist", boom)
    monkeypatch.setattr(realtime, "run_speaker_verification", lambda s, p: 0.9)

    with client.websocket_connect(WS_URL.format(sid=enrolled)) as ws:
        ws.send_json(start_msg())
        ws.receive_json()
        ws.send_bytes(ONE_WINDOW)
        err = ws.receive_json()

    assert set(err) == {"type", "code", "message"}
    assert err["code"] == "INFERENCE_ERROR"
    assert "Traceback" not in err["message"] and "File " not in err["message"]


def test_speaker_deleted_mid_session_is_reported(client, enrolled, monkeypatch):
    from app import speaker_service

    def gone(_sid, _path):
        raise speaker_service.SpeakerNotFoundError("rahul")

    monkeypatch.setattr(realtime, "run_aasist", lambda p: 0.5)
    monkeypatch.setattr(realtime, "run_speaker_verification", gone)

    with client.websocket_connect(WS_URL.format(sid=enrolled)) as ws:
        ws.send_json(start_msg())
        ws.receive_json()
        ws.send_bytes(ONE_WINDOW)
        err = ws.receive_json()
    assert err["code"] == "SPEAKER_NOT_FOUND"


# --------------------------------------------------------------------------
# 9. Session isolation -- two clients do not share state
# --------------------------------------------------------------------------

def test_sessions_are_isolated(client, enrolled, ws_ml):
    from app import database

    database.insert_speaker("saanvi", "Saanvi", np.zeros(256, dtype=np.float32))

    with client.websocket_connect(WS_URL.format(sid="rahul")) as ws_a, \
         client.websocket_connect(WS_URL.format(sid="saanvi")) as ws_b:
        ws_a.send_json(start_msg(context={"transfer": True}, transaction_amount=200_000))
        ws_b.send_json(start_msg(context={}, transaction_amount=0))
        assert ws_a.receive_json()["speaker_id"] == "rahul"
        assert ws_b.receive_json()["speaker_id"] == "saanvi"

        # half a window to A only -> A must not analyse (B's bytes are separate)
        ws_a.send_bytes(b"\x00" * (realtime.WINDOW_BYTES // 2))
        ws_b.send_bytes(ONE_WINDOW)
        b_analysis = ws_b.receive_json()
        assert b_analysis["type"] == "analysis"

        ws_a.send_json({"type": "stop"})
        assert ws_a.receive_json() == {"type": "stopped", "analyses": 0}


# --------------------------------------------------------------------------
# 10 & 11. Backward compatibility with Phase 2 / Phase 3 endpoints
# --------------------------------------------------------------------------

def test_http_endpoints_still_work_alongside_ws(client, enrolled):
    assert client.get("/health").json() == {"status": "ok"}

    risk = client.post(
        "/analyze-call",
        json={"synthetic_score": 0.1, "speaker_match": 0.9, "transaction_amount": 0},
    )
    assert risk.status_code == 200
    assert risk.json()["risk_level"] == "LOW"

    speakers = client.get("/speakers").json()["speakers"]
    assert any(s["speaker_id"] == "rahul" for s in speakers)
