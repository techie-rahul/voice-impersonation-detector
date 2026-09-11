# Real-Time Voice Analysis WebSocket (Phase 4)

Near-real-time risk scoring of a live call. Audio is streamed in; for every
~4 seconds of audio the server runs AASIST + speaker verification + the Phase 2
risk engine and streams back a result.

No call is terminated and no transaction is blocked here — that is a later
(prevention/UI) phase.

## Connection

Two routes, same protocol:

```
WS /ws/analyze-call                 # Quick Scan  — AASIST synthetic-voice detection only
WS /ws/analyze-call/{speaker_id}    # Identity Protection — AASIST + speaker verification
```

**Identity Protection** — `speaker_id` must already be enrolled via `POST /enroll`
(`ws://<host>/ws/analyze-call/rahul`). If it is not enrolled the server sends a
`SPEAKER_NOT_FOUND` error and closes the socket.

**Quick Scan** — no `speaker_id`, no enrollment. Speaker verification is skipped;
`analysis.speaker_match` is `null`. The risk engine is still run exactly as-is,
using a fixed identity-uncertainty prior (`speaker_match = 0.2`) for the missing
signal, so an unverified synthetic voice can still reach HIGH risk. AASIST and
the risk formula are unchanged.

The `started` reply includes `"mode": "quick-scan" | "identity"` and
`"speaker_id"` (`null` for Quick Scan).

## 1. Start message (client → server, required first message)

Send exactly one JSON text frame before any audio:

```json
{
  "type": "start",
  "context": {
    "transfer": true,
    "otp": false,
    "urgent": true,
    "unknown_caller": false
  },
  "transaction_amount": 150000,
  "mic_mode": true
}
```

* `context` — optional; each flag defaults to `false`. Values are parsed
  leniently (`true`/`"true"`/`1` → `true`).
* `transaction_amount` — INR, must be a finite number `>= 0`; defaults to `0`.
* `mic_mode` — optional bool, default `false`. Set `true` for live-microphone
  input. Documented MVP calibration for mic vs ASVspoof studio audio: the raw
  AASIST `synthetic_score` is still reported unchanged, but for borderline
  scores (`< 0.90`) a fixed `0.15` reduction is applied to the value handed to
  the risk engine, shifting only the ALLOW/VERIFY/BLOCK decision. Silence-trim +
  loudness-normalise is applied to the speaker-verification copy only (feeding it
  to AASIST was measured to *raise* false positives). Not a production-grade fix.

The server replies:

```json
{ "type": "started", "speaker_id": "rahul", "mic_calibrated": true, "window_seconds": 4.0, "sample_rate": 16000 }
```

## 2. Audio format

* PCM **WAV**, **mono**, **16 kHz**, **16-bit** little-endian.
* Send raw PCM samples as **binary** WebSocket frames of any size.
* A leading `RIFF/WAVE` header (i.e. sending a whole `.wav` file) is detected
  and stripped once, so streaming a file's bytes directly also works.
* Invalid audio → a clear JSON error (see below); the server never guesses or
  transcodes.

## 3. Binary audio message (client → server)

Just send the bytes — no envelope:

```
<binary frame: 16-bit PCM samples>
```

Audio is accumulated until a full `window_seconds` (default **4 s** = 128000
bytes) is available. Each completed window triggers **one** AASIST inference and
**one** speaker-verification inference. Partial trailing audio is discarded.

## 4. Analysis response (server → client, one per completed window)

```json
{
  "type": "analysis",
  "synthetic_score": 0.91,
  "speaker_match": 0.68,
  "risk_score": 82.5,
  "risk_level": "HIGH",
  "decision": "BLOCK",
  "risk_factors": ["High synthetic voice probability", "Speaker identity mismatch"]
}
```

* `synthetic_score` — AASIST `p_spoof`, `0..1` (higher = more likely synthetic).
* `speaker_match` — raw Resemblyzer similarity (higher = same speaker), or
  `null` in Quick Scan (no identity verified).
* `risk_score` / `risk_level` / `decision` / `risk_factors` — from the **same**
  `calculate_risk()` used by `POST /analyze-call` (weights/thresholds unchanged).

## 5. Alert response (server → client)

Sent immediately after an `analysis` whenever `risk_score > 70`:

```json
{
  "type": "alert",
  "severity": "HIGH",
  "message": "Potential voice impersonation detected",
  "risk_score": 82.5
}
```

## 6. Stop message (client → server)

```json
{ "type": "stop" }
```

Server replies and closes the socket:

```json
{ "type": "stopped", "analyses": 3 }
```

## 7. Error response (server → client)

```json
{ "type": "error", "code": "SPEAKER_NOT_FOUND", "message": "Speaker 'rahul' is not enrolled" }
```

| `code` | meaning | fatal? |
|---|---|---|
| `SPEAKER_NOT_FOUND` | speaker not enrolled (at connect or mid-session) | yes — socket closes |
| `INVALID_START` | first frame was not a valid JSON `start` message | yes |
| `INVALID_CONTEXT` | `context` is not an object of boolean-ish flags | yes |
| `INVALID_TRANSACTION_AMOUNT` | not a finite number `>= 0` | yes |
| `INVALID_MESSAGE` | unexpected text frame after `start` (not `stop`) | no |
| `EMPTY_AUDIO` | received a zero-length binary frame | no |
| `MALFORMED_AUDIO` | PCM window not 16-bit aligned / unwritable | yes |
| `INFERENCE_ERROR` | AASIST/verification/risk failed for one window | no — that window is skipped |

Stack traces are never sent to the client.

## Manual test client

`backend/tests/ws_test_client.py` streams a file end-to-end:

```
python backend/tests/ws_test_client.py rahul sample.wav --amount 150000 --transfer --urgent
```
