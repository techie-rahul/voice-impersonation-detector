# Mic-mode calibration: real-data findings (2026-09-11)

Investigated whether `MIC_MODE_THRESHOLD_OFFSET` / `MIC_OFFSET_PSPOOF_CEILING`
in `backend/app/realtime.py` could be recalibrated from real genuine-voice
`p_spoof` distributions. Conclusion: **no defensible offset/ceiling pair was
found**; this looks like an AASIST domain-mismatch issue, not something two
constants can absorb.

## Round 1 — WhatsApp export samples (rejected as non-representative)

21 genuine WhatsApp voice notes (Opus-compressed, resampled to 16kHz mono WAV)
scored through `detect_synthetic()`:

* min 0.0168, max 1.0000, median 0.9993, mean 0.9093, stdev 0.2676
* 18/21 (86%) scored `p_spoof > 0.93`, most at 0.999+
* 3 outliers (IQR method) scored low: 0.0168, 0.2042, 0.9371

Rejected: WhatsApp audio goes through Opus compression + resampling, which the
live demo's browser mic path does not. Not representative of what
`MIC_MODE_THRESHOLD_OFFSET` actually needs to calibrate against.

## Round 2 — real live-mic capture (representative)

Added an opt-in debug hook to `analyze_window()` in `realtime.py`
(`RT_DEBUG_CAPTURE_DIR` env var, no-op unless set) that copies each raw 4s
window WAV — the exact bytes the browser's `startMicCapture()` /
`realtimeSocket.ts` pipeline sends over the WebSocket — before the temp file
is deleted. Captured 17 windows (~70s) from a real Live Analysis mic session
through the actual frontend, then scored them the same way:

* min 0.2746, max 1.0000, median 0.9994, mean 0.9304, stdev 0.1983
* 15/17 (88%) scored `p_spoof >= 0.987`, most at 0.999+
* 2 outliers (IQR method): 0.2746 and 0.5696 (plausibly leading/trailing
  windows with partial silence or partial speech)

This reproduces the same saturation pattern as the WhatsApp data, on a clean,
uncompressed signal — ruling out Opus/resampling as the cause.

## Why no offset/ceiling change is proposed

The current `MIC_OFFSET_PSPOOF_CEILING = 0.90` means `MIC_MODE_THRESHOLD_OFFSET`
never applies to this cluster — genuine mic windows at 0.99+ pass through to
the risk engine essentially unadjusted. Raising the ceiling enough to cover the
observed genuine-speech cluster (into the high 0.99x range) would leave little
to no score-space separating genuine live speech from an actual synthetic
clone, since a real clone would also be expected to score near the ceiling.
Widening the discount that far risks masking real detections rather than
fixing false positives on genuine speech.

**Conclusion:** this is very likely an AASIST model/domain-mismatch problem —
trained on ASVspoof-style studio recordings, not raw laptop/browser mic
capture — rather than a decision-threshold calibration problem. A fix would
need to address the detector/preprocessing itself (e.g. a mic-domain
fine-tune, alternative preprocessing before AASIST, or a different detector
for live-mic mode), not `MIC_MODE_THRESHOLD_OFFSET` /
`MIC_OFFSET_PSPOOF_CEILING`.

## Reproducing

```bash
RT_DEBUG_CAPTURE_DIR=/tmp/mic_capture uvicorn app.main:app  # from backend/
# run a real Live Analysis mic session in the frontend for 40-80s, then stop
# each ~4s window lands in /tmp/mic_capture/ as mic_<timestamp>.wav
```

Score the captures with `detect_synthetic()` from `detector.py` (unmodified)
and compute min/max/median/mean/stdev + an IQR outlier check, same as above.
