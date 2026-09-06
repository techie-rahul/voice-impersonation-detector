#!/usr/bin/env python3
"""Manual smoke client for the Phase 4 real-time WebSocket.

Streams an audio file to ``/ws/analyze-call/{speaker_id}`` as 16 kHz / mono /
16-bit PCM and prints every JSON message the server returns (``started`` /
``analysis`` / ``alert`` / ``error`` / ``stopped``).

Prereqs: the API running (``cd backend && uvicorn app.main:app``) and the
speaker already enrolled (``POST /enroll``).

Examples
--------
    python backend/tests/ws_test_client.py rahul sample.wav
    python backend/tests/ws_test_client.py rahul clip.flac \\
        --url ws://127.0.0.1:8000 --amount 150000 --transfer --urgent
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import librosa
import numpy as np
import websockets

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 1.0
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_SECONDS) * 2  # 16-bit


def load_pcm16(path: str) -> bytes:
    """Load any audio file as 16 kHz mono 16-bit little-endian PCM bytes."""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype("<i2").tobytes()


async def _receiver(ws) -> None:
    try:
        async for raw in ws:
            msg = json.loads(raw)
            kind = msg.get("type", "?").upper()
            print(f"<< {kind}: {json.dumps(msg)}")
            if msg.get("type") in {"stopped", "error"}:
                return
    except websockets.ConnectionClosed:
        print("<< connection closed by server")


async def run(args: argparse.Namespace) -> int:
    pcm = load_pcm16(args.audio)
    duration = len(pcm) / 2 / SAMPLE_RATE
    uri = f"{args.url.rstrip('/')}/ws/analyze-call/{args.speaker_id}"
    print(f"connecting: {uri}")
    print(f"audio: {args.audio}  ({duration:.2f}s -> {len(pcm)} PCM bytes)")

    async with websockets.connect(uri, max_size=None) as ws:
        reader = asyncio.create_task(_receiver(ws))

        start = {
            "type": "start",
            "context": {
                "transfer": args.transfer,
                "otp": args.otp,
                "urgent": args.urgent,
                "unknown_caller": args.unknown_caller,
            },
            "transaction_amount": args.amount,
        }
        print(f">> START: {json.dumps(start)}")
        await ws.send(json.dumps(start))

        for i in range(0, len(pcm), CHUNK_BYTES):
            chunk = pcm[i : i + CHUNK_BYTES]
            await ws.send(chunk)
            print(f">> audio chunk {len(chunk)} bytes")
            await asyncio.sleep(args.realtime_delay)

        print(">> STOP")
        await ws.send(json.dumps({"type": "stop"}))

        try:
            await asyncio.wait_for(reader, timeout=30)
        except asyncio.TimeoutError:
            print("!! timed out waiting for server messages")
            return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("speaker_id")
    p.add_argument("audio", help="WAV/FLAC/... file to stream")
    p.add_argument("--url", default="ws://127.0.0.1:8000", help="base ws:// URL")
    p.add_argument("--amount", type=float, default=0.0, help="transaction_amount (INR)")
    p.add_argument("--transfer", action="store_true")
    p.add_argument("--otp", action="store_true")
    p.add_argument("--urgent", action="store_true")
    p.add_argument("--unknown-caller", dest="unknown_caller", action="store_true")
    p.add_argument("--realtime-delay", type=float, default=0.2,
                   help="seconds to wait between chunks (0 = flush as fast as possible)")
    args = p.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
