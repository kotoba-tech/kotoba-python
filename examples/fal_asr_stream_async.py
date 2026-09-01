"""Async streaming ASR against the Fal deployment.

Same session API as ``asr_stream_async.py`` — the fal provider adds Key
auth and cold-start session-init retry (a "No available batch slot"
rejection is retried with backoff until a worker boots). The fal
streaming STT app's WebSocket path is ``/v1/realtime``.

Usage:
    export FAL_KEY=...
    export KOTOBA_ASR_URL=wss://fal.run/<team>/<asr-app>/v1/realtime
    uv run examples/fal_asr_stream_async.py [path/to/clip.mp3]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path
from typing import AsyncIterator

import kotoba
from kotoba.audio import load_mono_pcm16_wav, resample_mono_pcm16

# Each cold-start retry is logged with its underlying error.
logging.basicConfig(level=logging.INFO)

DEFAULT_AUDIO = Path(__file__).parent / "audio" / "ja" / "example.mp3"

SAMPLE_RATE = 24000
CHUNK_MS = 200
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000


async def mic_like_source(pcm16: bytes) -> AsyncIterator[bytes]:
    """Yield pcm16 chunks at wall-clock pace, mimicking a live mic."""
    chunk_bytes = CHUNK_SAMPLES * 2
    for offset in range(0, len(pcm16), chunk_bytes):
        await asyncio.sleep(CHUNK_MS / 1000.0)
        yield pcm16[offset : offset + chunk_bytes]


async def main(input_audio: str, language: str) -> None:
    audio, sr = load_mono_pcm16_wav(input_audio)
    if sr != SAMPLE_RATE:
        audio = resample_mono_pcm16(audio, sr, SAMPLE_RATE)
    pcm16 = audio.astype("<i2").tobytes()

    # provider="fal" is optional here — a fal.run URL is auto-detected.
    client = kotoba.AsyncKotobaClient(provider="fal")

    t_connect = time.monotonic()
    parts: list[str] = []
    # On a cold app the first delta includes boot + session-init retries.
    async for delta in client.asr.transcribe_stream(
        mic_like_source(pcm16), language=language
    ):
        if not parts:
            print(f"[first delta: {time.monotonic() - t_connect:.1f} s]")
        parts.append(delta)
        print(delta, end="", flush=True)

    print(f"\nfull transcript: {''.join(parts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_audio", nargs="?", default=str(DEFAULT_AUDIO))
    parser.add_argument("--language", default="ja")
    args = parser.parse_args()
    asyncio.run(main(args.input_audio, args.language))
