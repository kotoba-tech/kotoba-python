"""Async bidirectional ASR: audio generator in, transcript deltas out.

Shows ``transcribe_stream(audio_iterable)`` — the feeder and the delta
drain run as concurrent tasks, so the first ``partial_transcript``
surfaces while audio is still being pushed. Swap the
``mic_like_source`` coroutine for a real microphone queue and the
shape is identical.

Usage:
    export KOTOBA_API_KEY=...
    export KOTOBA_ASR_URL=wss://.../asr
    uv run examples/asr_stream_async.py [path/to/clip.mp3]
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import AsyncIterator

import kotoba
from kotoba.audio import load_mono_pcm16_wav, resample_mono_pcm16

DEFAULT_AUDIO = Path(__file__).parent / "audio" / "ja" / "example.mp3"

SAMPLE_RATE = 24000
CHUNK_MS = 200
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000


async def mic_like_source(pcm16: bytes, t0: float) -> AsyncIterator[bytes]:
    """Yield pcm16 chunks at wall-clock pace, mimicking a live mic."""

    chunk_bytes = CHUNK_SAMPLES * 2
    for i, offset in enumerate(range(0, len(pcm16), chunk_bytes)):
        await asyncio.sleep(CHUNK_MS / 1000.0)
        chunk = pcm16[offset : offset + chunk_bytes]
        print(
            f"[{(time.monotonic() - t0) * 1000:7.0f} ms]    mic -> {len(chunk)} bytes (chunk {i})"
        )
        yield chunk


async def main(input_audio: str, language: str) -> None:
    audio, sr = load_mono_pcm16_wav(input_audio)
    if sr != SAMPLE_RATE:
        audio = resample_mono_pcm16(audio, sr, SAMPLE_RATE)
    pcm16 = audio.astype("<i2").tobytes()

    client = kotoba.AsyncKotobaClient()
    t0 = time.monotonic()
    first_text_at: float | None = None
    parts: list[str] = []

    async for delta in client.asr.transcribe_stream(
        mic_like_source(pcm16, t0), language=language
    ):
        now = time.monotonic() - t0
        if first_text_at is None:
            first_text_at = now
            print(
                f"[{now * 1000:7.0f} ms] first transcript "
                f"(latency = {first_text_at * 1000:.0f} ms from start)"
            )
        parts.append(delta)
        print(f"[{now * 1000:7.0f} ms] <- {delta!r}")

    print()
    print("full transcript:", "".join(parts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_audio", nargs="?", default=str(DEFAULT_AUDIO))
    parser.add_argument("--language", default="ja")
    args = parser.parse_args()
    asyncio.run(main(args.input_audio, args.language))
