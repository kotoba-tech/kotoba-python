"""Async bidirectional ASR: audio generator in, transcript deltas out.

Uses the low-level ``client.asr.stream(...)`` session so the WebSocket
connection and ``open`` → ``session.created`` handshake complete *before*
the timer starts. The reported "first delta" latency is then the true
model-side delay (server processing + one round-trip from the first
audio chunk sent), not a cold-start metric.

Swap the ``mic_like_source`` coroutine for a real microphone queue and
the shape is identical.

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


async def mic_like_source(
    pcm16: bytes, t0_holder: dict[str, float]
) -> AsyncIterator[bytes]:
    """Yield pcm16 chunks at wall-clock pace, mimicking a live mic.

    Records the wall-clock time of the *first* yielded chunk in
    ``t0_holder['first_chunk_sent_at']`` so the caller can compute the
    true first-delta latency relative to "first byte sent."
    """

    chunk_bytes = CHUNK_SAMPLES * 2
    for i, offset in enumerate(range(0, len(pcm16), chunk_bytes)):
        await asyncio.sleep(CHUNK_MS / 1000.0)
        chunk = pcm16[offset : offset + chunk_bytes]
        if "first_chunk_sent_at" not in t0_holder:
            t0_holder["first_chunk_sent_at"] = time.monotonic()
        ms_since_send = (time.monotonic() - t0_holder["first_chunk_sent_at"]) * 1000
        print(f"[{ms_since_send:7.0f} ms]    mic -> {len(chunk)} bytes (chunk {i})")
        yield chunk


async def main(input_audio: str, language: str) -> None:
    audio, sr = load_mono_pcm16_wav(input_audio)
    if sr != SAMPLE_RATE:
        audio = resample_mono_pcm16(audio, sr, SAMPLE_RATE)
    pcm16 = audio.astype("<i2").tobytes()

    client = kotoba.AsyncKotobaClient()

    t_connect = time.monotonic()
    async with client.asr.stream(language=language) as session:
        connect_ms = (time.monotonic() - t_connect) * 1000
        print(f"[connect+handshake: {connect_ms:.0f} ms]")

        t0_holder: dict[str, float] = {}
        feeder = asyncio.create_task(session.feed(mic_like_source(pcm16, t0_holder)))

        first_text_at: float | None = None
        parts: list[str] = []
        try:
            async for event in session:
                if event.type == "partial_transcript" and event.text:
                    if first_text_at is None and "first_chunk_sent_at" in t0_holder:
                        first_text_at = time.monotonic() - t0_holder["first_chunk_sent_at"]
                        print(
                            f"[{first_text_at*1000:7.0f} ms] first transcript "
                            f"(TTFB from first audio sent = {first_text_at*1000:.0f} ms)"
                        )
                    parts.append(event.text)
                    elapsed = time.monotonic() - t0_holder.get(
                        "first_chunk_sent_at", time.monotonic()
                    )
                    print(f"[{elapsed*1000:7.0f} ms] <- {event.text!r}")
                elif event.type == "final_transcript" and event.text:
                    parts = [event.text]
                elif event.type == "done":
                    break
            await feeder
        finally:
            if not feeder.done():
                feeder.cancel()

    print()
    print("full transcript:", "".join(parts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_audio", nargs="?", default=str(DEFAULT_AUDIO))
    parser.add_argument("--language", default="ja")
    args = parser.parse_args()
    asyncio.run(main(args.input_audio, args.language))
