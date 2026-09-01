"""Async streaming TTS against the Fal deployment.

Identical session API to ``tts_stream_async.py`` — the fal provider only
changes what happens underneath: ``Authorization: Key`` auth, and session
init retried with backoff while a cold app boots (capacity rejections,
worker-not-up connection failures). A cold first connect can block for
minutes, bounded by the retry deadline; enable INFO logging to watch it.

Usage:
    export FAL_KEY=...
    export KOTOBA_TTS_JA_URL=wss://fal.run/<team>/<tts-app>/v2/tts/ws
    uv run examples/fal_tts_stream_async.py
"""

from __future__ import annotations

import asyncio
import logging
import time

import kotoba
from kotoba.audio import save_pcm_f32_as_wav

# Each cold-start retry is logged with its underlying error.
logging.basicConfig(level=logging.INFO)

SAMPLE_TEXT_JA = "こんにちは、フォル上のコトバ音声合成のデモです。"


async def main() -> None:
    # provider="fal" is optional here — a fal.run URL is auto-detected.
    client = kotoba.AsyncKotobaClient(provider="fal")

    t_connect = time.monotonic()
    async with client.tts.stream(language="ja") as session:
        # On a cold app this includes boot + session-init retries.
        print(f"[connect+handshake: {time.monotonic() - t_connect:.1f} s]")

        chunks: list[bytes] = []
        t0 = time.monotonic()
        await session.synthesize(SAMPLE_TEXT_JA)
        async for event in session:
            if event.type == "audio_chunk" and event.audio:
                if not chunks:
                    print(f"[TTFA: {(time.monotonic() - t0) * 1000:.0f} ms]")
                chunks.append(event.audio)
            elif event.type == "done":
                break

    save_pcm_f32_as_wav("out_fal_ja.wav", b"".join(chunks), sample_rate=24000)
    print(f"wrote out_fal_ja.wav ({sum(len(c) for c in chunks)} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
