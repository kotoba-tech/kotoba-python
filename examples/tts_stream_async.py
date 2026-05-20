"""Async streaming TTS — one-shot text in, audio chunks streamed out.

Uses the low-level ``client.tts.stream(...)`` session so the WebSocket
connection and ``open`` → ``session.created`` handshake complete *before*
the timer starts. The reported TTFA is then the true model-side
time-to-first-audio (server inference + one round-trip) rather than a
cold-start metric that includes TCP/TLS setup.

Usage:
    export KOTOBA_API_KEY=...
    export KOTOBA_TTS_JA_URL=wss://.../v2/tts/ws
    uv run examples/tts_stream_async.py
"""

from __future__ import annotations

import asyncio
import time

import kotoba
from kotoba.audio import save_pcm_f32_as_wav


SAMPLE_TEXT_JA = (
    "こんにちは、本日はお集まりいただき誠にありがとうございます。"
    "これから新しい音声合成エンジンのデモを行います。"
    "どうぞ最後までお楽しみください。"
)


async def main() -> None:
    client = kotoba.AsyncKotobaClient()

    # Pre-warm: TCP / TLS / WebSocket upgrade + `open` -> `session.created`
    # all happen inside `__aenter__`. The TTFA timer below starts only
    # after the session is ready, so it isolates server-side latency.
    t_connect = time.monotonic()
    async with client.tts.stream(language="ja") as session:
        connect_ms = (time.monotonic() - t_connect) * 1000
        print(f"[connect+handshake: {connect_ms:.0f} ms]")

        chunks: list[bytes] = []
        first_audio_at: float | None = None
        t0 = time.monotonic()
        await session.synthesize(SAMPLE_TEXT_JA)

        async for event in session:
            if event.type == "audio_chunk" and event.audio:
                now = time.monotonic() - t0
                if first_audio_at is None:
                    first_audio_at = now
                    print(f"[{now*1000:7.0f} ms] TTFA = {first_audio_at*1000:.0f} ms")
                chunks.append(event.audio)
                print(f"[{now*1000:7.0f} ms] <- audio chunk: {len(event.audio)} bytes")
            elif event.type == "done":
                break

    save_pcm_f32_as_wav("out_ja.wav", b"".join(chunks), sample_rate=24000)
    print(
        f"[{(time.monotonic()-t0)*1000:7.0f} ms] wrote out_ja.wav "
        f"({sum(len(c) for c in chunks)} bytes)"
    )


if __name__ == "__main__":
    asyncio.run(main())
