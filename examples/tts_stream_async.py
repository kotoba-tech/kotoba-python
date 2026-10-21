"""Async streaming TTS — one-shot text in, audio chunks streamed out.

Sends the known text with ``initial_text`` to overlap the initial synthesis
request with session setup. The timer starts before context entry, so the
reported latency includes DNS, TCP/TLS, WebSocket upgrade and session setup.
It measures first received audio, not playback onset or model-only latency.

Usage:
    export KOTOBA_API_KEY=...
    export KOTOBA_TTS_URL=wss://.../v2/tts/ws
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

    t_connect = time.monotonic()
    async with client.tts.stream(
        language="ja", speaker_id="male-01", initial_text=SAMPLE_TEXT_JA
    ) as session:
        connect_ms = (time.monotonic() - t_connect) * 1000
        print(f"[session ready: {connect_ms:.0f} ms since connection start]")

        chunks: list[bytes] = []
        first_audio_at: float | None = None

        async for event in session:
            if event.type == "audio_chunk" and event.audio:
                now = time.monotonic() - t_connect
                if first_audio_at is None:
                    first_audio_at = now
                    print(f"[{now*1000:7.0f} ms] connection-start to first audio = {first_audio_at*1000:.0f} ms")
                chunks.append(event.audio)
                print(f"[{now*1000:7.0f} ms] <- audio chunk: {len(event.audio)} bytes")

    save_pcm_f32_as_wav("out_ja.wav", b"".join(chunks), sample_rate=session.sample_rate)
    print(
        f"[{(time.monotonic()-t_connect)*1000:7.0f} ms] wrote out_ja.wav "
        f"({sum(len(c) for c in chunks)} bytes)"
    )


if __name__ == "__main__":
    asyncio.run(main())
