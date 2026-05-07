"""Async streaming TTS fed from an LLM-style async generator.

Shows the ``synthesize_stream(text)`` entry point, which accepts any
``AsyncIterable[str]`` (or sync iterable, or plain str). The feeder and
audio drain run concurrently, so the first audio chunk surfaces as soon
as the server emits it — not after the generator is exhausted. Swap
``fake_llm`` for a real Anthropic / OpenAI streaming call and the shape
is identical.

Usage:
    export KOTOBA_API_KEY=...
    export KOTOBA_TTS_JA_URL=wss://.../tts
    uv run examples/tts_stream_async.py
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import kotoba
from kotoba.audio import save_pcm_f32_as_wav


SAMPLE_TOKENS_JA = [
    "こん", "にち", "は、", "本", "日", "は", "お", "集まり", "いた", "だき",
    "誠に", "あり", "がとう", "ござい", "ます。", "これ", "から", "新しい",
    "音声", "合成", "エン", "ジン", "の", "デモ", "を", "行い", "ます。",
    "どう", "ぞ", "最後", "まで", "お", "楽しみ", "くだ", "さい。",
]


async def fake_llm(tokens: list[str], t0: float) -> AsyncIterator[str]:
    for tok in tokens:
        await asyncio.sleep(0.2)  # mimic LLM token pacing
        print(f"[{(time.monotonic()-t0)*1000:7.0f} ms]    llm -> {tok!r}")
        yield tok


async def main() -> None:
    client = kotoba.AsyncKotobaClient()

    chunks: list[bytes] = []
    t0 = time.monotonic()
    first_audio_at: float | None = None

    async for pcm in client.tts.synthesize_stream(
        fake_llm(SAMPLE_TOKENS_JA, t0), language="ja"
    ):
        now = time.monotonic() - t0
        if first_audio_at is None:
            first_audio_at = now
            print(
                f"[{now*1000:7.0f} ms] first audio "
                f"(latency = {first_audio_at*1000:.0f} ms from start)"
            )
        chunks.append(pcm)
        print(f"[{now*1000:7.0f} ms] <- audio chunk: {len(pcm)} bytes")

    save_pcm_f32_as_wav("out_ja.wav", b"".join(chunks), sample_rate=24000)
    print(
        f"[{(time.monotonic()-t0)*1000:7.0f} ms] wrote out_ja.wav "
        f"({sum(len(c) for c in chunks)} bytes)"
    )


if __name__ == "__main__":
    asyncio.run(main())
