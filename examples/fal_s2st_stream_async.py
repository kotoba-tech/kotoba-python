"""Async streaming speech-to-speech translation against the Fal deployment.

Same session API as ``s2st_stream_async.py``; the fal provider adds Key
auth and cold-start session-init retry. The fal S2ST app's WebSocket path
is ``/v1/realtime_voice``.

Usage:
    export FAL_KEY=...
    export KOTOBA_S2ST_EN_JA_URL=wss://fal.run/<team>/<sts-app>/v1/realtime_voice
    uv run examples/fal_s2st_stream_async.py [path/to/clip.mp3] --src en --tgt ja
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import numpy as np

import kotoba
from kotoba.audio import (
    load_mono_pcm16_wav,
    resample_mono_pcm16,
    save_mono_pcm16_wav,
)

# Each cold-start retry is logged with its underlying error.
logging.basicConfig(level=logging.INFO)

DEFAULT_AUDIO = Path(__file__).parent / "audio" / "en" / "example.mp3"

SAMPLE_RATE = 24000
CHUNK_MS = 40
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000


async def main(input_audio: str, src: str, tgt: str, output_wav: str) -> None:
    audio, sr = load_mono_pcm16_wav(input_audio)
    if sr != SAMPLE_RATE:
        audio = resample_mono_pcm16(audio, sr, SAMPLE_RATE)

    # provider="fal" is optional here — a fal.run URL is auto-detected.
    client = kotoba.AsyncKotobaClient(provider="fal")
    transcript_parts: list[str] = []
    audio_chunks: list[bytes] = []

    async with client.s2st.stream(src=src, tgt=tgt) as session:
        for i in range(0, len(audio), CHUNK_SAMPLES):
            chunk = audio[i : i + CHUNK_SAMPLES].astype("<i2").tobytes()
            await session.send_audio(chunk)
            await asyncio.sleep(CHUNK_MS / 1000.0)  # simulate real-time pacing
        await session.commit()

        async for event in session:
            if event.type == "partial_transcript" and event.text:
                transcript_parts.append(event.text)
                print(event.text, end="", flush=True)
            elif event.type == "audio_chunk" and event.audio is not None:
                audio_chunks.append(event.audio)
            elif event.type == "done":
                break

    print()
    out_pcm = b"".join(audio_chunks)
    if out_pcm:
        out = np.frombuffer(out_pcm, dtype="<i2").copy()
        save_mono_pcm16_wav(output_wav, out, SAMPLE_RATE)
        print(f"Wrote {output_wav} ({len(out)} samples @ {SAMPLE_RATE} Hz)")
    else:
        print("(no translated audio received)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_audio", nargs="?", default=str(DEFAULT_AUDIO))
    parser.add_argument("--src", default="en")
    parser.add_argument("--tgt", default="ja")
    parser.add_argument("--output", default="translated_fal.wav")
    args = parser.parse_args()
    asyncio.run(main(args.input_audio, args.src, args.tgt, args.output))
