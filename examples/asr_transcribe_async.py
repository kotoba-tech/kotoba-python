"""Async REST ASR example: one-shot ``transcribe()`` with ``AsyncKotobaClient``.

Same request as ``asr_transcribe_sync.py`` (one ``POST /v1/speech-to-text`` to an
``stt`` deployment; on fal ``kotoba-stt``), but uses ``AsyncKotobaClient`` as a
context manager so the underlying HTTP pool is closed on exit.

Usage:
    export KOTOBA_API_KEY=...
    export KOTOBA_ASR_REST_URL=https://<host>/v1       # REST base, including the /v1 prefix
    uv run examples/asr_transcribe_async.py [path/to/clip.mp3] [--timestamps]
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import kotoba

DEFAULT_AUDIO = Path(__file__).parent / "audio" / "ja" / "example.mp3"


async def main(input_audio: str, language: str | None, with_timestamps: bool) -> None:
    async with kotoba.AsyncKotobaClient() as client:
        result = await client.asr.transcribe(input_audio, language=language, with_timestamps=with_timestamps)
    print(result.text)
    for seg in result.segments or []:
        print(f"{seg.start:6.2f} - {seg.end:6.2f}  {seg.text}")
    if duration := result.metadata.get("audio_duration_secs"):
        print(f"({duration:.1f}s of audio)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_audio", nargs="?", default=str(DEFAULT_AUDIO))
    parser.add_argument("--language", default=None, help="defaults to the server's language")
    parser.add_argument("--timestamps", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.input_audio, args.language, args.timestamps))
