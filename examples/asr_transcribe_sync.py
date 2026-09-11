"""Sync REST ASR example: one-shot ``transcribe()``.

Uploads an audio file in a single synchronous ``POST /v1/speech-to-text``
and prints the transcript (plus per-segment timestamps with ``--timestamps``).
Use this against an ``stt`` deployment (on fal ``kotoba-stt``).
``asr_transcribe_async.py`` is the same with ``AsyncKotobaClient``;
``asr_transcribe_batch_sync.py`` shows ``transcribe(..., batch=True)``, the job
API of self-hosted ``streaming_stt`` deployments, which is not available on fal.

Usage:
    export KOTOBA_API_KEY=...
    export KOTOBA_ASR_REST_URL=https://<host>/v1       # REST base, including the /v1 prefix
    uv run examples/asr_transcribe_sync.py [path/to/clip.mp3] [--timestamps]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import kotoba

DEFAULT_AUDIO = Path(__file__).parent / "audio" / "ja" / "example.mp3"


def main(input_audio: str, language: str | None, with_timestamps: bool) -> None:
    client = kotoba.KotobaClient()
    result = client.asr.transcribe(
        input_audio, language=language, with_timestamps=with_timestamps
    )
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
    main(args.input_audio, args.language, args.timestamps)
