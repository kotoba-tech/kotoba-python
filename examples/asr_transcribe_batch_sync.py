"""Sync REST ASR example: the job API, ``transcribe(..., batch=True)``.

Self-hosted ``streaming_stt`` deployments only (not available on fal; there use
``transcribe()`` without ``batch``, see ``asr_transcribe_sync.py``).

Submit an audio file via POST + poll, print the transcript and
per-segment timestamps. The job API is served by self-hosted ``streaming_stt`` deployments only;
on fal it is not supported, use ``asr_transcribe_sync.py`` there.

Usage:
    export KOTOBA_API_KEY=...
    export KOTOBA_ASR_REST_URL=https://.../v1
    uv run examples/asr_transcribe_batch_sync.py [path/to/clip.mp3]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import kotoba

DEFAULT_AUDIO = Path(__file__).parent / "audio" / "ja" / "example.mp3"


def main(input_audio: str, language: str) -> None:
    client = kotoba.KotobaClient()
    result = client.asr.transcribe(
        input_audio, batch=True, language=language, with_timestamps=True
    )
    print(result.text)
    for seg in result.segments or []:
        print(f"{seg.start:6.2f} - {seg.end:6.2f}  {seg.text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_audio", nargs="?", default=str(DEFAULT_AUDIO))
    parser.add_argument("--language", default="ja")
    args = parser.parse_args()
    main(args.input_audio, args.language)
