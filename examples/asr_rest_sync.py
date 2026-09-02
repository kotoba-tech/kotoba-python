"""Sync REST ASR example.

Transcribe an audio file in one synchronous ``POST /v1/speech-to-text``
and print the transcript. REST is the default transport for batch ASR.

Usage:
    export KOTOBA_API_KEY=...
    export KOTOBA_ASR_REST_URL=https://<host>          # server root
    uv run examples/asr_rest_sync.py [path/to/clip.mp3]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import kotoba

DEFAULT_AUDIO = Path(__file__).parent / "audio" / "ja" / "example.mp3"


def main(input_audio: str, language: str) -> None:
    client = kotoba.KotobaClient()
    result = client.asr.transcribe(input_audio, language=language)
    print(result.text)
    if duration := result.metadata.get("audio_duration_secs"):
        print(f"({duration:.1f}s of audio)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_audio", nargs="?", default=str(DEFAULT_AUDIO))
    parser.add_argument("--language", default="ja")
    args = parser.parse_args()
    main(args.input_audio, args.language)
