"""Sync batch ASR against the Fal deployment.

On the fal provider, ``transcribe()`` is a single synchronous
``POST /v1/speech-to-text`` (no job submit + poll), and a readiness probe
runs first so a cold request is never parked on a booting app.
``warmup()`` makes that wait explicit; it is optional — ``transcribe()``
probes automatically.

Notes vs the kotoba provider: point ``KOTOBA_ASR_REST_URL`` at the app
*root* (no ``/v1`` suffix), and per-segment timestamps are not available.

Usage:
    export FAL_KEY=...
    export KOTOBA_ASR_REST_URL=https://fal.run/<team>/<stt-app>
    uv run examples/fal_asr_rest_sync.py [path/to/clip.mp3]
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import kotoba

DEFAULT_AUDIO = Path(__file__).parent / "audio" / "ja" / "example.mp3"

# Cold-start waits (readiness probes, session retries) log at INFO.
logging.basicConfig(level=logging.INFO)


def main(input_audio: str, language: str) -> None:
    # provider="fal" is optional here — a fal.run URL is auto-detected.
    client = kotoba.KotobaClient(provider="fal")

    t0 = time.monotonic()
    client.warmup()  # blocks until the app is ready (no-op when warm)
    print(f"[warmup: {time.monotonic() - t0:.1f} s]")

    t0 = time.monotonic()
    result = client.asr.transcribe(input_audio, language=language)
    print(f"[transcribe: {time.monotonic() - t0:.1f} s]")
    print(result.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_audio", nargs="?", default=str(DEFAULT_AUDIO))
    parser.add_argument("--language", default="ja")
    args = parser.parse_args()
    main(args.input_audio, args.language)
