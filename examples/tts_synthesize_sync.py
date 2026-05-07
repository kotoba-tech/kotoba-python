"""Sync TTS batch synthesis with explicit speaker_id.

One-shot ``synthesize(text).to_wav(path)`` — the simplest TTS surface.

Usage:
    export KOTOBA_API_KEY=...
    export KOTOBA_TTS_JA_URL=wss://.../tts
    uv run examples/tts_synthesize_sync.py
"""

from __future__ import annotations

import kotoba

client = kotoba.KotobaClient()
result = client.tts.synthesize(
    "こんにちは、世界。今日はいい天気ですね。",
    language="ja",
    speaker_id="ja-man-kei_2",
)
result.to_wav("tts_hello.wav")
print(
    f"wrote tts_hello.wav ({len(result.data)} bytes @ {result.sample_rate} Hz)"
)
