"""Live ASR with speaker diarization: who said what, as it is said.

Streams a file at real-time pace (swap ``paced_source`` for a microphone
queue) with ``diarization=True``. Partial transcripts arrive at once with a
provisional speaker; each speaker's turn then ends with a final transcript that
replaces the partials of its ``item_id`` (text and speaker may change) and,
with ``timestamps=True``, carries the turn's span on the audio clock. An empty
final retires a provisional item, and a final can arrive without partials. The
deployment must run a diarization service; ``keywords`` can be passed as well.

Usage:
    export KOTOBA_API_KEY=...
    export KOTOBA_ASR_URL=wss://.../v1/realtime
    uv run examples/asr_stream_diarization_async.py [path/to/two_speakers.wav]
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import AsyncIterator

import kotoba
from kotoba.audio import load_mono_pcm16_wav, resample_mono_pcm16

DEFAULT_AUDIO = Path(__file__).parent / "audio" / "ja" / "example.mp3"
SAMPLE_RATE = 24000
CHUNK_MS = 100


async def paced_source(pcm16: bytes) -> AsyncIterator[bytes]:
    """Yield pcm16 chunks at wall-clock pace, like a live microphone."""

    chunk_bytes = SAMPLE_RATE * CHUNK_MS // 1000 * 2
    for offset in range(0, len(pcm16), chunk_bytes):
        await asyncio.sleep(CHUNK_MS / 1000.0)
        yield pcm16[offset : offset + chunk_bytes]


async def main(input_audio: str, language: str) -> None:
    audio, sr = load_mono_pcm16_wav(input_audio)
    if sr != SAMPLE_RATE:
        audio = resample_mono_pcm16(audio, sr, SAMPLE_RATE)
    pcm16 = audio.astype("<i2").tobytes()

    client = kotoba.AsyncKotobaClient()
    provisional: dict[str, str] = {}  # item_id -> partial text so far
    last = None  # item_id of the last non-blank partial
    try:
        async for event in client.asr.transcribe_stream(
            paced_source(pcm16),
            language=language,
            sample_rate=SAMPLE_RATE,
            diarization=True,
            timestamps=True,
        ):
            item_id = event.metadata["item_id"]
            if event.type == "partial_transcript":
                if not event.text.strip():
                    # The space between words has no speaker and no item_id; it joins the previous partial's item.
                    if last in provisional:
                        provisional[last] += event.text
                    continue
                last = item_id
                provisional[item_id] = provisional.get(item_id, "") + event.text
                # Provisional: the speaker talking when the text was sent.
                print(f"  ~{event.speaker_id}: {provisional[item_id].strip()}")
            elif event.text:
                # Settled: this turn replaces the item's partials, speaker included.
                provisional.pop(item_id, None)
                span = f"{event.metadata.get('start_ms')}-{event.metadata.get('end_ms')} ms"
                print(f"{event.speaker_id} [{span}] {event.text.strip()}")
            elif provisional.pop(item_id, None) is not None:
                print(f"  (retired a provisional item: {item_id})")
    except kotoba.ProtocolError as exc:
        if exc.param != "session.diarization":
            raise
        print(f"diarization unavailable on this deployment: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_audio", nargs="?", default=str(DEFAULT_AUDIO))
    parser.add_argument("--language", default="ja")
    args = parser.parse_args()
    asyncio.run(main(args.input_audio, args.language))
