"""Async streaming speech-to-speech translation from a live microphone.

Captures audio from the default input device, feeds 40 ms PCM16 chunks
to the S2ST server, prints the source-language transcript as it
arrives, and writes the translated audio to a WAV file on Ctrl-C.

Requires the ``mic`` optional extra::

    pip install 'kotoba-sdk[mic]'

System prerequisite: PortAudio. On Linux::

    apt install libportaudio2

Usage:
    export KOTOBA_API_KEY=...
    export KOTOBA_S2ST_EN_JA_URL=wss://.../sts
    uv run examples/s2st_mic_async.py
"""

from __future__ import annotations

import argparse
import asyncio

import numpy as np
import sounddevice as sd

import kotoba
from kotoba.audio import save_mono_pcm16_wav

SAMPLE_RATE = 24000
CHUNK_MS = 40
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000


MIC_QUEUE_MAXSIZE = 200  # ~8 s of 40 ms PCM16 chunks; drop on overflow


def mic_chunks(loop: asyncio.AbstractEventLoop) -> tuple[asyncio.Queue, sd.RawInputStream]:
    """Open the default input device and stream PCM16 frames into a queue."""

    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=MIC_QUEUE_MAXSIZE)

    def _enqueue(chunk: bytes) -> None:
        try:
            queue.put_nowait(chunk)
        except asyncio.QueueFull:
            # Network feed is slower than the mic; drop the oldest chunk so
            # the queue stays bounded and recent audio stays fresh.
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                queue.put_nowait(chunk)
            except asyncio.QueueFull:
                pass

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            print(f"[mic status] {status}", flush=True)
        loop.call_soon_threadsafe(_enqueue, bytes(indata))

    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SAMPLES,
        channels=1,
        dtype="int16",
        callback=callback,
    )
    return queue, stream


async def main(src: str, tgt: str, output_wav: str) -> None:
    loop = asyncio.get_running_loop()
    queue, stream = mic_chunks(loop)
    transcript_parts: list[str] = []
    audio_chunks: list[bytes] = []

    print("Listening on default input device. Press Ctrl-C to stop.", flush=True)
    stream.start()

    try:
        client = kotoba.AsyncKotobaClient()
        async with client.s2st.stream(src=src, tgt=tgt) as session:
            feeder = asyncio.create_task(_feed(session, queue))
            try:
                async for event in session:
                    if event.type == "partial_transcript" and event.text:
                        transcript_parts.append(event.text)
                        print(event.text, end="", flush=True)
                    elif event.type == "audio_chunk" and event.audio is not None:
                        audio_chunks.append(event.audio)
                    elif event.type == "done":
                        break
            finally:
                feeder.cancel()
                try:
                    await feeder
                except asyncio.CancelledError:
                    pass
    finally:
        stream.stop()
        stream.close()
        print()
        out_pcm = b"".join(audio_chunks)
        if out_pcm:
            out = np.frombuffer(out_pcm, dtype="<i2").copy()
            save_mono_pcm16_wav(output_wav, out, SAMPLE_RATE)
            print(f"Wrote {output_wav} ({len(out)} samples @ {SAMPLE_RATE} Hz)")
        else:
            print("(no translated audio received)")


async def _feed(session, queue: asyncio.Queue[bytes]) -> None:
    while True:
        chunk = await queue.get()
        await session.send_audio(chunk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="en")
    parser.add_argument("--tgt", default="ja")
    parser.add_argument("--output", default="translated_mic.wav")
    args = parser.parse_args()
    try:
        asyncio.run(main(args.src, args.tgt, args.output))
    except KeyboardInterrupt:
        pass
