"""End-to-end protocol tests against a fake WebSocket server.

We spin up `websockets.serve(...)` on an ephemeral port and run a small
state machine that mimics the production TTS / S2ST servers' frame
sequences. This validates the session classes against real frames
without touching the network.
"""

from __future__ import annotations

import asyncio
import base64
import json
import struct
from typing import AsyncIterator

import pytest
import websockets

from kotoba._ws_s2st import AsyncS2STSession, S2STSession
from kotoba._ws_tts import AsyncTTSSession, TTSSession
from kotoba.tts import AsyncTTSClient, TTSClient


# ---------- TTS fake server -----------------------------------------------


async def _fake_tts_server(websocket):
    open_msg = json.loads(await websocket.recv())
    assert open_msg["type"] == "open"
    await websocket.send(
        json.dumps(
            {
                "type": "session.created",
                "format": "pcm_f32",
                "sample_rate": 24000,
                "channels": 1,
                "language": open_msg["language"],
                "speaker_id": open_msg["speaker_id"],
                "client_id": "fake-client-1",
            }
        )
    )

    while True:
        try:
            raw = await websocket.recv()
        except websockets.exceptions.ConnectionClosed:
            return
        msg = json.loads(raw)
        msg_type = msg["type"]

        if msg_type == "response.create":
            await websocket.send(json.dumps({"type": "response.created"}))
        elif msg_type == "input_text_buffer.append":
            # Fake: nothing to do until commit.
            pass
        elif msg_type == "input_text_buffer.commit":
            # Emit two non-final chunks then a final chunk.
            chunk = struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)
            for _ in range(2):
                await websocket.send(
                    json.dumps(
                        {
                            "type": "audio.chunk",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                            "isFinal": False,
                        }
                    )
                )
            await websocket.send(
                json.dumps(
                    {
                        "type": "audio.chunk",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                        "isFinal": True,
                    }
                )
            )
            await websocket.send(
                json.dumps(
                    {"type": "response.done", "response": {"status": "completed"}}
                )
            )


# ---------- S2ST fake server ----------------------------------------------


async def _fake_s2st_server(websocket):
    update = json.loads(await websocket.recv())
    assert update["type"] == "voice_session.update"

    await websocket.send(json.dumps({"type": "voice_session.created"}))
    await websocket.send(json.dumps({"type": "voice_session.updated"}))

    chunks_received = 0
    while True:
        try:
            raw = await websocket.recv()
        except websockets.exceptions.ConnectionClosed:
            return
        msg = json.loads(raw)
        msg_type = msg["type"]

        if msg_type == "input_audio_buffer.append":
            chunks_received += 1
            if chunks_received == 1:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "conversation.item.input_audio_transcription.text.delta",
                            "delta": "hello ",
                        }
                    )
                )
        elif msg_type == "input_audio_buffer.commit":
            await websocket.send(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.text.delta",
                        "delta": "world",
                    }
                )
            )
            audio = (b"\x10\x00" * 10)  # 20 bytes pcm16
            await websocket.send(
                json.dumps(
                    {
                        "type": "conversation.item.input_audio_transcription.audio.delta",
                        "delta": base64.b64encode(audio).decode("ascii"),
                    }
                )
            )
            await websocket.send(json.dumps({"type": "input_audio_buffer.committed"}))
            return


# ---------- fixtures -------------------------------------------------------
#
# We run the fake server on a daemon-thread event loop so the same fixture
# can be consumed by both sync tests and pytest-asyncio tests. (An async
# fixture on the test's own loop wouldn't be reachable from the sync
# session's background loop.)


import threading
from typing import Iterator


def _serve_in_thread(handler) -> tuple[str, callable]:
    loop = asyncio.new_event_loop()
    started = threading.Event()
    state: dict = {}

    async def _main():
        server = await websockets.serve(handler, "127.0.0.1", 0)
        host, port = server.sockets[0].getsockname()[:2]
        state["url"] = f"ws://{host}:{port}"
        state["server"] = server
        started.set()
        await server.wait_closed()

    def _runner():
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_main())
        finally:
            loop.close()

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    started.wait(timeout=5.0)

    def stop():
        loop.call_soon_threadsafe(state["server"].close)
        thread.join(timeout=2.0)

    return state["url"], stop


@pytest.fixture
def tts_server_url() -> Iterator[str]:
    url, stop = _serve_in_thread(_fake_tts_server)
    try:
        yield url
    finally:
        stop()


@pytest.fixture
def s2st_server_url() -> Iterator[str]:
    url, stop = _serve_in_thread(_fake_s2st_server)
    try:
        yield url
    finally:
        stop()


# ---------- TTS tests ------------------------------------------------------


@pytest.mark.asyncio
async def test_tts_async_streaming(tts_server_url):
    chunks: list[bytes] = []
    saw_done = False
    async with AsyncTTSSession(
        tts_server_url, language="ja", speaker_id="ja-man-1"
    ) as session:
        assert session.sample_rate == 24000
        assert session.audio_format == "pcm_f32"
        await session.synthesize("hello")
        async for event in session:
            if event.type == "audio_chunk":
                if event.audio:
                    chunks.append(event.audio)
            elif event.type == "done":
                saw_done = True
                break

    assert saw_done
    assert len(chunks) == 3
    assert chunks[0] == struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)


# A second fake TTS server that emits one audio chunk per
# `input_text_buffer.append` frame — lets tests observe the feeder and
# receiver running concurrently (first audio chunk arrives while later
# appends are still in flight).


async def _fake_tts_per_append_server(websocket):
    open_msg = json.loads(await websocket.recv())
    assert open_msg["type"] == "open"
    await websocket.send(
        json.dumps(
            {
                "type": "session.created",
                "format": "pcm_f32",
                "sample_rate": 24000,
                "channels": 1,
                "language": open_msg["language"],
                "speaker_id": open_msg["speaker_id"],
                "client_id": "fake-client-per-append",
            }
        )
    )

    append_count = 0
    while True:
        try:
            raw = await websocket.recv()
        except websockets.exceptions.ConnectionClosed:
            return
        msg = json.loads(raw)
        msg_type = msg["type"]

        if msg_type == "response.create":
            append_count = 0
            await websocket.send(json.dumps({"type": "response.created"}))
        elif msg_type == "input_text_buffer.append":
            append_count += 1
            # Encode the append index into the audio payload so tests can
            # assert which-chunk-for-which-token ordering.
            chunk = struct.pack("<f", float(append_count))
            await websocket.send(
                json.dumps(
                    {
                        "type": "audio.chunk",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                        "isFinal": False,
                    }
                )
            )
        elif msg_type == "input_text_buffer.commit":
            final_chunk = struct.pack("<f", 0.0)
            await websocket.send(
                json.dumps(
                    {
                        "type": "audio.chunk",
                        "audio": base64.b64encode(final_chunk).decode("ascii"),
                        "isFinal": True,
                    }
                )
            )
            await websocket.send(
                json.dumps(
                    {"type": "response.done", "response": {"status": "completed"}}
                )
            )


@pytest.fixture
def tts_per_append_url() -> Iterator[str]:
    url, stop = _serve_in_thread(_fake_tts_per_append_server)
    try:
        yield url
    finally:
        stop()


def test_tts_sync_streaming(tts_server_url):
    chunks: list[bytes] = []
    with TTSSession(tts_server_url, language="ja", speaker_id="ja-man-1") as session:
        session.synthesize("hello")
        for event in session:
            if event.type == "audio_chunk" and event.audio:
                chunks.append(event.audio)
            if event.type == "done":
                break
    assert len(chunks) == 3


# ---------- S2ST tests -----------------------------------------------------


@pytest.mark.asyncio
async def test_s2st_async_streaming(s2st_server_url):
    transcripts: list[str] = []
    audio_chunks: list[bytes] = []

    async with AsyncS2STSession(
        s2st_server_url, src_language="en", tgt_language="ja"
    ) as session:
        await session.send_audio(b"\x00\x00" * 100)
        await session.send_audio(b"\x00\x00" * 100)
        await session.commit()
        async for event in session:
            if event.type == "partial_transcript" and event.text:
                transcripts.append(event.text)
            elif event.type == "audio_chunk" and event.audio:
                audio_chunks.append(event.audio)
            elif event.type == "done":
                break

    assert "".join(transcripts) == "hello world"
    assert audio_chunks and len(audio_chunks[0]) == 20


# ---------- TTS generator-input tests --------------------------------------


@pytest.mark.asyncio
async def test_tts_synthesize_stream_async_generator(tts_per_append_url):
    async def gen():
        for tok in ["a", "b", "c", "d"]:
            yield tok

    client = AsyncTTSClient(api_key=None)
    chunks: list[bytes] = []
    async for pcm in client.synthesize_stream(
        gen(), language="ja", speaker_id="ja-man-1", url=tts_per_append_url
    ):
        chunks.append(pcm)

    # 4 append chunks (one per token) + 1 final chunk from commit.
    assert len(chunks) == 5
    indices = [struct.unpack("<f", c)[0] for c in chunks]
    assert indices == [1.0, 2.0, 3.0, 4.0, 0.0]


def test_tts_synthesize_stream_sync_generator(tts_per_append_url):
    def gen():
        for tok in ["a", "b", "c", "d"]:
            yield tok

    client = TTSClient(api_key=None)
    chunks: list[bytes] = []
    for pcm in client.synthesize_stream(
        gen(), language="ja", speaker_id="ja-man-1", url=tts_per_append_url
    ):
        chunks.append(pcm)

    assert len(chunks) == 5
    indices = [struct.unpack("<f", c)[0] for c in chunks]
    assert indices == [1.0, 2.0, 3.0, 4.0, 0.0]


@pytest.mark.asyncio
async def test_tts_synthesize_stream_interleaved(tts_per_append_url):
    """Feeder and drainer run concurrently — the first audio chunk must
    surface before the generator has yielded its last token."""

    first_audio = asyncio.Event()
    tokens_yielded: list[str] = []

    async def gen():
        for i, tok in enumerate(["a", "b", "c", "d"]):
            if i >= 1:
                # Wait for the first audio chunk to surface before
                # yielding the second token. If the implementation drained
                # only after commit, this would deadlock (no audio until
                # after gen() finishes).
                await asyncio.wait_for(first_audio.wait(), timeout=2.0)
            tokens_yielded.append(tok)
            yield tok

    client = AsyncTTSClient(api_key=None)
    chunks: list[bytes] = []
    async for pcm in client.synthesize_stream(
        gen(), language="ja", speaker_id="ja-man-1", url=tts_per_append_url
    ):
        if not first_audio.is_set():
            first_audio.set()
        chunks.append(pcm)

    assert len(chunks) == 5
    assert tokens_yielded == ["a", "b", "c", "d"]


@pytest.mark.asyncio
async def test_tts_synthesize_batch_accepts_generator(tts_per_append_url):
    async def gen():
        for tok in ["x", "y", "z"]:
            yield tok

    client = AsyncTTSClient(api_key=None)
    result = await client.synthesize(
        gen(), language="ja", speaker_id="ja-man-1", url=tts_per_append_url
    )
    # 3 append chunks + 1 final chunk, each 4 bytes (float32).
    assert len(result.data) == 4 * 4
    assert result.sample_rate == 24000


def test_s2st_sync_streaming(s2st_server_url):
    transcripts: list[str] = []
    with S2STSession(s2st_server_url, src_language="en", tgt_language="ja") as session:
        session.send_audio(b"\x00\x00" * 100)
        session.send_audio(b"\x00\x00" * 100)
        session.commit()
        for event in session:
            if event.type == "partial_transcript" and event.text:
                transcripts.append(event.text)
            if event.type == "done":
                break
    assert "".join(transcripts) == "hello world"
