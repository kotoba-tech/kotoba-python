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
import time
from http import HTTPStatus
from typing import AsyncIterator

import pytest
import websockets
from kotoba._ws_asr import AsyncASRSession
from kotoba._ws_s2st import AsyncS2STSession, S2STSession
from kotoba._ws_tts import AsyncTTSSession, TTSSession
from kotoba.errors import AuthError, ProtocolError
from kotoba.tts import AsyncTTSClient, TTSClient, _content_type_for


# The server reads the kana preference from
# input_audio_transcription.style_preference; a flat `kana` key is ignored.
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("style_preference", "expected"),
    [(None, None), ({"human_name": "kana"}, {"human_name": "kana"})],
)
async def test_asr_handshake_style_preference(style_preference, expected):
    session = AsyncASRSession(
        "ws://unused", language="ja", style_preference=style_preference
    )
    sent: list[dict] = []

    async def _send(payload: dict) -> None:
        sent.append(payload)
        session._session_ready.set()

    session._send_json = _send  # type: ignore[method-assign]
    await session._handshake()
    transcription = sent[0]["session"]["input_audio_transcription"]
    assert transcription.get("style_preference") == expected
    assert "kana" not in transcription


async def _capture_tts_open(**kwargs) -> dict:
    session = AsyncTTSSession("ws://unused", language="ja", speaker_id="s", **kwargs)
    sent: list[dict] = []

    async def _send(payload: dict) -> None:
        sent.append(payload)
        session._session_ready.set()

    session._send_json = _send  # type: ignore[method-assign]
    await session._handshake()
    return sent[0]


@pytest.mark.asyncio
async def test_tts_handshake_omits_format_when_unset():
    open_frame = await _capture_tts_open()
    assert "format" not in open_frame
    assert "sample_rate" not in open_frame


@pytest.mark.asyncio
async def test_tts_handshake_sends_negotiated_format():
    open_frame = await _capture_tts_open(audio_format="mulaw", sample_rate=8000)
    assert open_frame["format"] == "mulaw"
    assert open_frame["sample_rate"] == 8000


@pytest.mark.asyncio
async def test_tts_handshake_partial_negotiation():
    open_frame = await _capture_tts_open(audio_format="pcm16")
    assert open_frame["format"] == "pcm16"
    assert "sample_rate" not in open_frame


@pytest.mark.asyncio
async def test_tts_session_created_lowercases_audio_format():
    """The wire format is negotiated case-insensitively; an older server (or
    a proxy) that still echoes the client's raw casing must not leak an
    upper-case string into audio_format, which mirrors the AudioFormat
    Literal's lower-case-only spelling."""
    session = AsyncTTSSession("ws://unused", language="ja", speaker_id="s")
    await session._handle_text_frame(
        {
            "type": "session.created",
            "format": "PCM_F32",
            "sample_rate": 24000,
            "client_id": "fake-client-1",
        }
    )
    assert session.audio_format == "pcm_f32"


@pytest.mark.parametrize(
    ("audio_format", "expected"),
    [
        ("pcm_f32", "audio/pcm;rate=24000;encoding=pcm_f32"),
        ("pcm16", "audio/pcm;rate=8000;encoding=pcm16"),
        ("mulaw", "audio/basic"),
        ("ulaw", "audio/basic"),
        ("twilio", "audio/basic"),
        ("opus", "audio/ogg"),
    ],
)
def test_content_type_for(audio_format, expected):
    rate = 8000 if audio_format == "pcm16" else 24000
    assert _content_type_for(rate, audio_format) == expected


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
            assert msg.get("text"), "server contract: text must be non-empty"
            response_id = msg.get("response_id") or "fake-response-1"
            await websocket.send(
                json.dumps(
                    {"type": "response.created", "response": {"id": response_id}}
                )
            )
            chunk = struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)
            for _ in range(2):
                await websocket.send(
                    json.dumps(
                        {
                            "type": "audio.chunk",
                            "response_id": response_id,
                            "audio": base64.b64encode(chunk).decode("ascii"),
                            "isFinal": False,
                        }
                    )
                )
            await websocket.send(
                json.dumps(
                    {
                        "type": "audio.chunk",
                        "response_id": response_id,
                        "audio": base64.b64encode(chunk).decode("ascii"),
                        "isFinal": True,
                    }
                )
            )
            await websocket.send(
                json.dumps(
                    {
                        "type": "response.done",
                        "response": {"id": response_id, "status": "completed"},
                    }
                )
            )
        elif msg_type == "response.cancel":
            await websocket.send(
                json.dumps(
                    {
                        "type": "response.done",
                        "response": {"id": "fake-response-1", "status": "cancelled"},
                    }
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


def _serve_in_thread(handler, **serve_kwargs) -> tuple[str, callable]:
    loop = asyncio.new_event_loop()
    started = threading.Event()
    state: dict = {}

    async def _main():
        server = await websockets.serve(handler, "127.0.0.1", 0, **serve_kwargs)
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


# ---------- TTS cancel test ------------------------------------------------


@pytest.mark.asyncio
async def test_tts_cancel(tts_server_url):
    """response.cancel surfaces as a `done` event with status='cancelled'."""

    async with AsyncTTSSession(
        tts_server_url, language="ja", speaker_id="ja-man-1"
    ) as session:
        await session.cancel()
        statuses: list[str] = []
        async for event in session:
            if event.type == "done":
                statuses.append(event.metadata.get("status", "completed"))
                break
        assert statuses == ["cancelled"]


# ---------- TTS high-level facade ------------------------------------------


@pytest.mark.asyncio
async def test_tts_synthesize_stream_async(tts_server_url):
    client = AsyncTTSClient(api_key=None)
    chunks: list[bytes] = []
    async for pcm in client.synthesize_stream(
        "hello", language="ja", speaker_id="ja-man-1", url=tts_server_url
    ):
        chunks.append(pcm)
    assert len(chunks) == 3
    assert chunks[0] == struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)


def test_tts_synthesize_stream_sync(tts_server_url):
    client = TTSClient(api_key=None)
    chunks: list[bytes] = []
    for pcm in client.synthesize_stream(
        "hello", language="ja", speaker_id="ja-man-1", url=tts_server_url
    ):
        chunks.append(pcm)
    assert len(chunks) == 3


@pytest.mark.asyncio
async def test_tts_synthesize_batch_async(tts_server_url):
    client = AsyncTTSClient(api_key=None)
    result = await client.synthesize(
        "hello", language="ja", speaker_id="ja-man-1", url=tts_server_url
    )
    # 3 chunks * 4 floats * 4 bytes.
    assert len(result.data) == 3 * 4 * 4
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


# ---------- session-init failure surfacing ---------------------------------
#
# A server (or the fal gateway) can answer session init with an error frame
# and close. That must raise the real error immediately, not a generic
# TimeoutError once handshake_timeout expires.


def _rejecting_tts_server(frame: dict, counters: dict):
    async def handler(websocket):
        counters["connections"] = counters.get("connections", 0) + 1
        await websocket.recv()  # the client's open frame
        await websocket.send(json.dumps(frame))
        await websocket.close()

    return handler


_CAPACITY_ERROR = {"type": "error", "code": "capacity", "message": "No available batch slot"}


async def test_error_frame_during_session_init_surfaces_promptly():
    counters: dict = {}
    url, stop = _serve_in_thread(_rejecting_tts_server(_CAPACITY_ERROR, counters))
    try:
        started = time.monotonic()
        with pytest.raises(ProtocolError, match="batch slot"):
            async with AsyncTTSSession(url, language="ja", speaker_id="ja-man-1"):
                pass
        assert time.monotonic() - started < 5.0  # not the 15s handshake timeout
        assert counters["connections"] == 1
    finally:
        stop()


def test_error_frame_during_session_init_surfaces_promptly_sync():
    counters: dict = {}
    url, stop = _serve_in_thread(_rejecting_tts_server(_CAPACITY_ERROR, counters))
    try:
        started = time.monotonic()
        with pytest.raises(ProtocolError, match="batch slot"):
            with TTSSession(url, language="ja", speaker_id="ja-man-1"):
                pass
        assert time.monotonic() - started < 5.0
    finally:
        stop()


async def test_x_fal_error_frame_surfaces_as_protocol_error():
    frame = {"type": "x-fal-error", "message": "No available batch slot"}
    url, stop = _serve_in_thread(_rejecting_tts_server(frame, {}))
    try:
        with pytest.raises(ProtocolError, match="Fal platform error"):
            async with AsyncTTSSession(url, language="ja", speaker_id="ja-man-1"):
                pass
    finally:
        stop()


async def test_ws_auth_reject_raises_auth_error():
    def reject(connection, request):
        return connection.respond(HTTPStatus.UNAUTHORIZED, "Unauthorized")

    url, stop = _serve_in_thread(_fake_tts_server, process_request=reject)
    try:
        with pytest.raises(AuthError):
            async with AsyncTTSSession(url, language="ja", speaker_id="ja-man-1"):
                pass
    finally:
        stop()


async def test_ws_sends_bearer_auth_for_non_fal_host():
    seen: dict = {}

    def capture(connection, request):
        seen["authorization"] = request.headers.get("Authorization")

    url, stop = _serve_in_thread(_fake_tts_server, process_request=capture)
    try:
        async with AsyncTTSSession(
            url, language="ja", speaker_id="ja-man-1", api_key="secret"
        ) as session:
            await session.cancel()
            async for event in session:
                if event.type == "done":
                    break
        # A local ws:// host is not fal.run, so the scheme is Bearer.
        assert seen["authorization"] == "Bearer secret"
    finally:
        stop()
