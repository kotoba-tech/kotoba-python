"""Streaming ASR diarization: request serialization, speaker ids on recorded staging
frames, transcribe_stream over a local WebSocket replay, and the typed rejection."""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
import threading
import time
import typing
from pathlib import Path
from typing import Literal

import pytest
import websockets
from kotoba import (
    ASRClient,
    AsyncASRClient,
    ProtocolError,
    SemanticVAD,
    StreamEvent,
)
from kotoba._ws_asr import ASRSession, AsyncASRSession

RECORDED = json.loads((Path(__file__).parent / "data" / "asr_stream_diarization_frames.json").read_text())


def _update(factory, **options) -> dict:
    session = factory().stream(url="ws://unused", language="ja", **options)
    if isinstance(session, ASRSession):
        session = session._asr
    return session._session_update()["session"]


@pytest.mark.parametrize("factory", [ASRClient, AsyncASRClient], ids=["sync", "async"])
@pytest.mark.parametrize(("options", "diarization", "turn_detection"), [
    pytest.param({}, None, None, id="off-omits-both"),
    pytest.param({"diarization": True}, True, None, id="on-leaves-turn-detection-to-the-server"),
    pytest.param({"diarization": True, "turn_detection": False}, True, False, id="on-provisional-only"),
    pytest.param(
        {"diarization": True, "turn_detection": SemanticVAD(eagerness="high")},
        True, {"type": "semantic_vad", "eagerness": "high"}, id="on-explicit-turn-detection",
    ),
])
def test_session_update(factory, options, diarization, turn_detection):
    session = _update(factory, **options)
    assert session.get("diarization") is diarization
    assert session.get("turn_detection", None) == turn_detection
    if turn_detection is None:
        assert "turn_detection" not in session


@pytest.mark.parametrize("factory", [ASRClient, AsyncASRClient], ids=["sync", "async"])
def test_diarization_and_keywords_are_sent_together(factory):
    session = _update(factory, diarization=True, keywords=["コトバ", "  "])
    assert session["diarization"] is True
    assert session["input_audio_transcription"]["keywords"] == ["コトバ"]


@pytest.mark.skipif(sys.version_info < (3, 11), reason="typing.get_overloads needs Python 3.11")
@pytest.mark.parametrize("client", [ASRClient, AsyncASRClient], ids=["sync", "async"])
def test_the_diarization_overload_of_transcribe_stream_accepts_keywords(client):
    # Type checkers resolve transcribe_stream(diarization=True, keywords=...) through this overload.
    diarized = [
        overload for overload in typing.get_overloads(client.transcribe_stream)
        if typing.get_type_hints(overload)["diarization"] == Literal[True]
    ]
    assert len(diarized) == 1
    assert "keywords" in inspect.signature(diarized[0]).parameters


@pytest.mark.skipif(sys.version_info < (3, 11), reason="typing.get_overloads needs Python 3.11")
@pytest.mark.parametrize("client", [ASRClient, AsyncASRClient], ids=["sync", "async"])
def test_transcribe_stream_has_an_overload_for_a_bool_diarization_flag(client):
    # transcribe_stream(audio, diarization=flag) with flag: bool matches neither literal overload.
    hints = [typing.get_type_hints(overload)["diarization"] for overload in typing.get_overloads(client.transcribe_stream)]
    assert hints == [Literal[False], Literal[True], bool]


async def _decode(frames: list[dict]) -> list[StreamEvent]:
    session = AsyncASRSession("ws://unused", language="ja")
    for frame in frames:
        await session._handle_text_frame(frame)
    return [event async for event in session]


@pytest.mark.asyncio
async def test_recorded_diarized_frames_expose_provisional_and_settled_speakers():
    events = await _decode(RECORDED["diarization_on"]["frames"])
    partials = [e for e in events if e.type == "partial_transcript"]
    finals = [e for e in events if e.type == "final_transcript"]
    assert events[0].type == "session_ready"
    assert events[0].metadata["session"]["turn_detection"] == {"type": "semantic_vad", "eagerness": "medium"}
    assert partials and finals
    assert all(e.speaker_id and e.speaker_id == e.metadata["speaker_id"] for e in partials + finals)
    assert {e.speaker_id for e in finals} == {"speaker_1", "speaker_2"}
    assert all(isinstance(e.metadata["start_ms"], int) and e.metadata["end_ms"] >= e.metadata["start_ms"] for e in finals)
    # A turn whose first deltas were credited to the speaker talking at that moment
    # and whose completion settled on another one (the recorded staging session has one).
    settled = {e.metadata["item_id"]: e.speaker_id for e in finals}
    changed = {
        e.metadata["item_id"] for e in partials
        if e.metadata["item_id"] in settled and e.speaker_id != settled[e.metadata["item_id"]]
    }
    assert len(changed) == 1
    assert [e.type for e in events[-2:]] == ["committed", "done"]
    assert events.index(finals[-1]) < events.index(events[-2])


@pytest.mark.asyncio
async def test_a_whitespace_only_delta_keeps_its_text_without_a_speaker():
    # The server attributes non-blank deltas only. The recording drops blank deltas, so
    # this inserts the space a session sends between English words after the first delta.
    frames = RECORDED["diarization_on"]["frames"]
    first = next(i for i, f in enumerate(frames) if f["type"].endswith(".delta") and f["delta"].strip())
    space = {key: value for key, value in frames[first].items() if key not in ("speaker_id", "start_ms", "end_ms")}
    space["delta"] = " "
    events = await _decode(frames[: first + 1] + [space] + frames[first + 1 :])
    partials = [e for e in events if e.type == "partial_transcript"]
    blank = [e for e in partials if not e.text.strip()]
    assert [e.text for e in blank] == [" "]
    assert blank[0].speaker_id is None
    assert all(e.speaker_id for e in partials if e.text.strip())


@pytest.mark.asyncio
async def test_recorded_frames_without_diarization_have_no_speaker():
    events = await _decode(RECORDED["diarization_off"]["frames"])
    assert any(e.type == "partial_transcript" for e in events)
    assert all(e.speaker_id is None and "speaker_id" not in e.metadata for e in events)


# ---------- over a local WebSocket server replaying the recordings ----------


def _replay_server(recording: dict, received: list[dict]):
    """Answer like the staging app: created on connect, then the recorded frames.

    A recorded rejection is sent right after the session update; otherwise the
    transcript frames follow the client's commit.
    """
    frames = recording["frames"]

    async def handler(websocket):
        await websocket.send(json.dumps(frames[0]))
        update = json.loads(await websocket.recv())
        received.append(update)
        if frames[1]["type"] == "error":
            await websocket.send(json.dumps(frames[1]))
            await websocket.close()
            return
        await websocket.send(json.dumps(frames[1]))
        while json.loads(await websocket.recv())["type"] != "input_audio_buffer.commit":
            pass
        for frame in frames[2:]:
            await websocket.send(json.dumps(frame))
        await websocket.wait_closed()

    return handler


def _serve_in_thread(handler) -> tuple[str, callable]:
    loop = asyncio.new_event_loop()
    started = threading.Event()
    state: dict = {}

    async def main():
        server = await websockets.serve(handler, "127.0.0.1", 0)
        host, port = server.sockets[0].getsockname()[:2]
        state.update(url=f"ws://{host}:{port}", server=server)
        started.set()
        await server.wait_closed()

    thread = threading.Thread(target=lambda: loop.run_until_complete(main()), daemon=True)
    thread.start()
    started.wait(timeout=5.0)

    def stop():
        loop.call_soon_threadsafe(state["server"].close)
        thread.join(timeout=2.0)

    return state["url"], stop


_SILENCE = [b"\x00\x00" * 3200] * 3


@pytest.mark.parametrize(("recording", "diarization"), [
    pytest.param("diarization_on", True, id="on-yields-events"),
    pytest.param("diarization_off", False, id="off-yields-text"),
])
def test_transcribe_stream_sync(recording, diarization):
    received: list[dict] = []
    url, stop = _serve_in_thread(_replay_server(RECORDED[recording], received))
    try:
        out = list(ASRClient().transcribe_stream(_SILENCE, url=url, diarization=diarization, timestamps=True))
    finally:
        stop()
    assert received[0]["session"].get("diarization") is (True if diarization else None)
    if diarization:
        assert all(isinstance(e, StreamEvent) and e.speaker_id for e in out)
        assert {e.type for e in out} == {"partial_transcript", "final_transcript"}
    else:
        assert out and all(isinstance(text, str) for text in out)


@pytest.mark.asyncio
async def test_transcribe_stream_async_yields_events_with_speakers():
    received: list[dict] = []
    url, stop = _serve_in_thread(_replay_server(RECORDED["diarization_on"], received))
    try:
        out = [e async for e in AsyncASRClient().transcribe_stream(_SILENCE, url=url, diarization=True)]
    finally:
        stop()
    assert received[0]["session"]["diarization"] is True
    finals = [e for e in out if e.type == "final_transcript"]
    assert len(finals) == sum(f["type"].endswith(".completed") for f in RECORDED["diarization_on"]["frames"])
    assert all(e.speaker_id for e in out)


def _assert_rejection(exc: ProtocolError) -> None:
    assert (exc.code, exc.param) == ("invalid_parameters", "session.diarization")
    assert "retry without diarization" in str(exc)
    assert exc.payload["error"]["param"] == "session.diarization"


@pytest.mark.asyncio
async def test_server_rejection_raises_a_protocol_error_with_the_param_async():
    # The staging rejection from a deployment without a diarization service; a model
    # without a timestamp calibration answers with the same code and param.
    url, stop = _serve_in_thread(_replay_server(RECORDED["diarization_unavailable"], []))
    try:
        started = time.monotonic()
        with pytest.raises(ProtocolError) as caught:
            async with AsyncASRClient().stream(url=url, language="ja", diarization=True):
                pass
        assert time.monotonic() - started < 5.0
    finally:
        stop()
    _assert_rejection(caught.value)


def test_server_rejection_raises_a_protocol_error_with_the_param_sync():
    url, stop = _serve_in_thread(_replay_server(RECORDED["diarization_unavailable"], []))
    try:
        with pytest.raises(ProtocolError) as caught:
            list(ASRClient().transcribe_stream(_SILENCE, url=url, diarization=True))
    finally:
        stop()
    _assert_rejection(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(("session", "rejected"), [
    pytest.param({"diarization": True}, False, id="enabled"),
    pytest.param({"diarization": False}, True, id="disabled"),
    pytest.param({}, True, id="server-without-diarization"),
])
async def test_an_acknowledgement_that_does_not_enable_diarization_is_rejected(session, rejected):
    client_session = AsyncASRSession("ws://unused", language="ja", diarization=True)
    frame = {"type": "transcription_session.updated", "session": session}
    if rejected:
        with pytest.raises(ProtocolError) as caught:
            await client_session._handle_text_frame(frame)
        assert (caught.value.code, caught.value.param) == ("invalid_parameters", "session.diarization")
        assert not client_session._session_ready.is_set()
    else:
        await client_session._handle_text_frame(frame)
        assert client_session._session_ready.is_set()


@pytest.mark.asyncio
async def test_an_acknowledgement_is_not_checked_without_diarization():
    client_session = AsyncASRSession("ws://unused", language="ja")
    await client_session._handle_text_frame({"type": "transcription_session.updated", "session": {}})
    assert client_session._session_ready.is_set()


@pytest.mark.asyncio
async def test_other_invalid_parameters_keep_their_param_and_the_generic_message():
    session = AsyncASRSession("ws://unused", language="ja")
    frame = {"type": "error", "error": {"code": "invalid_parameters", "param": "session.delay", "message": "bad"}}
    with pytest.raises(ProtocolError) as caught:
        await session._handle_text_frame(frame)
    assert (caught.value.code, caught.value.param) == ("invalid_parameters", "session.delay")
    assert "retry without diarization" not in str(caught.value)
