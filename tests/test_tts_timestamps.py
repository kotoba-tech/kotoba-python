"""Public TTS timestamp options against a real local WebSocket transport."""

import asyncio
import base64
import json

import pytest
import websockets
from kotoba._ws_tts import AsyncTTSSession, TTSSession
from kotoba.errors import ProtocolError
from kotoba.tts import AsyncTTSClient, TTSClient

PCM = b"\x00" * 16
INVALID_TIMESTAMP_OPTIONS = [None, 0, 1, "false", "true", {}]


def make_session(sync, surface, url, **options):
    if surface == "client":
        client_class = TTSClient if sync else AsyncTTSClient
        return client_class(api_key=None).stream(url=url, language="ja", **options)
    session_class = TTSSession if sync else AsyncTTSSession
    return session_class(url, language="ja", **options)


async def send_response(ws, request):
    payload = {
        "type": "audio.chunk",
        "response_id": request["text"],
        "audio": base64.b64encode(PCM).decode(),
        "isFinal": True,
    }
    if request.get("with_timestamps") is True:
        payload["alignments"] = [{"start": 0.0, "end": 0.25, "text": request["text"]}]
    await ws.send(json.dumps(payload))
    await ws.send(
        json.dumps({"type": "response.done", "response": {"status": "completed"}})
    )


def assert_response(events, text, with_timestamps):
    chunks = [event for event in events if event.type == "audio_chunk"]
    assert len(chunks) == 1 and chunks[0].audio == PCM
    expected_metadata = {"type": "audio.chunk", "response_id": text, "isFinal": True}
    if with_timestamps:
        expected_metadata["alignments"] = [{"start": 0.0, "end": 0.25, "text": text}]
    assert chunks[0].metadata == expected_metadata
    assert events[-1].type == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
@pytest.mark.parametrize("surface", ["client", "session"])
@pytest.mark.parametrize("upgrade", [False, True], ids=["frames", "upgrade-header"])
@pytest.mark.parametrize(
    "options,default",
    [
        pytest.param({}, False, id="legacy-default"),
        pytest.param({"with_timestamps": False}, False, id="disabled-default"),
        pytest.param({"with_timestamps": True}, True, id="enabled-default"),
    ],
)
async def test_timestamp_initial_request_and_turn_overrides(sync, surface, upgrade, options, default):
    received = []
    wire_requests = []
    finished = asyncio.Event()

    async def server(ws):
        try:
            if upgrade:
                initial = json.loads(ws.request.headers["X-Kotoba-TTS-Init"])
                assert set(initial) == {"open", "request"}
                opening = initial["open"]
                request = initial["request"]
                wire_requests.append(json.dumps(request))
            else:
                assert "X-Kotoba-TTS-Init" not in ws.request.headers
                opening = json.loads(await ws.recv())
                raw = await asyncio.wait_for(ws.recv(), 1)
                request = json.loads(raw)
                wire_requests.append(raw)
            assert opening == {"type": "open", "language": "ja"}
            received.append(request)
            await ws.send(
                json.dumps({"type": "session.created", "initial_request_in_upgrade": upgrade})
            )
            await send_response(ws, request)
            async for raw in ws:
                request = json.loads(raw)
                received.append(request)
                wire_requests.append(raw)
                await send_response(ws, request)
        finally:
            finished.set()

    turns = [
        ("inherited", {}, default),
        ("disabled", {"with_timestamps": False}, False),
        ("enabled", {"with_timestamps": True}, True),
        ("explicit-none", {"with_timestamps": None}, default),
    ]
    async with websockets.serve(server, "127.0.0.1", 0) as listener:
        url = f"ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}"
        session = make_session(
            sync, surface, url, initial_text="initial", initial_request_in_upgrade=upgrade, **options
        )

        def run_sync():
            with session:
                assert_response(list(session), "initial", default)
                for text, override, enabled in turns:
                    session.synthesize(text, response_id=text, **override)
                    assert_response(list(session), text, enabled)

        async def run_async():
            async with session:
                assert_response([event async for event in session], "initial", default)
                for text, override, enabled in turns:
                    await session.synthesize(text, response_id=text, **override)
                    assert_response([event async for event in session], text, enabled)

        if sync:
            await asyncio.to_thread(run_sync)
        else:
            await run_async()
        await asyncio.wait_for(finished.wait(), 1)

    initial = {"type": "response.create", "text": "initial"}
    if default:
        initial["with_timestamps"] = True
    expected = [initial]
    for text, _, enabled in turns:
        request = {"type": "response.create", "text": text, "response_id": text}
        if enabled:
            request["with_timestamps"] = True
        expected.append(request)
    assert received == expected
    assert wire_requests == [json.dumps(request) for request in expected]


@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
@pytest.mark.parametrize("surface", ["client", "session"])
@pytest.mark.parametrize("option", INVALID_TIMESTAMP_OPTIONS)
def test_invalid_timestamp_default_fails_before_connection(sync, surface, option):
    with pytest.raises(ValueError, match="with_timestamps must be a bool"):
        make_session(sync, surface, "ws://127.0.0.1:1", with_timestamps=option)


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
async def test_invalid_timestamp_override_is_not_sent(sync):
    received = []
    finished = asyncio.Event()

    async def server(ws):
        try:
            await ws.recv()
            await ws.send(json.dumps({"type": "session.created"}))
            async for raw in ws:
                request = json.loads(raw)
                received.append(request)
                await send_response(ws, request)
        finally:
            finished.set()

    async with websockets.serve(server, "127.0.0.1", 0) as listener:
        url = f"ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}"
        session = make_session(sync, "client", url, with_timestamps=True)

        def run_sync():
            with session:
                for option in INVALID_TIMESTAMP_OPTIONS[1:]:
                    with pytest.raises(ValueError, match="with_timestamps must be a bool"):
                        session.synthesize("invalid", with_timestamps=option)
                session.synthesize("valid")
                assert_response(list(session), "valid", True)

        async def run_async():
            async with session:
                for option in INVALID_TIMESTAMP_OPTIONS[1:]:
                    with pytest.raises(ValueError, match="with_timestamps must be a bool"):
                        await session.synthesize("invalid", with_timestamps=option)
                await session.synthesize("valid")
                assert_response([event async for event in session], "valid", True)

        if sync:
            await asyncio.to_thread(run_sync)
        else:
            await run_async()
        await asyncio.wait_for(finished.wait(), 1)

    assert received == [{"type": "response.create", "text": "valid", "with_timestamps": True}]


def test_max_length_is_deprecated_and_not_sent():
    from kotoba._ws_tts import _synthesis_request

    with pytest.warns(DeprecationWarning, match="max_length"):
        frame = _synthesis_request("text", max_length=100)
    assert frame == {"type": "response.create", "text": "text"}


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
async def test_a_failed_response_leaves_the_session_usable(sync):
    async def server(ws):
        await ws.recv()
        await ws.send(json.dumps({"type": "session.created"}))
        async for raw in ws:
            request = json.loads(raw)
            if request.get("with_timestamps") is True:
                # A profile without timestamp calibration fails only this response.
                error = {"code": "timestamps_unavailable", "message": "timestamp capture is unavailable"}
                await ws.send(json.dumps({"type": "response.done", "response": {"status": "failed", "error": error}}))
            else:
                await send_response(ws, request)

    async with websockets.serve(server, "127.0.0.1", 0) as listener:
        url = f"ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}"
        session = make_session(sync, "session", url)

        def run_sync():
            with session:
                session.synthesize("timed", with_timestamps=True)
                with pytest.raises(ProtocolError) as failed:
                    list(session)
                assert failed.value.code == "timestamps_unavailable"
                session.synthesize("plain")
                assert_response(list(session), "plain", False)

        async def run_async():
            async with session:
                await session.synthesize("timed", with_timestamps=True)
                with pytest.raises(ProtocolError) as failed:
                    [event async for event in session]
                assert failed.value.code == "timestamps_unavailable"
                await session.synthesize("plain")
                assert_response([event async for event in session], "plain", False)

        if sync:
            await asyncio.to_thread(run_sync)
        else:
            await run_async()
