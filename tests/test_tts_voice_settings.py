"""TTS voice_settings: the session default rides on `open`, overrides on each `response.create`."""

import asyncio
import base64
import json

import pytest
import websockets
from kotoba._ws_tts import AsyncTTSSession, TTSSession
from kotoba.errors import ProtocolError
from kotoba.tts import AsyncTTSClient, TTSClient

PCM = b"\x00" * 16
SESSION_DEFAULT = {"speed": 1.2, "volume": 0.8}


async def send_response(ws, request):
    await ws.send(json.dumps({"type": "audio.chunk", "audio": base64.b64encode(PCM).decode(), "isFinal": True}))
    await ws.send(json.dumps({"type": "response.done", "response": {"status": "completed"}}))


def assert_response(events):
    assert [event.audio for event in events if event.type == "audio_chunk"] == [PCM]
    assert events[-1].type == "done"


def make_session(sync, surface, url, **options):
    if surface == "client":
        client_class = TTSClient if sync else AsyncTTSClient
        return client_class(api_key=None).stream(url=url, language="ja", **options)
    session_class = TTSSession if sync else AsyncTTSSession
    return session_class(url, language="ja", **options)


@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
@pytest.mark.parametrize("surface", ["client", "session"])
@pytest.mark.parametrize("upgrade", [False, True], ids=["frames", "upgrade-header"])
async def test_session_default_is_opened_and_overrides_are_per_request(sync, surface, upgrade):
    openings, received = [], []
    finished = asyncio.Event()

    async def server(ws):
        try:
            if upgrade:
                initial = json.loads(ws.request.headers["X-Kotoba-TTS-Init"])
                openings.append(initial["open"])
                request = initial["request"]
            else:
                openings.append(json.loads(await ws.recv()))
                request = json.loads(await ws.recv())
            received.append(request)
            await ws.send(json.dumps({"type": "session.created", "initial_request_in_upgrade": upgrade}))
            await send_response(ws, request)
            async for raw in ws:
                request = json.loads(raw)
                received.append(request)
                await send_response(ws, request)
        finally:
            finished.set()

    async with websockets.serve(server, "127.0.0.1", 0) as listener:
        url = f"ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}"
        session = make_session(
            sync, surface, url, voice_settings=SESSION_DEFAULT, initial_text="initial",
            initial_request_in_upgrade=upgrade,
        )

        def run_sync():
            with session:
                assert_response(list(session))
                session.synthesize("override", response_id="override", voice_settings={"pitch": -2})
                assert_response(list(session))
                session.synthesize("inherited", response_id="inherited")
                assert_response(list(session))

        async def run_async():
            async with session:
                assert_response([event async for event in session])
                await session.synthesize("override", response_id="override", voice_settings={"pitch": -2})
                assert_response([event async for event in session])
                await session.synthesize("inherited", response_id="inherited")
                assert_response([event async for event in session])

        if sync:
            await asyncio.to_thread(run_sync)
        else:
            await run_async()
        await asyncio.wait_for(finished.wait(), 1)

    assert openings == [{"type": "open", "language": "ja", "voice_settings": SESSION_DEFAULT}]
    assert received == [
        {"type": "response.create", "text": "initial"},
        {"type": "response.create", "text": "override", "response_id": "override", "voice_settings": {"pitch": -2}},
        {"type": "response.create", "text": "inherited", "response_id": "inherited"},
    ]


@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
async def test_rejected_override_ends_only_that_response(sync):
    finished = asyncio.Event()

    async def server(ws):
        try:
            await ws.recv()
            await ws.send(json.dumps({"type": "session.created"}))
            async for raw in ws:
                request = json.loads(raw)
                if request.get("voice_settings", {}).get("speed") == 2:
                    await ws.send(json.dumps({"type": "error", "code": "invalid_request",
                                              "message": "voice_settings.speed must be between 0.7 and 1.5"}))
                    continue
                await send_response(ws, request)
        finally:
            finished.set()

    async with websockets.serve(server, "127.0.0.1", 0) as listener:
        url = f"ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}"
        session = make_session(sync, "session", url)

        def run_sync():
            with session:
                session.synthesize("rejected", voice_settings={"speed": 2})
                with pytest.raises(ProtocolError) as rejected:
                    list(session)
                assert rejected.value.code == "invalid_request"
                session.synthesize("next")
                assert_response(list(session))

        async def run_async():
            async with session:
                await session.synthesize("rejected", voice_settings={"speed": 2})
                with pytest.raises(ProtocolError) as rejected:
                    [event async for event in session]
                assert rejected.value.code == "invalid_request"
                await session.synthesize("next")
                assert_response([event async for event in session])

        if sync:
            await asyncio.to_thread(run_sync)
        else:
            await run_async()
        await asyncio.wait_for(finished.wait(), 1)
