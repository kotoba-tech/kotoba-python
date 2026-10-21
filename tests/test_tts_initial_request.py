"""Initial TTS requests must arrive before readiness, exactly once."""

import asyncio
import base64
import json

import pytest
import websockets
from kotoba._ws_tts import AsyncTTSSession, TTSSession
from kotoba.errors import APIError, ProtocolError
from kotoba.tts import AsyncTTSClient, TTSClient

PCM = b"\x00\x00\x01\x00"


async def send_response(ws, response_id):
    await ws.send(
        json.dumps(
            {
                "type": "audio.chunk",
                "audio": base64.b64encode(PCM).decode(),
                "response_id": response_id,
                "isFinal": True,
            }
        )
    )
    await ws.send(
        json.dumps(
            {
                "type": "response.done",
                "response": {"id": response_id, "status": "completed"},
            }
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("upgrade", [False, True], ids=["frames", "upgrade-header"])
@pytest.mark.parametrize("surface", ["session", "stream", "batch"])
@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
async def test_initial_request_precedes_ready_and_preserves_negotiation(surface, sync, upgrade):
    received = []
    finished = asyncio.Event()

    async def server(ws):
        try:
            assert ws.request.headers["Authorization"] == "Bearer test-key"
            if upgrade:
                initial = json.loads(ws.request.headers["X-Kotoba-TTS-Init"])
                opening = initial["open"]
                received.append(initial["request"])
            else:
                opening = json.loads(await ws.recv())
                # Readiness is withheld until the initial text has arrived.
                received.append(json.loads(await asyncio.wait_for(ws.recv(), 1)))
            assert opening["type"] == "open"
            await ws.send(
                json.dumps(
                    {
                        "type": "session.created",
                        "sample_rate": 16000,
                        "format": "pcm16",
                        "client_id": "test-session",
                        "initial_request_in_upgrade": upgrade,
                    }
                )
            )
            await send_response(ws, "first")
            if surface == "session":
                received.append(json.loads(await asyncio.wait_for(ws.recv(), 1)))
                await send_response(ws, "second")
            async for extra in ws:
                received.append(json.loads(extra))
        finally:
            finished.set()

    async with websockets.serve(server, "127.0.0.1", 0) as listener:
        url = f"ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}"
        kwargs = dict(
            language="ja",
            speaker_id="test",
            url=url,
            audio_format="pcm16",
            sample_rate=16000,
            initial_request_in_upgrade=upgrade,
        )

        def run_sync():
            client = TTSClient(api_key="test-key")
            if surface == "stream":
                return b"".join(client.synthesize_stream("first", **kwargs))
            if surface == "batch":
                result = client.synthesize("first", **kwargs)
                assert result.sample_rate == 16000 and result.audio_format == "pcm16"
                return result.data
            with client.stream(initial_text="first", **kwargs) as session:
                assert session.sample_rate == 16000 and session.audio_format == "pcm16"
                first = b"".join(e.audio for e in session if e.type == "audio_chunk")
                session.synthesize("second", response_id="second")
                second = b"".join(e.audio for e in session if e.type == "audio_chunk")
                assert second == PCM
                return first

        async def run_async():
            client = AsyncTTSClient(api_key="test-key")
            if surface == "stream":
                return b"".join([chunk async for chunk in client.synthesize_stream("first", **kwargs)])
            if surface == "batch":
                result = await client.synthesize("first", **kwargs)
                assert result.sample_rate == 16000 and result.audio_format == "pcm16"
                return result.data
            async with client.stream(initial_text="first", **kwargs) as session:
                assert session.sample_rate == 16000 and session.audio_format == "pcm16"
                first = b"".join([e.audio async for e in session if e.type == "audio_chunk"])
                await session.synthesize("second", response_id="second")
                second = b"".join([e.audio async for e in session if e.type == "audio_chunk"])
                assert second == PCM
                return first

        result = await asyncio.to_thread(run_sync) if sync else await run_async()
        assert result == PCM
        await asyncio.wait_for(finished.wait(), 1)

    assert received == [
        {"type": "response.create", "text": "first"},
        *([{"type": "response.create", "text": "second", "response_id": "second"}] if surface == "session" else []),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
@pytest.mark.parametrize("reject", [False, True], ids=["disconnect", "invalid-open"])
async def test_initial_request_setup_failure_closes_without_retry(sync, reject):
    requests = []

    async def server(ws):
        await ws.recv()
        requests.append(json.loads(await ws.recv()))
        if reject:
            await ws.send(json.dumps({"type": "error", "message": "unsupported_language"}))
        await ws.close()

    async with websockets.serve(server, "127.0.0.1", 0) as listener:
        url = f"ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}"
        cls = TTSSession if sync else AsyncTTSSession
        session = cls(url, language="ja", speaker_id="test", initial_text="first")

        def run_sync():
            with session:
                pytest.fail("A failed handshake must not yield a session")

        async def run_async():
            async with session:
                pytest.fail("A failed handshake must not yield a session")

        with pytest.raises(ProtocolError if reject else APIError):
            if sync:
                await asyncio.to_thread(run_sync)
            else:
                await run_async()
        async_session = session._async if sync else session
        assert async_session._ws is None
        assert async_session._receiver_task.done()
        if sync:
            assert session._loop is None and session._thread is None
    assert requests == [{"type": "response.create", "text": "first"}]


@pytest.mark.asyncio
async def test_cancel_during_initial_request_handshake_closes_transport():
    received = asyncio.Event()
    disconnected = asyncio.Event()

    async def server(ws):
        await ws.recv()
        assert json.loads(await ws.recv()) == {"type": "response.create", "text": "first"}
        received.set()
        await ws.wait_closed()
        disconnected.set()

    async with websockets.serve(server, "127.0.0.1", 0) as listener:
        url = f"ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}"
        session = AsyncTTSSession(url, language="ja", speaker_id="test", initial_text="first")
        starting = asyncio.create_task(session.start())
        await asyncio.wait_for(received.wait(), 1)
        starting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await starting
        await asyncio.wait_for(disconnected.wait(), 1)
        assert session._ws is None and session._receiver_task.done()


@pytest.mark.parametrize("client_class", [TTSClient, AsyncTTSClient])
@pytest.mark.parametrize("text", ["", 0, False, ["not a string"]])
def test_invalid_initial_text_fails_before_connection(client_class, text):
    with pytest.raises(ValueError, match="non-empty string"):
        client_class(api_key=None).stream(initial_text=text, url="ws://unused")


@pytest.mark.parametrize("client_class", [TTSClient, AsyncTTSClient])
@pytest.mark.parametrize("text", [None, "あ" * 10000])
def test_upgrade_requires_bounded_initial_text(client_class, text):
    with pytest.raises(ValueError):
        client_class(api_key=None).stream(
            initial_text=text,
            initial_request_in_upgrade=True,
            url="ws://unused",
        )


@pytest.mark.asyncio
async def test_upgrade_missing_ack_does_not_resend_initial_text():
    frames = []

    async def server(ws):
        assert "X-Kotoba-TTS-Init" in ws.request.headers
        await ws.send(json.dumps({"type": "session.created"}))
        async for frame in ws:
            frames.append(frame)

    async with websockets.serve(server, "127.0.0.1", 0) as listener:
        url = f"ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}"
        with pytest.raises(ProtocolError, match="acknowledge"):
            async with AsyncTTSSession(
                url,
                language="ja",
                speaker_id="test",
                initial_text="first",
                initial_request_in_upgrade=True,
            ):
                pytest.fail("Missing acknowledgement must fail setup")
    assert frames == []


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [None, "", 123])
@pytest.mark.parametrize("stream", [False, True], ids=["batch", "chunks"])
@pytest.mark.parametrize("sync", [False, True], ids=["async", "sync"])
async def test_helpers_reject_invalid_text_before_connecting(text, stream, sync):
    # An unreachable URL makes accidental connection attempts fail this test.
    kwargs = {"url": "ws://127.0.0.1:1", "language": "ja"}
    with pytest.raises(ValueError, match="non-empty string"):
        if sync:
            client = TTSClient(api_key=None)
            if stream:
                list(client.synthesize_stream(text, **kwargs))
            else:
                client.synthesize(text, **kwargs)
        else:
            client = AsyncTTSClient(api_key=None)
            if stream:
                [chunk async for chunk in client.synthesize_stream(text, **kwargs)]
            else:
                await client.synthesize(text, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("language,speaker", [("ja", None), ("en", None), ("ja", "explicit")])
async def test_speaker_default_is_owned_and_negotiated_by_server(language, speaker):
    async def server(ws):
        opening = json.loads(await ws.recv())
        assert opening["language"] == language
        if speaker is None:
            assert "speaker_id" not in opening
        else:
            assert opening["speaker_id"] == speaker
        await ws.send(
            json.dumps(
                {
                    "type": "session.created",
                    "speaker_id": speaker or f"configured-{language}",
                    "client_id": "speaker-test",
                    "sample_rate": 24000,
                    "format": "pcm_f32",
                }
            )
        )
        await ws.wait_closed()

    async with websockets.serve(server, "127.0.0.1", 0) as listener:
        url = f"ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}"
        async with AsyncTTSSession(url, language=language, speaker_id=speaker) as session:
            assert session.speaker_id == (speaker or f"configured-{language}")
