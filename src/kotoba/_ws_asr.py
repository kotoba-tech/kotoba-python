"""ASR WebSocket session.

Implements the protocol from `kotoba-realtime-subtitles`'s
`v1/realtime.py` server. Summary:

    (open)                                   — server accepts WS
    transcription_session.created  (S->C)    — informational, fired on connect
    transcription_session.update   (C->S)    — client config: format, rate, lang
    transcription_session.updated  (S->C)    — server ack; after this, audio accepted
    input_audio_buffer.append      (C->S)*   — base64 pcm16 chunks
    conversation.item.created      (S->C)    — one-time, on first append
    conversation.item.input_audio_transcription.delta (S->C)* — transcript tokens
    input_audio_buffer.commit      (C->S)    — tell server "no more audio"
    …more delta frames flush out of the buffer…
    input_audio_buffer.committed   (S->C)    — terminal; session ends shortly after
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any, AsyncIterable, Iterable, Union

from kotoba._ws_base import AsyncSession, SyncSession
from kotoba.errors import APIError, ProtocolError
from kotoba.models import StreamEvent

AudioSource = Union[Iterable[bytes], AsyncIterable[bytes]]
_FEED_STOP = object()


class AsyncASRSession(AsyncSession):
    """Async streaming ASR session.

    Usage:
        async with AsyncASRSession(url, language="ja", api_key=...) as s:
            await s.send_audio(pcm16_chunk)
            ...
            await s.commit()
            async for event in s:
                if event.type == "partial_transcript":
                    print(event.text, end="")
                elif event.type == "done":
                    break
    """

    def __init__(
        self,
        url: str,
        *,
        language: str,
        sample_rate: int = 24000,
        keywords: list[str] | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(url, api_key=api_key)
        self._language = language
        self._sample_rate = sample_rate
        self._keywords = keywords
        self._session_ready = asyncio.Event()
        self._event_counter = 0

    # -- handshake ---------------------------------------------------------

    async def _handshake(self) -> None:
        update = {
            "type": "transcription_session.update",
            "session": {
                "input_audio_format": "pcm16",
                "input_audio_sample_rate": self._sample_rate,
                "input_audio_number_of_channels": 1,
                "input_audio_transcription": {
                    "language": self._language,
                    "target_language": self._language,
                },
                "turn_detection": False,
            },
        }
        if self._keywords:
            update["session"]["input_audio_transcription"]["keywords"] = self._keywords
        await self._send_json(update)
        await self._session_ready.wait()

    # -- senders -----------------------------------------------------------

    async def send_audio(self, pcm16_chunk: bytes) -> None:
        """Send one PCM16 LE audio chunk as `input_audio_buffer.append`."""

        if not pcm16_chunk:
            return
        if len(pcm16_chunk) % 2 != 0:
            raise ValueError("pcm16 chunk length must be a multiple of 2 bytes")
        self._event_counter += 1
        await self._send_json(
            {
                "event_id": f"chunk-{self._event_counter:07d}",
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm16_chunk).decode("ascii"),
            }
        )

    async def commit(self) -> None:
        """Tell the server we're done sending audio for this turn."""

        await self._send_json({"type": "input_audio_buffer.commit"})

    async def feed(self, audio: AudioSource) -> None:
        """Drive a full turn from an audio source, then commit.

        Accepts a sync iterable (e.g. file chunks) or an async iterable
        (e.g. a live microphone queue) of PCM16 byte chunks. Sync iterators
        are pulled on a worker thread so the receiver loop keeps draining
        transcript deltas in parallel.
        """

        if isinstance(audio, AsyncIterable):
            async for chunk in audio:
                if chunk:
                    await self.send_audio(chunk)
        elif isinstance(audio, Iterable):
            it = iter(audio)
            while True:
                chunk = await asyncio.to_thread(next, it, _FEED_STOP)
                if chunk is _FEED_STOP:
                    break
                if chunk:
                    await self.send_audio(chunk)
        else:
            raise TypeError(
                f"feed() expected Iterable[bytes] / AsyncIterable[bytes], got {type(audio).__name__}"
            )
        await self.commit()

    # -- receiver ----------------------------------------------------------

    async def _handle_text_frame(self, payload: dict[str, Any]) -> None:
        msg_type = payload.get("type")

        if msg_type == "transcription_session.created":
            # Informational — server greets on connect. Not ready yet.
            return

        if msg_type == "transcription_session.updated":
            self._session_ready.set()
            await self._emit(StreamEvent(type="session_ready", metadata=payload))
            return

        if msg_type == "conversation.item.created":
            # No user-facing signal; skip.
            return

        if msg_type == "conversation.item.input_audio_transcription.delta":
            delta = payload.get("delta", "")
            if delta:
                await self._emit(
                    StreamEvent(type="partial_transcript", text=delta, metadata=payload)
                )
            return

        if msg_type == "conversation.item.input_audio_transcription.completed":
            transcript = payload.get("transcript", "")
            if transcript:
                await self._emit(
                    StreamEvent(
                        type="final_transcript", text=transcript, metadata=payload
                    )
                )
            return

        if msg_type == "input_audio_buffer.committed":
            await self._emit(StreamEvent(type="committed", metadata=payload))
            await self._emit(StreamEvent(type="done", metadata=payload))
            await self._emit_done()
            return

        if msg_type == "error":
            err = payload.get("error", payload)
            code = err.get("code") if isinstance(err, dict) else None
            raise ProtocolError(
                f"ASR server error: {err}",
                code=str(code) if code is not None else "error",
                payload=payload,
            )


class ASRSession(SyncSession):
    """Sync wrapper over `AsyncASRSession`. Same API minus `await`."""

    def __init__(
        self,
        url: str,
        *,
        language: str,
        sample_rate: int = 24000,
        keywords: list[str] | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            AsyncASRSession(
                url,
                language=language,
                sample_rate=sample_rate,
                keywords=keywords,
                api_key=api_key,
            )
        )

    @property
    def _asr(self) -> AsyncASRSession:
        assert isinstance(self._async, AsyncASRSession)
        return self._async

    def send_audio(self, pcm16_chunk: bytes) -> None:
        if self._loop is None:
            raise APIError("ASRSession is not started; use it as a context manager")
        self._run(self._asr.send_audio(pcm16_chunk))

    def commit(self) -> None:
        if self._loop is None:
            raise APIError("ASRSession is not started; use it as a context manager")
        self._run(self._asr.commit())

    def feed(self, audio: AudioSource) -> None:
        """Blocking mirror of ``AsyncASRSession.feed``.

        Note: callers that need transcript deltas to surface *while* the
        audio source is still yielding (the realtime-mic case) should use
        the higher-level ``ASRClient.transcribe_stream`` — it schedules
        the feeder on the background loop so iteration and feed run in
        parallel. ``feed()`` here blocks until the final ``commit``.
        """

        if self._loop is None:
            raise APIError("ASRSession is not started; use it as a context manager")
        self._run(self._asr.feed(audio))
