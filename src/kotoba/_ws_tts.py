"""TTS WebSocket session.

Implements the protocol from the kotoba-realtime-subtitles
`input_streaming_tts/ws.py` server (see plan for the full frame table).

Lifecycle:
    open                       (C->S)
    session.created            (S->C, sets sample_rate / format / client_id)
    response.create            (C->S)  ─┐ per turn
    response.created           (S->C)   │
    input_text_buffer.append   (C->S)*  │  zero or more
    input_text_buffer.commit   (C->S)   │
    audio.chunk (isFinal=...)  (S->C)*  │  emitted as audio_chunk events
    response.done              (S->C)  ─┘
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any, AsyncIterable, Iterable

from kotoba._ws_base import AsyncSession, SyncSession
from kotoba.errors import APIError, ProtocolError
from kotoba.models import StreamEvent, TextSource

_FEED_STOP = object()

DEFAULT_SPEAKER_BY_LANGUAGE = {
    "ja": "",
    # TODO(open-question-2): add Korean default once user provides speaker IDs.
}


class AsyncTTSSession(AsyncSession):
    """Async streaming TTS session.

    Usage:
        async with AsyncTTSSession(url, language="ja", api_key=...) as tts:
            await tts.start_response()
            await tts.append_text("こんにちは。")
            await tts.commit()
            async for event in tts:
                if event.type == "audio_chunk":
                    play(event.audio)
                elif event.type == "done":
                    break
    """

    def __init__(
        self,
        url: str,
        *,
        language: str,
        speaker_id: str | None = None,
        spk_ref_audio_tokens: Any = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(url, api_key=api_key)
        self._language = language
        self._speaker_id = speaker_id or DEFAULT_SPEAKER_BY_LANGUAGE.get(language)
        if self._speaker_id is None:
            raise ValueError(
                f"No default speaker_id for language {language!r}; "
                f"pass speaker_id= explicitly."
            )
        self._spk_ref = spk_ref_audio_tokens
        self._session_ready = asyncio.Event()

        # Populated from session.created.
        self.sample_rate: int = 24000
        self.audio_format: str = "pcm_f32"
        self.client_id: str | None = None

    # -- handshake ---------------------------------------------------------

    async def _handshake(self) -> None:
        open_frame: dict[str, Any] = {
            "type": "open",
            "language": self._language,
            "speaker_id": self._speaker_id,
        }
        if self._spk_ref is not None:
            open_frame["spk_ref_audio_tokens"] = self._spk_ref
        await self._send_json(open_frame)
        await self._session_ready.wait()

    # -- senders -----------------------------------------------------------

    async def start_response(self) -> None:
        """Begin a new TTS turn. Required before append_text / commit."""

        await self._send_json({"type": "response.create"})

    async def append_text(self, text: str) -> None:
        """Push a text fragment to the active turn."""

        if not text:
            return
        await self._send_json({"type": "input_text_buffer.append", "text": text})

    async def commit(self) -> None:
        """Mark the end of the current turn's text input."""

        await self._send_json({"type": "input_text_buffer.commit"})

    async def cancel(self) -> None:
        """Cancel the in-flight response."""

        await self._send_json({"type": "response.cancel"})

    async def synthesize(self, text: str) -> None:
        """One-shot helper: response.create -> append_text -> commit."""

        await self.start_response()
        await self.append_text(text)
        await self.commit()

    async def feed(self, text: TextSource) -> None:
        """Drive a full turn from a text source.

        Accepts a plain string, a sync iterable (e.g. an LLM SSE generator
        that blocks between tokens), or an async iterable. Sends
        `response.create`, streams `input_text_buffer.append` per chunk,
        then `input_text_buffer.commit`. Blocking sync iterators are pulled
        one token at a time on a worker thread so the receiver loop keeps
        draining audio frames in parallel.
        """

        await self.start_response()
        if isinstance(text, str):
            await self.append_text(text)
        elif isinstance(text, AsyncIterable):
            async for chunk in text:
                if chunk:
                    await self.append_text(chunk)
        elif isinstance(text, Iterable):
            it = iter(text)
            while True:
                chunk = await asyncio.to_thread(next, it, _FEED_STOP)
                if chunk is _FEED_STOP:
                    break
                if chunk:
                    await self.append_text(chunk)
        else:
            raise TypeError(
                f"feed() expected str / Iterable[str] / AsyncIterable[str], got {type(text).__name__}"
            )
        await self.commit()

    # -- receiver ----------------------------------------------------------

    async def _handle_text_frame(self, payload: dict[str, Any]) -> None:
        msg_type = payload.get("type")

        if msg_type == "session.created":
            self.sample_rate = int(payload.get("sample_rate", self.sample_rate))
            self.audio_format = str(payload.get("format", self.audio_format))
            self.client_id = payload.get("client_id")
            self._session_ready.set()
            await self._emit(StreamEvent(type="session_ready", metadata=payload))
            return

        if msg_type == "response.created":
            # Internal — turn started; no event emitted.
            return

        if msg_type == "audio.chunk":
            encoded = payload.get("audio", "")
            is_final = bool(payload.get("isFinal", False))
            audio_bytes = b""
            if encoded:
                try:
                    audio_bytes = base64.b64decode(encoded)
                except Exception as exc:
                    raise ProtocolError(f"Failed to decode audio chunk: {exc}") from exc
            await self._emit(
                StreamEvent(
                    type="audio_chunk",
                    audio=audio_bytes,
                    is_final=is_final,
                    metadata={k: v for k, v in payload.items() if k != "audio"},
                )
            )
            return

        if msg_type == "response.done":
            response = payload.get("response", {})
            status = response.get("status", "completed")
            if status == "failed":
                err = response.get("error", {})
                raise ProtocolError(
                    f"TTS response failed: {err.get('message', err)}",
                    code=str(err.get("code", "failed")),
                    payload=payload,
                )
            await self._emit(StreamEvent(type="done", metadata=payload))
            await self._emit_done()
            return

        if msg_type == "error":
            raise ProtocolError(
                f"TTS server error: {payload.get('message', payload)}",
                code=str(payload.get("code", "error")),
                payload=payload,
            )

        if msg_type == "timeout":
            raise ProtocolError(
                f"TTS server timeout: {payload.get('message', 'timeout')}",
                code="timeout",
                payload=payload,
            )


class TTSSession(SyncSession):
    """Sync wrapper over `AsyncTTSSession`. Same API minus `await`."""

    def __init__(
        self,
        url: str,
        *,
        language: str,
        speaker_id: str | None = None,
        spk_ref_audio_tokens: Any = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            AsyncTTSSession(
                url,
                language=language,
                speaker_id=speaker_id,
                spk_ref_audio_tokens=spk_ref_audio_tokens,
                api_key=api_key,
            )
        )

    @property
    def _tts(self) -> AsyncTTSSession:
        assert isinstance(self._async, AsyncTTSSession)
        return self._async

    @property
    def sample_rate(self) -> int:
        return self._tts.sample_rate

    @property
    def audio_format(self) -> str:
        return self._tts.audio_format

    def _ensure(self) -> None:
        if self._loop is None:
            raise APIError("TTSSession is not started; use it as a context manager")

    def start_response(self) -> None:
        self._ensure()
        self._run(self._tts.start_response())

    def append_text(self, text: str) -> None:
        self._ensure()
        self._run(self._tts.append_text(text))

    def commit(self) -> None:
        self._ensure()
        self._run(self._tts.commit())

    def cancel(self) -> None:
        self._ensure()
        self._run(self._tts.cancel())

    def synthesize(self, text: str) -> None:
        self._ensure()
        self._run(self._tts.synthesize(text))

    def feed(self, text: TextSource) -> None:
        """Blocking mirror of ``AsyncTTSSession.feed``.

        Note: callers that need audio chunks to surface *while* the
        generator is still yielding (the common LLM case) should use the
        higher-level ``TTSClient.synthesize_stream`` — it schedules the
        feeder on the background loop so iteration and feed run in
        parallel. ``feed()`` here blocks until the final ``commit`` frame
        has been sent.
        """

        self._ensure()
        self._run(self._tts.feed(text))
