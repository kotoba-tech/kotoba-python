"""TTS WebSocket session.

Implements the one-shot protocol from the kotoba-realtime-subtitles
`v2/tts/ws.py` server.

Lifecycle:
    open                       (C->S)
    session.created            (S->C, sets sample_rate / format / client_id)
    response.create            (C->S)  ─┐ per turn (text in one frame)
    response.created           (S->C)   │
    audio.chunk (isFinal=...)  (S->C)*  │  emitted as audio_chunk events
    response.done              (S->C)  ─┘

Cancel:
    response.cancel            (C->S)
    response.done(cancelled)   (S->C)
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

from kotoba._ws_base import AsyncSession, SyncSession
from kotoba.errors import APIError, ProtocolError
from kotoba.models import StreamEvent

logger = logging.getLogger(__name__)

DEFAULT_SPEAKER_BY_LANGUAGE = {
    # Available Japanese speakers: ``ja-man-m02-azawa`` (male) and
    # ``ja-woman-f04-me`` (female). Pass ``speaker_id=`` to override.
    "ja": "ja-man-m02-azawa",
}


class AsyncTTSSession(AsyncSession):
    """Async streaming TTS session (one-shot text in, audio chunks out).

    Usage:
        async with AsyncTTSSession(url, language="ja", api_key=...) as tts:
            await tts.synthesize("こんにちは。")
            async for event in tts:
                if event.type == "audio_chunk":
                    play(event.audio)
                elif event.type == "done":
                    break

    Server-side `timeout` frames (non-fatal worker-result timeouts) are
    logged at WARNING and otherwise ignored — the server escalates to a
    hard `error` close after its own retry budget is exhausted.
    """

    def __init__(
        self,
        url: str,
        *,
        language: str,
        speaker_id: str | None = None,
        spk_ref_audio_tokens: Any = None,
        audio_format: str | None = None,
        sample_rate: int | None = None,
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
        self._req_audio_format = audio_format
        self._req_sample_rate = sample_rate
        self._session_ready = asyncio.Event()

        # Negotiated result echoed back by session.created (not the request).
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
        # Omit the keys entirely when unset so the frame is byte-identical to the
        # pre-negotiation client: the server defaults to pcm_f32 @ 24 kHz.
        if self._req_audio_format is not None:
            open_frame["format"] = self._req_audio_format
        if self._req_sample_rate is not None:
            open_frame["sample_rate"] = self._req_sample_rate
        await self._send_json(open_frame)
        await self._session_ready.wait()

    # -- senders -----------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        *,
        response_id: str | None = None,
        max_length: int | None = None,
    ) -> None:
        """Send the full ``text`` as a single ``response.create`` frame.

        Does not wait for completion — caller drains audio via iteration.
        """

        if not isinstance(text, str) or not text:
            raise ValueError("synthesize() requires a non-empty string")
        frame: dict[str, Any] = {"type": "response.create", "text": text}
        if response_id is not None:
            frame["response_id"] = response_id
        if max_length is not None:
            frame["max_length"] = max_length
        await self._send_json(frame)

    async def cancel(self) -> None:
        """Cancel the in-flight response."""

        await self._send_json({"type": "response.cancel"})

    # -- receiver ----------------------------------------------------------

    async def _handle_text_frame(self, payload: dict[str, Any]) -> None:
        msg_type = payload.get("type")

        if msg_type == "session.created":
            self.sample_rate = int(payload.get("sample_rate", self.sample_rate))
            # lower-case defensively: an older server echoing the client's raw
            # casing must not break the lower-case-only AudioFormat Literal
            self.audio_format = str(payload.get("format", self.audio_format)).lower()
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
            # `completed` and `cancelled` both surface as `done`.
            metadata = dict(payload)
            metadata["status"] = status
            await self._emit(StreamEvent(type="done", metadata=metadata))
            await self._emit_done()
            return

        if msg_type == "error":
            raise ProtocolError(
                f"TTS server error: {payload.get('message', payload)}",
                code=str(payload.get("code", "error")),
                payload=payload,
            )

        if msg_type == "timeout":
            logger.warning(
                "TTS server reported worker-result timeout: %s",
                payload.get("message"),
            )
            return


class TTSSession(SyncSession):
    """Sync wrapper over `AsyncTTSSession`. Same API minus `await`."""

    def __init__(
        self,
        url: str,
        *,
        language: str,
        speaker_id: str | None = None,
        spk_ref_audio_tokens: Any = None,
        audio_format: str | None = None,
        sample_rate: int | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            AsyncTTSSession(
                url,
                language=language,
                speaker_id=speaker_id,
                spk_ref_audio_tokens=spk_ref_audio_tokens,
                audio_format=audio_format,
                sample_rate=sample_rate,
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

    def synthesize(
        self,
        text: str,
        *,
        response_id: str | None = None,
        max_length: int | None = None,
    ) -> None:
        self._ensure()
        self._run(
            self._tts.synthesize(text, response_id=response_id, max_length=max_length)
        )

    def cancel(self) -> None:
        self._ensure()
        self._run(self._tts.cancel())
