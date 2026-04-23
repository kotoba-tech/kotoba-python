"""S2ST (speech-to-speech translation) WebSocket session.

Implements the OpenAI-Realtime-style protocol used by the existing
`sts-eval-*` endpoints (voice_session.update / input_audio_buffer.* /
conversation.item.input_audio_transcription.{text,audio}.delta).
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any

from kotoba._ws_base import AsyncSession, SyncSession
from kotoba.errors import APIError, ProtocolError
from kotoba.models import StreamEvent


class AsyncS2STSession(AsyncSession):
    """Async streaming S2S session.

    Lifecycle:
        async with AsyncS2STSession(url, src="en", tgt="ja", api_key=...) as s:
            await s.send_audio(pcm16_chunk)
            ...
            await s.commit()
            async for event in s:
                ...
    """

    def __init__(
        self,
        url: str,
        *,
        src_language: str,
        tgt_language: str,
        sample_rate: int = 24000,
        api_key: str | None = None,
    ) -> None:
        super().__init__(url, api_key=api_key)
        self._src = src_language
        self._tgt = tgt_language
        self._sample_rate = sample_rate
        self._session_ready = asyncio.Event()
        self._event_counter = 0

    # -- handshake ---------------------------------------------------------

    async def _handshake(self) -> None:
        update = {
            "type": "voice_session.update",
            "session": {
                "object": "realtime.voice_session",
                "turn_detection": False,
                "input_audio_format": "pcm16",
                "input_audio_sample_rate": self._sample_rate,
                "input_audio_number_of_channels": 1,
                "input_audio_transcription": {
                    "language": self._src,
                    "target_language": self._tgt,
                },
            },
        }
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
                "event_id": f"{self._event_counter:07d}",
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm16_chunk).decode("ascii"),
            }
        )

    async def commit(self) -> None:
        """Tell the server we're done sending audio for this turn."""

        await self._send_json({"type": "input_audio_buffer.commit"})

    # -- receiver ----------------------------------------------------------

    async def _handle_text_frame(self, payload: dict[str, Any]) -> None:
        msg_type = payload.get("type")

        if msg_type in ("voice_session.created", "voice_session.updated"):
            self._session_ready.set()
            await self._emit(StreamEvent(type="session_ready", metadata=payload))
            return

        if msg_type == "conversation.item.input_audio_transcription.text.delta":
            delta = payload.get("delta", "")
            if delta:
                await self._emit(
                    StreamEvent(type="partial_transcript", text=delta, metadata=payload)
                )
            return

        if msg_type == "conversation.item.input_audio_transcription.audio.delta":
            encoded = payload.get("delta", "")
            if encoded:
                try:
                    audio_bytes = base64.b64decode(encoded)
                except Exception as exc:
                    raise ProtocolError(f"Failed to decode audio delta: {exc}") from exc
                await self._emit(
                    StreamEvent(type="audio_chunk", audio=audio_bytes, metadata=payload)
                )
            return

        if msg_type == "input_audio_buffer.committed":
            await self._emit(StreamEvent(type="committed", metadata=payload))
            await self._emit(StreamEvent(type="done", metadata=payload))
            await self._emit_done()
            return

        if msg_type == "error":
            err = payload.get("error", payload)
            raise ProtocolError(
                f"S2ST server error: {err}",
                code=str(err.get("code") if isinstance(err, dict) else "error"),
                payload=payload,
            )


class S2STSession(SyncSession):
    """Sync wrapper over `AsyncS2STSession`.

    Same API minus `await`. Owns a background event loop while inside `with`.
    """

    def __init__(
        self,
        url: str,
        *,
        src_language: str,
        tgt_language: str,
        sample_rate: int = 24000,
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            AsyncS2STSession(
                url,
                src_language=src_language,
                tgt_language=tgt_language,
                sample_rate=sample_rate,
                api_key=api_key,
            )
        )

    @property
    def _s2st(self) -> AsyncS2STSession:
        # Type-narrowing accessor for the modality-specific async session.
        assert isinstance(self._async, AsyncS2STSession)
        return self._async

    def send_audio(self, pcm16_chunk: bytes) -> None:
        if self._loop is None:
            raise APIError("S2STSession is not started; use it as a context manager")
        self._run(self._s2st.send_audio(pcm16_chunk))

    def commit(self) -> None:
        if self._loop is None:
            raise APIError("S2STSession is not started; use it as a context manager")
        self._run(self._s2st.commit())
