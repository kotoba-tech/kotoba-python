"""TTS WebSocket session.

Implements the one-shot protocol from the kotoba-realtime-subtitles
`v2/tts/ws.py` server.

Lifecycle:
    open                       (C->S)
    response.create            (C->S, optional initial_text before session.created)
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
import json
import logging
import warnings
from typing import Any

from kotoba._ws_base import AsyncSession, SyncSession
from kotoba.errors import APIError, ProtocolError
from kotoba.models import StreamEvent, VoiceSettings

logger = logging.getLogger(__name__)

UPGRADE_ACK_TIMEOUT_SECONDS = 8.0


def _synthesis_request(
    text: str,
    *,
    response_id: str | None = None,
    max_length: int | None = None,
    with_timestamps: bool = False,
    voice_settings: VoiceSettings | None = None,
) -> dict[str, Any]:
    if not isinstance(text, str) or not text:
        raise ValueError("synthesize() requires a non-empty string")
    if not isinstance(with_timestamps, bool):
        raise ValueError("with_timestamps must be a bool")
    frame: dict[str, Any] = {"type": "response.create", "text": text}
    if response_id is not None:
        frame["response_id"] = response_id
    if max_length is not None:
        # The server takes the length limit from its serving profile
        # (max-seq-length); a per-request value was never applied.
        warnings.warn(
            "max_length is ignored by the server and is no longer sent; it will be removed",
            DeprecationWarning,
            stacklevel=3,
        )
    if with_timestamps:
        frame["with_timestamps"] = True
    if voice_settings is not None:
        frame["voice_settings"] = dict(voice_settings)
    return frame


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

    ``voice_settings`` (``speed`` and ``volume`` multipliers, ``pitch`` in
    semitones) is the session default; ``synthesize(voice_settings=...)`` overrides it field by
    field for one response. The server validates the values.
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
        with_timestamps: bool = False,
        voice_settings: VoiceSettings | None = None,
        initial_text: str | None = None,
        initial_request_in_upgrade: bool = False,
        api_key: str | None = None,
        connect_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(with_timestamps, bool):
            raise ValueError("with_timestamps must be a bool")
        super().__init__(url, api_key=api_key, connect_kwargs=connect_kwargs)
        self._language = language
        self._speaker_id = speaker_id
        self._spk_ref = spk_ref_audio_tokens
        self._req_audio_format = audio_format
        self._req_sample_rate = sample_rate
        self._with_timestamps = with_timestamps
        self._voice_settings = dict(voice_settings) if voice_settings is not None else None
        self._session_ready = asyncio.Event()
        self._initial_request_in_upgrade = initial_request_in_upgrade
        self._upgrade_ack = False
        self._initial_request = (
            _synthesis_request(initial_text, with_timestamps=with_timestamps)
            if initial_text is not None else None
        )

        # Negotiated result echoed back by session.created (not the request).
        self.sample_rate: int = 24000
        self.audio_format: str = "pcm_f32"
        self.client_id: str | None = None
        self.speaker_id: str | None = None

        if initial_request_in_upgrade:
            if self._initial_request is None:
                raise ValueError("initial_request_in_upgrade requires initial_text")
            header = json.dumps({"open": self._open_frame(), "request": self._initial_request})
            if len(header.encode("ascii")) > 8192:
                raise ValueError("initial request header exceeds 8192 bytes; use frame-based setup")
            self._handshake_headers["X-Kotoba-TTS-Init"] = header
            self._initial_request = None

    # -- handshake ---------------------------------------------------------

    def _open_frame(self) -> dict[str, Any]:
        open_frame: dict[str, Any] = {
            "type": "open",
            "language": self._language,
        }
        if self._speaker_id is not None:
            open_frame["speaker_id"] = self._speaker_id
        if self._spk_ref is not None:
            open_frame["spk_ref_audio_tokens"] = self._spk_ref
        # Omit the keys entirely when unset so the frame is byte-identical to the
        # pre-negotiation client: the server defaults to pcm_f32 @ 24 kHz.
        if self._req_audio_format is not None:
            open_frame["format"] = self._req_audio_format
        if self._req_sample_rate is not None:
            open_frame["sample_rate"] = self._req_sample_rate
        if self._voice_settings is not None:
            open_frame["voice_settings"] = self._voice_settings
        return open_frame

    async def _handshake(self) -> None:
        if self._initial_request_in_upgrade:
            try:
                # Below the server's 10 s open timeout, so a stripped header
                # surfaces as this error rather than a close about `open`.
                await asyncio.wait_for(self._session_ready.wait(), UPGRADE_ACK_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                raise ProtocolError(
                    "The server did not receive the X-Kotoba-TTS-Init header (a proxy may strip it); "
                    "retry without initial_request_in_upgrade"
                ) from None
            if not self._upgrade_ack:
                raise ProtocolError("Server did not acknowledge initial request in Upgrade")
            return
        await self._send_json(self._open_frame())
        # The server queues text while preparing the session. Consume the
        # request before sending: an ambiguous send must not be retried.
        initial_request, self._initial_request = self._initial_request, None
        if initial_request is not None:
            await self._send_json(initial_request)
        await self._session_ready.wait()

    # -- senders -----------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        *,
        response_id: str | None = None,
        max_length: int | None = None,
        with_timestamps: bool | None = None,
        voice_settings: VoiceSettings | None = None,
    ) -> None:
        """Send the full ``text`` as a single ``response.create`` frame.

        Does not wait for completion — caller drains audio via iteration.
        ``max_length`` is deprecated and ignored (the server's profile sets the
        limit); a response that reaches it ends ``completed`` with
        ``finish_reason="length"`` in the ``done`` event metadata.
        ``with_timestamps=None`` uses the session default; an explicit bool
        overrides it for this response. Alignments remain in event metadata.
        ``voice_settings`` overrides the session default's fields for this response.
        """

        await self._send_json(
            _synthesis_request(
                text,
                response_id=response_id,
                max_length=max_length,
                with_timestamps=self._with_timestamps if with_timestamps is None else with_timestamps,
                voice_settings=voice_settings,
            )
        )

    async def cancel(self) -> None:
        """Cancel the in-flight response."""

        await self._send_json({"type": "response.cancel"})

    # -- receiver ----------------------------------------------------------

    async def _handle_text_frame(self, payload: dict[str, Any]) -> None:
        msg_type = payload.get("type")

        if msg_type == "session.created":
            self._upgrade_ack = payload.get("initial_request_in_upgrade") is True
            self.sample_rate = int(payload.get("sample_rate", self.sample_rate))
            # lower-case defensively: an older server echoing the client's raw
            # casing must not break the lower-case-only AudioFormat Literal
            self.audio_format = str(payload.get("format", self.audio_format)).lower()
            self.client_id = payload.get("client_id")
            self.speaker_id = payload.get("speaker_id")
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
                # Response-scoped (e.g. timestamps_unavailable): the server keeps the connection for the next request.
                await self._emit_error(
                    ProtocolError(
                        f"TTS response failed: {err.get('message', err)}",
                        code=str(err.get("code", "failed")),
                        payload=payload,
                    )
                )
                return
            # `completed` and `cancelled` both surface as `done`.
            metadata = dict(payload)
            metadata["status"] = status
            await self._emit(StreamEvent(type="done", metadata=metadata))
            await self._emit_done()
            return

        if msg_type == "error":
            error = ProtocolError(
                f"TTS server error: {payload.get('message', payload)}",
                code=str(payload.get("code", "error")),
                payload=payload,
            )
            if error.code == "invalid_request":
                # The server rejected only that request (e.g. voice_settings out of range) and keeps the connection.
                await self._emit_error(error)
                return
            raise error

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
        with_timestamps: bool = False,
        voice_settings: VoiceSettings | None = None,
        initial_text: str | None = None,
        initial_request_in_upgrade: bool = False,
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
                with_timestamps=with_timestamps,
                voice_settings=voice_settings,
                initial_text=initial_text,
                initial_request_in_upgrade=initial_request_in_upgrade,
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

    @property
    def speaker_id(self) -> str | None:
        return self._tts.speaker_id

    def _ensure(self) -> None:
        if self._loop is None:
            raise APIError("TTSSession is not started; use it as a context manager")

    def synthesize(
        self,
        text: str,
        *,
        response_id: str | None = None,
        max_length: int | None = None,
        with_timestamps: bool | None = None,
        voice_settings: VoiceSettings | None = None,
    ) -> None:
        self._ensure()
        self._run(
            self._tts.synthesize(
                text,
                response_id=response_id,
                max_length=max_length,
                with_timestamps=with_timestamps,
                voice_settings=voice_settings,
            )
        )

    def cancel(self) -> None:
        self._ensure()
        self._run(self._tts.cancel())
