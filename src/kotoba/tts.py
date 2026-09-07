"""High-level TTS facade.

Three surfaces:

- `stream(...)` returns a session the caller drives manually
  (`session.synthesize(text)` then iterate events). Use this when you need
  direct access to the underlying protocol (e.g., to issue ``cancel()``).
- `synthesize_stream(text)` accepts a plain ``str`` and yields raw PCM audio
  chunks as they arrive from the server. Audio streaming is server→client
  only — text input is sent in one frame.
- `synthesize(text)` is a batch convenience that collects
  `synthesize_stream` into one `AudioResult`.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Iterator

from kotoba._ws_tts import AsyncTTSSession, TTSSession
from kotoba.models import AudioResult
from kotoba.routing import endpoint_for


def _resolve_url(url: str | None, language: str) -> str:
    if url is not None:
        return url
    return endpoint_for("tts", None, language)


def _content_type_for(sample_rate: int, audio_format: str) -> str:
    """MIME type for a negotiated TTS output format.

    mu-law (Twilio) is served as ``audio/basic`` and Opus as ``audio/ogg``;
    PCM formats keep the rate/encoding-tagged ``audio/pcm`` form.
    """

    fmt = audio_format.lower()
    if fmt in ("mulaw", "ulaw", "twilio"):
        return "audio/basic"
    if fmt == "opus":
        return "audio/ogg"
    return f"audio/pcm;rate={sample_rate};encoding={audio_format}"


class TTSClient:
    """Sync TTS client."""

    def __init__(self, api_key: str | None) -> None:
        self._api_key = api_key

    def stream(
        self,
        *,
        language: str = "ja",
        speaker_id: str | None = None,
        spk_ref_audio_tokens: Any = None,
        audio_format: str | None = None,
        sample_rate: int | None = None,
        url: str | None = None,
    ) -> TTSSession:
        return TTSSession(
            _resolve_url(url, language),
            language=language,
            speaker_id=speaker_id,
            spk_ref_audio_tokens=spk_ref_audio_tokens,
            audio_format=audio_format,
            sample_rate=sample_rate,
            api_key=self._api_key,
        )

    def synthesize_stream(
        self,
        text: str,
        *,
        language: str = "ja",
        speaker_id: str | None = None,
        spk_ref_audio_tokens: Any = None,
        audio_format: str | None = None,
        sample_rate: int | None = None,
        url: str | None = None,
    ) -> Iterator[bytes]:
        """Yield PCM audio chunks for ``text`` as the server emits them."""

        session = self.stream(
            language=language,
            speaker_id=speaker_id,
            spk_ref_audio_tokens=spk_ref_audio_tokens,
            audio_format=audio_format,
            sample_rate=sample_rate,
            url=url,
        )
        with session:
            session.synthesize(text)
            for event in session:
                if event.type == "audio_chunk" and event.audio:
                    yield event.audio
                elif event.type == "done":
                    break

    def synthesize(
        self,
        text: str,
        *,
        language: str = "ja",
        speaker_id: str | None = None,
        audio_format: str | None = None,
        sample_rate: int | None = None,
        url: str | None = None,
    ) -> AudioResult:
        chunks: list[bytes] = []
        negotiated_rate = 24000
        negotiated_format = "pcm_f32"
        session = self.stream(
            language=language,
            speaker_id=speaker_id,
            audio_format=audio_format,
            sample_rate=sample_rate,
            url=url,
        )
        with session:
            session.synthesize(text)
            negotiated_rate = session.sample_rate
            negotiated_format = session.audio_format
            for event in session:
                if event.type == "audio_chunk" and event.audio:
                    chunks.append(event.audio)
                elif event.type == "done":
                    break
        return AudioResult(
            data=b"".join(chunks),
            sample_rate=negotiated_rate,
            audio_format=negotiated_format,
            content_type=_content_type_for(negotiated_rate, negotiated_format),
        )


class AsyncTTSClient:
    """Async TTS client."""

    def __init__(self, api_key: str | None) -> None:
        self._api_key = api_key

    def stream(
        self,
        *,
        language: str = "ja",
        speaker_id: str | None = None,
        spk_ref_audio_tokens: Any = None,
        audio_format: str | None = None,
        sample_rate: int | None = None,
        url: str | None = None,
    ) -> AsyncTTSSession:
        return AsyncTTSSession(
            _resolve_url(url, language),
            language=language,
            speaker_id=speaker_id,
            spk_ref_audio_tokens=spk_ref_audio_tokens,
            audio_format=audio_format,
            sample_rate=sample_rate,
            api_key=self._api_key,
        )

    async def synthesize_stream(
        self,
        text: str,
        *,
        language: str = "ja",
        speaker_id: str | None = None,
        spk_ref_audio_tokens: Any = None,
        audio_format: str | None = None,
        sample_rate: int | None = None,
        url: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Yield PCM audio chunks for ``text`` as the server emits them."""

        async with self.stream(
            language=language,
            speaker_id=speaker_id,
            spk_ref_audio_tokens=spk_ref_audio_tokens,
            audio_format=audio_format,
            sample_rate=sample_rate,
            url=url,
        ) as session:
            await session.synthesize(text)
            async for event in session:
                if event.type == "audio_chunk" and event.audio:
                    yield event.audio
                elif event.type == "done":
                    break

    async def synthesize(
        self,
        text: str,
        *,
        language: str = "ja",
        speaker_id: str | None = None,
        audio_format: str | None = None,
        sample_rate: int | None = None,
        url: str | None = None,
    ) -> AudioResult:
        chunks: list[bytes] = []
        negotiated_rate = 24000
        negotiated_format = "pcm_f32"
        async with self.stream(
            language=language,
            speaker_id=speaker_id,
            audio_format=audio_format,
            sample_rate=sample_rate,
            url=url,
        ) as session:
            await session.synthesize(text)
            negotiated_rate = session.sample_rate
            negotiated_format = session.audio_format
            async for event in session:
                if event.type == "audio_chunk" and event.audio:
                    chunks.append(event.audio)
                elif event.type == "done":
                    break
        return AudioResult(
            data=b"".join(chunks),
            sample_rate=negotiated_rate,
            audio_format=negotiated_format,
            content_type=_content_type_for(negotiated_rate, negotiated_format),
        )
