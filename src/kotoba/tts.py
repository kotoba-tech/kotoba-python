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

from kotoba._providers import ProviderConfig
from kotoba._ws_tts import AsyncTTSSession, TTSSession
from kotoba.models import AudioResult
from kotoba.routing import endpoint_for


def _resolve_url(url: str | None, language: str) -> str:
    if url is not None:
        return url
    return endpoint_for("tts", None, language)


class TTSClient:
    """Sync TTS client."""

    def __init__(
        self,
        api_key: str | None,
        *,
        provider: str | ProviderConfig | None = None,
    ) -> None:
        self._api_key = api_key
        self._provider = provider

    def stream(
        self,
        *,
        language: str = "ja",
        speaker_id: str | None = None,
        spk_ref_audio_tokens: Any = None,
        url: str | None = None,
    ) -> TTSSession:
        return TTSSession(
            _resolve_url(url, language),
            language=language,
            speaker_id=speaker_id,
            spk_ref_audio_tokens=spk_ref_audio_tokens,
            api_key=self._api_key,
            provider=self._provider,
        )

    def synthesize_stream(
        self,
        text: str,
        *,
        language: str = "ja",
        speaker_id: str | None = None,
        spk_ref_audio_tokens: Any = None,
        url: str | None = None,
    ) -> Iterator[bytes]:
        """Yield PCM audio chunks for ``text`` as the server emits them."""

        session = self.stream(
            language=language,
            speaker_id=speaker_id,
            spk_ref_audio_tokens=spk_ref_audio_tokens,
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
        url: str | None = None,
    ) -> AudioResult:
        chunks: list[bytes] = []
        sample_rate = 24000
        audio_format = "pcm_f32"
        session = self.stream(language=language, speaker_id=speaker_id, url=url)
        with session:
            session.synthesize(text)
            sample_rate = session.sample_rate
            audio_format = session.audio_format  # type: ignore[assignment]
            for event in session:
                if event.type == "audio_chunk" and event.audio:
                    chunks.append(event.audio)
                elif event.type == "done":
                    break
        return AudioResult(
            data=b"".join(chunks),
            sample_rate=sample_rate,
            audio_format=audio_format,  # type: ignore[arg-type]
            content_type=f"audio/pcm;rate={sample_rate};encoding={audio_format}",
        )


class AsyncTTSClient:
    """Async TTS client."""

    def __init__(
        self,
        api_key: str | None,
        *,
        provider: str | ProviderConfig | None = None,
    ) -> None:
        self._api_key = api_key
        self._provider = provider

    def stream(
        self,
        *,
        language: str = "ja",
        speaker_id: str | None = None,
        spk_ref_audio_tokens: Any = None,
        url: str | None = None,
    ) -> AsyncTTSSession:
        return AsyncTTSSession(
            _resolve_url(url, language),
            language=language,
            speaker_id=speaker_id,
            spk_ref_audio_tokens=spk_ref_audio_tokens,
            api_key=self._api_key,
            provider=self._provider,
        )

    async def synthesize_stream(
        self,
        text: str,
        *,
        language: str = "ja",
        speaker_id: str | None = None,
        spk_ref_audio_tokens: Any = None,
        url: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Yield PCM audio chunks for ``text`` as the server emits them."""

        async with self.stream(
            language=language,
            speaker_id=speaker_id,
            spk_ref_audio_tokens=spk_ref_audio_tokens,
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
        url: str | None = None,
    ) -> AudioResult:
        chunks: list[bytes] = []
        sample_rate = 24000
        audio_format = "pcm_f32"
        async with self.stream(language=language, speaker_id=speaker_id, url=url) as session:
            await session.synthesize(text)
            sample_rate = session.sample_rate
            audio_format = session.audio_format
            async for event in session:
                if event.type == "audio_chunk" and event.audio:
                    chunks.append(event.audio)
                elif event.type == "done":
                    break
        return AudioResult(
            data=b"".join(chunks),
            sample_rate=sample_rate,
            audio_format=audio_format,  # type: ignore[arg-type]
            content_type=f"audio/pcm;rate={sample_rate};encoding={audio_format}",
        )
