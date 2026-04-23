"""High-level TTS facade.

Three surfaces:

- `stream(...)` returns a session the caller drives manually
  (`start_response` / `append_text` / `commit` / iterate events). Use this
  when you need direct access to the underlying protocol.
- `synthesize_stream(text)` accepts `str | Iterable[str] | AsyncIterable[str]`
  (e.g. an LLM token generator) and yields raw PCM audio chunks as they
  arrive. Text-feed and audio-drain run concurrently so first-audio latency
  is not blocked on the text source finishing.
- `synthesize(text)` is a batch convenience that collects
  `synthesize_stream` into one `AudioResult` — the "5-line hello world"
  shape. Also accepts a generator.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any, AsyncIterator, Iterator

from kotoba._ws_tts import AsyncTTSSession, TTSSession
from kotoba.models import AudioResult, TextSource
from kotoba.routing import endpoint_for


def _resolve_url(url: str | None, language: str) -> str:
    if url is not None:
        return url
    return endpoint_for("tts", None, language)


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
        url: str | None = None,
    ) -> TTSSession:
        return TTSSession(
            _resolve_url(url, language),
            language=language,
            speaker_id=speaker_id,
            spk_ref_audio_tokens=spk_ref_audio_tokens,
            api_key=self._api_key,
        )

    def synthesize_stream(
        self,
        text: TextSource,
        *,
        language: str = "ja",
        speaker_id: str | None = None,
        spk_ref_audio_tokens: Any = None,
        url: str | None = None,
    ) -> Iterator[bytes]:
        """Yield PCM audio chunks for ``text``.

        ``text`` may be a string, a sync generator (LLM SSE style), or an
        async generator. The feeder is scheduled on the session's
        background event loop so audio chunks surface as soon as the
        server emits them — not after the generator is exhausted.
        """

        session = self.stream(
            language=language,
            speaker_id=speaker_id,
            spk_ref_audio_tokens=spk_ref_audio_tokens,
            url=url,
        )
        with session:
            async_session = session._tts
            loop = session._loop
            assert loop is not None
            feeder = asyncio.run_coroutine_threadsafe(
                async_session.feed(text), loop
            )
            try:
                for event in session:
                    if event.type == "audio_chunk" and event.audio:
                        yield event.audio
                    elif event.type == "done":
                        break
                feeder.result()
            finally:
                if not feeder.done():
                    feeder.cancel()
                    with suppress(Exception):
                        asyncio.run_coroutine_threadsafe(
                            async_session.cancel(), loop
                        ).result(timeout=1.0)

    def synthesize(
        self,
        text: TextSource,
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
            sample_rate = session.sample_rate
            audio_format = session.audio_format  # type: ignore[assignment]
            async_session = session._tts
            loop = session._loop
            assert loop is not None
            feeder = asyncio.run_coroutine_threadsafe(
                async_session.feed(text), loop
            )
            try:
                for event in session:
                    if event.type == "audio_chunk" and event.audio:
                        chunks.append(event.audio)
                    elif event.type == "done":
                        break
                feeder.result()
            finally:
                if not feeder.done():
                    feeder.cancel()
        return AudioResult(
            data=b"".join(chunks),
            sample_rate=sample_rate,
            audio_format=audio_format,  # type: ignore[arg-type]
            content_type=f"audio/pcm;rate={sample_rate};encoding={audio_format}",
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
        url: str | None = None,
    ) -> AsyncTTSSession:
        return AsyncTTSSession(
            _resolve_url(url, language),
            language=language,
            speaker_id=speaker_id,
            spk_ref_audio_tokens=spk_ref_audio_tokens,
            api_key=self._api_key,
        )

    async def synthesize_stream(
        self,
        text: TextSource,
        *,
        language: str = "ja",
        speaker_id: str | None = None,
        spk_ref_audio_tokens: Any = None,
        url: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Yield PCM audio chunks for ``text``.

        ``text`` may be a string, a sync generator, or an async generator.
        The feeder runs as a concurrent task so the first ``audio.chunk``
        surfaces as soon as the server emits it.
        """

        async with self.stream(
            language=language,
            speaker_id=speaker_id,
            spk_ref_audio_tokens=spk_ref_audio_tokens,
            url=url,
        ) as session:
            feeder = asyncio.create_task(session.feed(text))
            try:
                async for event in session:
                    if event.type == "audio_chunk" and event.audio:
                        yield event.audio
                    elif event.type == "done":
                        break
                await feeder
            finally:
                if not feeder.done():
                    feeder.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await feeder

    async def synthesize(
        self,
        text: TextSource,
        *,
        language: str = "ja",
        speaker_id: str | None = None,
        url: str | None = None,
    ) -> AudioResult:
        chunks: list[bytes] = []
        sample_rate = 24000
        audio_format = "pcm_f32"
        async with self.stream(language=language, speaker_id=speaker_id, url=url) as session:
            sample_rate = session.sample_rate
            audio_format = session.audio_format
            feeder = asyncio.create_task(session.feed(text))
            try:
                async for event in session:
                    if event.type == "audio_chunk" and event.audio:
                        chunks.append(event.audio)
                    elif event.type == "done":
                        break
                await feeder
            finally:
                if not feeder.done():
                    feeder.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await feeder
        return AudioResult(
            data=b"".join(chunks),
            sample_rate=sample_rate,
            audio_format=audio_format,  # type: ignore[arg-type]
            content_type=f"audio/pcm;rate={sample_rate};encoding={audio_format}",
        )
