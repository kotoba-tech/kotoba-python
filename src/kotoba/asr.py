"""ASR client.

Two transports exposed through the same facade:

- **REST** (``transcribe``): one synchronous ``POST /v1/speech-to-text``
  with the audio file. Best for batch / non-interactive use.
- **WebSocket** (``stream`` / ``transcribe_stream``): push audio chunks and
  receive partial transcripts as they're produced. Best for live / latency-
  sensitive use.

The default ``transcribe(path)`` uses REST; call ``stream(...)`` or
``transcribe_stream(...)`` explicitly for the WS path.
"""

from __future__ import annotations

import asyncio
import mimetypes
from contextlib import suppress
from pathlib import Path
from typing import AsyncIterator, Iterator

from kotoba._http import AsyncHttpSession, HttpSession
from kotoba._pacing import apace, pace
from kotoba._providers import ProviderConfig
from kotoba._ws_asr import AsyncASRSession, ASRSession, AudioSource
from kotoba.models import TranscriptResult


DEFAULT_TIMEOUT = 1200.0  # 20 min; bound on the one-shot REST request
_WS_DEFAULT_SAMPLE_RATE = 24000
_WS_DEFAULT_CHUNK_MS = 200
_WS_DEFAULT_CHUNK_S = _WS_DEFAULT_CHUNK_MS / 1000.0


def _guess_content_type(path: Path) -> str:
    content_type, _ = mimetypes.guess_type(path.name)
    return content_type or "application/octet-stream"


def _resolve_ws_url(explicit: str | None) -> str:
    """Resolve the WS URL, falling back to the routing table."""

    if explicit is not None:
        return explicit
    from kotoba.routing import endpoint_for

    return endpoint_for("asr", None, None)


def _load_and_resample_pcm16(path: str | Path, target_rate: int) -> bytes:
    from kotoba.audio import load_mono_pcm16_wav, resample_mono_pcm16

    audio, sample_rate = load_mono_pcm16_wav(path)
    if sample_rate != target_rate:
        audio = resample_mono_pcm16(audio, sample_rate, target_rate)
    return audio.astype("<i2").tobytes()


def _chunk_iter(pcm16: bytes, sample_rate: int, chunk_ms: int = _WS_DEFAULT_CHUNK_MS):
    chunk_bytes = int(sample_rate * (chunk_ms / 1000.0)) * 2  # int16
    for i in range(0, len(pcm16), chunk_bytes):
        yield pcm16[i : i + chunk_bytes]


def _batch_endpoint(http: HttpSession | AsyncHttpSession) -> str:
    """Resolve the one-shot endpoint path against the session's base URL.

    The path carries its own version prefix (``/v1/...``); a base URL that
    already ends with that prefix (the older documented convention) is not
    doubled.
    """

    path = http.provider.batch_transcribe_path
    prefix = "/" + path.strip("/").split("/", 1)[0]
    if http.base_url.endswith(prefix) and path.startswith(prefix + "/"):
        return path[len(prefix):]
    return path


def _app_key_headers(api_key: str | None) -> dict[str, str]:
    # The fal gateway consumes Authorization; xi-api-key passes through and
    # satisfies the app's own token requirement. Harmless elsewhere.
    return {"xi-api-key": api_key} if api_key else {}


def _parse_transcript(payload: dict) -> TranscriptResult:
    # Everything but the text (e.g. audio_duration_secs) rides along as metadata.
    metadata = {k: v for k, v in payload.items() if k != "text"}
    return TranscriptResult(text=str(payload.get("text") or ""), metadata=metadata)


class ASRClient:
    """Sync client exposing both REST and WebSocket ASR entry points."""

    def __init__(
        self,
        http: HttpSession | None = None,
        *,
        api_key: str | None = None,
        provider: str | ProviderConfig | None = None,
    ) -> None:
        self._http = http
        self._api_key = api_key
        self._provider = provider

    # ---------- REST ------------------------------------------------------

    def warmup(self) -> None:
        """Wait until the REST endpoint is ready to serve (cold start).

        No-op for providers without a probe policy; ``transcribe`` also
        probes automatically, so calling this is optional.
        """
        self._require_http()
        self._http.ensure_ready()

    def transcribe(
        self,
        audio_file_path: str | Path,
        *,
        language: str = "ja",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> TranscriptResult:
        """Transcribe a file in one synchronous ``POST /v1/speech-to-text``.

        ``timeout`` bounds the single request (long files take a while on
        the server). On cold-start providers a readiness probe runs first.
        """

        self._require_http()
        self._http.ensure_ready()
        path = Path(audio_file_path)
        with path.open("rb") as f:
            response = self._http.post(
                _batch_endpoint(self._http),
                files={"file": (path.name, f, _guess_content_type(path))},
                data={"language": language, "file_format": "other"},
                headers=_app_key_headers(self._api_key),
                timeout=timeout,
            )
        return _parse_transcript(response.json())

    # ---------- WebSocket -------------------------------------------------

    def stream(
        self,
        *,
        language: str = "ja",
        sample_rate: int = _WS_DEFAULT_SAMPLE_RATE,
        keywords: list[str] | None = None,
        url: str | None = None,
    ) -> ASRSession:
        """Open a streaming ASR session. Caller drives send_audio / commit."""

        return ASRSession(
            _resolve_ws_url(url),
            language=language,
            sample_rate=sample_rate,
            keywords=keywords,
            api_key=self._api_key,
            provider=self._provider,
        )

    def _transcribe_file_ws(
        self,
        path: str | Path,
        *,
        language: str = "ja",
        sample_rate: int = _WS_DEFAULT_SAMPLE_RATE,
        keywords: list[str] | None = None,
        url: str | None = None,
    ) -> TranscriptResult:
        """Internal helper; not part of the documented public API.

        Prefer ``transcribe(path)`` (REST) for batch or
        ``transcribe_stream(iter)`` for live.
        """

        pcm16 = _load_and_resample_pcm16(path, sample_rate)
        parts: list[str] = []
        with self.stream(
            language=language,
            sample_rate=sample_rate,
            keywords=keywords,
            url=url,
        ) as session:
            for chunk in pace(
                _chunk_iter(pcm16, sample_rate), _WS_DEFAULT_CHUNK_S
            ):
                session.send_audio(chunk)
            session.commit()
            for event in session:
                if event.type == "partial_transcript" and event.text:
                    parts.append(event.text)
                elif event.type == "final_transcript" and event.text:
                    parts = [event.text]
                elif event.type == "done":
                    break
        return TranscriptResult(text="".join(parts))

    def transcribe_stream(
        self,
        audio: AudioSource,
        *,
        language: str = "ja",
        sample_rate: int = _WS_DEFAULT_SAMPLE_RATE,
        keywords: list[str] | None = None,
        url: str | None = None,
    ) -> Iterator[str]:
        """Yield transcript deltas for a streaming pcm16 source.

        ``audio`` may be a sync iterator (file chunks, mic queue drain) or an
        async iterator. The feeder is scheduled on the session's background
        loop so deltas surface as soon as the server emits them.
        """

        session = self.stream(
            language=language,
            sample_rate=sample_rate,
            keywords=keywords,
            url=url,
        )
        with session:
            async_session = session._asr
            loop = session._loop
            assert loop is not None
            feeder = asyncio.run_coroutine_threadsafe(
                async_session.feed(audio), loop
            )
            try:
                for event in session:
                    if event.type == "partial_transcript" and event.text:
                        yield event.text
                    elif event.type == "final_transcript" and event.text:
                        yield event.text
                    elif event.type == "done":
                        break
                feeder.result()
            finally:
                if not feeder.done():
                    feeder.cancel()
                    with suppress(Exception):
                        feeder.result(timeout=1.0)

    # ---------- helpers ---------------------------------------------------

    def _require_http(self) -> None:
        if self._http is None:
            raise RuntimeError(
                "REST endpoint not configured. Pass url=... to KotobaClient "
                "to enable transcribe()."
            )


class AsyncASRClient:
    """Async counterpart to :class:`ASRClient`."""

    def __init__(
        self,
        http: AsyncHttpSession | None = None,
        *,
        api_key: str | None = None,
        provider: str | ProviderConfig | None = None,
    ) -> None:
        self._http = http
        self._api_key = api_key
        self._provider = provider

    # ---------- REST ------------------------------------------------------

    async def warmup(self) -> None:
        """Async mirror of :meth:`ASRClient.warmup`."""
        self._require_http()
        await self._http.ensure_ready()

    async def transcribe(
        self,
        audio_file_path: str | Path,
        *,
        language: str = "ja",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> TranscriptResult:
        """Async mirror of :meth:`ASRClient.transcribe`."""

        self._require_http()
        await self._http.ensure_ready()
        path = Path(audio_file_path)
        data_bytes = await asyncio.to_thread(path.read_bytes)
        response = await self._http.post(
            _batch_endpoint(self._http),
            files={"file": (path.name, data_bytes, _guess_content_type(path))},
            data={"language": language, "file_format": "other"},
            headers=_app_key_headers(self._api_key),
            timeout=timeout,
        )
        return _parse_transcript(response.json())

    # ---------- WebSocket -------------------------------------------------

    def stream(
        self,
        *,
        language: str = "ja",
        sample_rate: int = _WS_DEFAULT_SAMPLE_RATE,
        keywords: list[str] | None = None,
        url: str | None = None,
    ) -> AsyncASRSession:
        return AsyncASRSession(
            _resolve_ws_url(url),
            language=language,
            sample_rate=sample_rate,
            keywords=keywords,
            api_key=self._api_key,
            provider=self._provider,
        )

    async def _transcribe_file_ws(
        self,
        path: str | Path,
        *,
        language: str = "ja",
        sample_rate: int = _WS_DEFAULT_SAMPLE_RATE,
        keywords: list[str] | None = None,
        url: str | None = None,
    ) -> TranscriptResult:
        """Internal helper; not part of the documented public API.

        Prefer ``transcribe(path)`` (REST) for batch or
        ``transcribe_stream(iter)`` for live.
        """
        pcm16 = _load_and_resample_pcm16(path, sample_rate)
        parts: list[str] = []
        async with self.stream(
            language=language,
            sample_rate=sample_rate,
            keywords=keywords,
            url=url,
        ) as session:
            async for chunk in apace(
                _chunk_iter(pcm16, sample_rate), _WS_DEFAULT_CHUNK_S
            ):
                await session.send_audio(chunk)
            await session.commit()
            async for event in session:
                if event.type == "partial_transcript" and event.text:
                    parts.append(event.text)
                elif event.type == "final_transcript" and event.text:
                    parts = [event.text]
                elif event.type == "done":
                    break
        return TranscriptResult(text="".join(parts))

    async def transcribe_stream(
        self,
        audio: AudioSource,
        *,
        language: str = "ja",
        sample_rate: int = _WS_DEFAULT_SAMPLE_RATE,
        keywords: list[str] | None = None,
        url: str | None = None,
    ) -> AsyncIterator[str]:
        async with self.stream(
            language=language,
            sample_rate=sample_rate,
            keywords=keywords,
            url=url,
        ) as session:
            feeder = asyncio.create_task(session.feed(audio))
            try:
                async for event in session:
                    if event.type == "partial_transcript" and event.text:
                        yield event.text
                    elif event.type == "final_transcript" and event.text:
                        yield event.text
                    elif event.type == "done":
                        break
                await feeder
            finally:
                if not feeder.done():
                    feeder.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await feeder

    # ---------- helpers ---------------------------------------------------

    def _require_http(self) -> None:
        if self._http is None:
            raise RuntimeError(
                "REST endpoint not configured. Pass url=... to AsyncKotobaClient."
            )
