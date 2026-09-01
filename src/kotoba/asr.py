"""ASR client.

Two transports exposed through the same facade:

- **REST** (``submit_job`` / ``get_job`` / ``transcribe``): POST an audio file
  and poll until the job finishes. Best for batch / non-interactive use.
- **WebSocket** (``stream`` / ``transcribe_stream``): push audio chunks and
  receive partial transcripts as they're produced. Best for live / latency-
  sensitive use.

The default ``transcribe(path)`` uses REST; call ``stream(...)`` or
``transcribe_stream(...)`` explicitly for the WS path.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import time
from contextlib import suppress
from pathlib import Path
from typing import AsyncIterator, Iterator

from kotoba._http import AsyncHttpSession, HttpSession
from kotoba._pacing import apace, pace
from kotoba._providers import ProviderConfig
from kotoba._ws_asr import AsyncASRSession, ASRSession, AudioSource
from kotoba.errors import JobNotFoundError, TimeoutError, TranscriptionError
from kotoba.models import JobIDResponse, JobState, JobStatus, TranscriptResult


DEFAULT_TIMEOUT = 1200.0  # 20 min; REST poll deadline

logger = logging.getLogger(__name__)
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

        No-op for providers without a probe policy; ``submit_job`` also
        probes automatically, so calling this is optional.
        """
        self._require_http()
        self._http.ensure_ready()

    def submit_job(
        self,
        audio_file_path: str | Path,
        *,
        language: str = "ja",
        with_timestamps: bool = False,
    ) -> JobIDResponse:
        """POST an audio file, return the server-assigned job_id."""

        self._require_http()
        self._http.ensure_ready()
        path = Path(audio_file_path)
        content_type = _guess_content_type(path)
        with path.open("rb") as f:
            response = self._http.post(
                "/transcription_jobs",
                files={"file": (path.name, f, content_type)},
                data={
                    "language": language,
                    "with_timestamps": str(with_timestamps).lower(),
                },
            )
        return JobIDResponse(**response.json())

    def get_job(self, job_id: str) -> JobStatus:
        """GET the status of a job. 202 → processing, 404 → JobNotFoundError."""

        self._require_http()
        response = self._http.get(
            f"/transcription_jobs/{job_id}",
            allow_statuses=(200, 202, 404),
        )
        if response.status_code == 202:
            return JobStatus(state=JobState.processing)
        if response.status_code == 404:
            raise JobNotFoundError(
                f"Job {job_id} not found", status_code=404, payload={}
            )
        return JobStatus(**response.json())

    def transcribe(
        self,
        audio_file_path: str | Path,
        *,
        language: str = "ja",
        with_timestamps: bool = False,
        poll_interval: float = 1.0,
        poll_backoff: float = 1.5,
        max_poll_interval: float = 10.0,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> TranscriptResult:
        """REST transcription. Returns the final transcript.

        On the kotoba provider this is submit + poll against the job API.
        On providers with a one-shot batch endpoint (fal), the file is sent
        in a single synchronous POST and ``timeout`` bounds that request;
        the polling parameters are unused.
        """

        self._require_http()
        one_shot = self._http.provider.batch_transcribe_path
        if one_shot is not None:
            return self._transcribe_one_shot(
                Path(audio_file_path),
                endpoint=one_shot,
                language=language,
                with_timestamps=with_timestamps,
                timeout=timeout,
            )

        job = self.submit_job(
            audio_file_path,
            language=language,
            with_timestamps=with_timestamps,
        )
        deadline = time.monotonic() + timeout
        interval = poll_interval

        while True:
            status = self.get_job(job.job_id)
            if status.state == JobState.done:
                return TranscriptResult(
                    text=status.transcription or "",
                    job_id=job.job_id,
                    segments=status.segments,
                )
            if status.state == JobState.error:
                raise TranscriptionError(
                    status.error_message or "Transcription failed",
                    payload={"job_id": job.job_id},
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Job {job.job_id} did not complete within {timeout}s",
                    payload={"job_id": job.job_id},
                )
            time.sleep(interval)
            interval = min(interval * poll_backoff, max_poll_interval)

    def _transcribe_one_shot(
        self,
        path: Path,
        *,
        endpoint: str,
        language: str,
        with_timestamps: bool,
        timeout: float,
    ) -> TranscriptResult:
        if with_timestamps:
            logger.warning(
                "with_timestamps is not supported by the one-shot %s endpoint; "
                "ignoring",
                endpoint,
            )
        self._http.ensure_ready()
        # The gateway consumes Authorization; xi-api-key passes through and
        # satisfies the app's own token requirement.
        headers = {"xi-api-key": self._api_key} if self._api_key else {}
        with path.open("rb") as f:
            response = self._http.post(
                endpoint,
                files={"file": (path.name, f, _guess_content_type(path))},
                data={"language": language, "file_format": "other"},
                headers=headers,
                timeout=timeout,
            )
        payload = response.json()
        return TranscriptResult(text=str(payload.get("text") or ""))

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
                "to enable transcribe() / submit_job() / get_job()."
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

    async def submit_job(
        self,
        audio_file_path: str | Path,
        *,
        language: str = "ja",
        with_timestamps: bool = False,
    ) -> JobIDResponse:
        self._require_http()
        await self._http.ensure_ready()
        path = Path(audio_file_path)
        content_type = _guess_content_type(path)
        data_bytes = await asyncio.to_thread(path.read_bytes)
        response = await self._http.post(
            "/transcription_jobs",
            files={"file": (path.name, data_bytes, content_type)},
            data={
                "language": language,
                "with_timestamps": str(with_timestamps).lower(),
            },
        )
        return JobIDResponse(**response.json())

    async def get_job(self, job_id: str) -> JobStatus:
        self._require_http()
        response = await self._http.get(
            f"/transcription_jobs/{job_id}",
            allow_statuses=(200, 202, 404),
        )
        if response.status_code == 202:
            return JobStatus(state=JobState.processing)
        if response.status_code == 404:
            raise JobNotFoundError(
                f"Job {job_id} not found", status_code=404, payload={}
            )
        return JobStatus(**response.json())

    async def transcribe(
        self,
        audio_file_path: str | Path,
        *,
        language: str = "ja",
        with_timestamps: bool = False,
        poll_interval: float = 1.0,
        poll_backoff: float = 1.5,
        max_poll_interval: float = 10.0,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> TranscriptResult:
        """Async mirror of :meth:`ASRClient.transcribe` (same dispatch)."""

        self._require_http()
        one_shot = self._http.provider.batch_transcribe_path
        if one_shot is not None:
            return await self._transcribe_one_shot(
                Path(audio_file_path),
                endpoint=one_shot,
                language=language,
                with_timestamps=with_timestamps,
                timeout=timeout,
            )

        job = await self.submit_job(
            audio_file_path,
            language=language,
            with_timestamps=with_timestamps,
        )
        deadline = time.monotonic() + timeout
        interval = poll_interval

        while True:
            status = await self.get_job(job.job_id)
            if status.state == JobState.done:
                return TranscriptResult(
                    text=status.transcription or "",
                    job_id=job.job_id,
                    segments=status.segments,
                )
            if status.state == JobState.error:
                raise TranscriptionError(
                    status.error_message or "Transcription failed",
                    payload={"job_id": job.job_id},
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Job {job.job_id} did not complete within {timeout}s",
                    payload={"job_id": job.job_id},
                )
            await asyncio.sleep(interval)
            interval = min(interval * poll_backoff, max_poll_interval)

    async def _transcribe_one_shot(
        self,
        path: Path,
        *,
        endpoint: str,
        language: str,
        with_timestamps: bool,
        timeout: float,
    ) -> TranscriptResult:
        if with_timestamps:
            logger.warning(
                "with_timestamps is not supported by the one-shot %s endpoint; "
                "ignoring",
                endpoint,
            )
        await self._http.ensure_ready()
        headers = {"xi-api-key": self._api_key} if self._api_key else {}
        data_bytes = await asyncio.to_thread(path.read_bytes)
        response = await self._http.post(
            endpoint,
            files={"file": (path.name, data_bytes, _guess_content_type(path))},
            data={"language": language, "file_format": "other"},
            headers=headers,
            timeout=timeout,
        )
        payload = response.json()
        return TranscriptResult(text=str(payload.get("text") or ""))

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
