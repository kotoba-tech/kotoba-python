"""ASR client.

Two transports exposed through the same facade:

- **REST** (``transcribe``): ``batch=False`` (default) is one
  ``POST /v1/speech-to-text`` with the transcript in the response, served by
  ``stt`` deployments (fal: ``kotoba-stt``). ``batch=True`` is the asynchronous
  job API — ``submit_job`` / ``get_job`` under the hood, polled until the job
  finishes — on self-hosted ``streaming_stt`` deployments only; not available
  on fal.
- **WebSocket** (``stream`` / ``transcribe_stream``): ``/v1/realtime`` on
  ``streaming_stt`` deployments (fal: ``kotoba-streaming-stt``); push audio
  chunks, receive partial transcripts. Best for live / latency-sensitive use.

On fal, only the WebSocket and ``transcribe()`` with ``batch=False`` are
available.

The default ``transcribe(path)`` uses REST; call ``stream(...)`` or
``transcribe_stream(...)`` explicitly for the WS path.
"""

from __future__ import annotations

import asyncio
import mimetypes
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Literal, overload

from kotoba._http import AsyncHttpSession, HttpSession, decode_json
from kotoba._pacing import apace, pace
from kotoba._rest_stt import FileFormat, SpeechToTextRequest, parse_speech_to_text
from kotoba._ws_asr import ASRSession, AsyncASRSession, AudioSource
from kotoba.errors import JobNotFoundError, TimeoutError, TranscriptionError
from kotoba.models import (
    JobIDResponse,
    JobState,
    JobStatus,
    ServerVAD,
    TranscriptionStylePreference,
    TranscriptResult,
)

DEFAULT_TIMEOUT = 1200.0  # 20 min; REST poll deadline
_WS_DEFAULT_SAMPLE_RATE = 24000
_WS_DEFAULT_CHUNK_MS = 200
_WS_DEFAULT_CHUNK_S = _WS_DEFAULT_CHUNK_MS / 1000.0


def _guess_content_type(path: Path) -> str:
    content_type, _ = mimetypes.guess_type(path.name)
    return content_type or "application/octet-stream"


def _job_data(
    language: str,
    with_timestamps: bool,
    style_preference: TranscriptionStylePreference | dict | None,
) -> dict[str, str]:
    """Build the transcription-job multipart form fields.

    ``style_preference`` is flattened into individual form fields (multipart
    can't carry nested objects); an unset preference contributes nothing, so
    the server applies its own default.
    """

    style_preference = TranscriptionStylePreference.coerce(style_preference)
    data = {"language": language, "with_timestamps": str(with_timestamps).lower()}
    if style_preference is not None:
        data.update(style_preference.to_wire())
    return data


# transcribe() arguments that exist in only one mode (True = batch, False = one-shot).
_MODE_ONLY = {
    "keywords": False,
    "file_format": False,
    "poll_interval": True,
    "poll_backoff": True,
    "max_poll_interval": True,
    "timeout": True,
}


def _check_mode_arguments(batch: bool, **given: Any) -> None:
    """Reject ``transcribe()`` arguments that belong to the other mode."""

    for name, value in given.items():
        if value is not None and _MODE_ONLY[name] != batch:
            raise ValueError(f"{name} is only valid with batch={_MODE_ONLY[name]}")


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
    ) -> None:
        self._http = http
        self._api_key = api_key

    # ---------- REST ------------------------------------------------------

    def submit_job(
        self,
        audio_file_path: str | Path,
        *,
        language: str = "ja",
        with_timestamps: bool = False,
        style_preference: TranscriptionStylePreference | dict | None = None,
    ) -> JobIDResponse:
        """POST an audio file, return the server-assigned job_id."""

        self._require_http()
        path = Path(audio_file_path)
        content_type = _guess_content_type(path)
        data = _job_data(language, with_timestamps, style_preference)
        with path.open("rb") as f:
            response = self._http.post(
                "/transcription_jobs",
                files={"file": (path.name, f, content_type)},
                data=data,
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

    @overload
    def transcribe(
        self,
        audio_file_path: str | Path,
        *,
        batch: Literal[False] = False,
        language: str | None = None,
        with_timestamps: bool = False,
        style_preference: TranscriptionStylePreference | dict | None = None,
        keywords: list[str] | None = None,
        file_format: FileFormat | None = None,
    ) -> TranscriptResult: ...

    @overload
    def transcribe(
        self,
        audio_file_path: str | Path,
        *,
        batch: Literal[True],
        language: str | None = None,
        with_timestamps: bool = False,
        style_preference: TranscriptionStylePreference | dict | None = None,
        poll_interval: float | None = None,
        poll_backoff: float | None = None,
        max_poll_interval: float | None = None,
        timeout: float | None = None,
    ) -> TranscriptResult: ...

    def transcribe(
        self,
        audio_file_path: str | Path,
        *,
        batch: bool = False,
        language: str | None = None,
        with_timestamps: bool = False,
        style_preference: TranscriptionStylePreference | dict | None = None,
        keywords: list[str] | None = None,
        file_format: FileFormat | None = None,
        poll_interval: float | None = None,
        poll_backoff: float | None = None,
        max_poll_interval: float | None = None,
        timeout: float | None = None,
    ) -> TranscriptResult:
        """Transcribe an audio file over REST.

        ``batch=False`` (default): one synchronous ``POST /v1/speech-to-text``
        with the transcript in the response — no job, no polling — served by
        ``stt`` deployments (fal: ``kotoba-stt``). The server's
        ``max_audio_seconds`` limit (120 s by default) applies, and the request
        runs under the client's per-request timeout (``KotobaClient(timeout=...)``),
        which long files or a cold deployment may need raised. ``language=None``
        uses the server's configured language; ``keywords``, ``style_preference``
        (kana) and ``with_timestamps`` must be enabled on the deployment.
        ``file_format="pcm_s16le_16"`` uploads raw 16-bit LE PCM at 16 kHz mono;
        ``"other"`` (default) is any encoded audio.

        ``batch=True``: the asynchronous job API — ``submit_job()``, then
        ``get_job()`` polled with exponential backoff until the job finishes.
        Self-hosted ``streaming_stt`` deployments only; not available on fal.
        ``language=None`` means ``"ja"``; ``timeout`` is the polling deadline
        (default 20 min) and ``poll_interval`` / ``poll_backoff`` /
        ``max_poll_interval`` shape the polling.

        An argument that belongs to the other mode raises ``ValueError`` before
        any request is sent. Both modes post relative to the session base URL
        (``KOTOBA_ASR_REST_URL``, which includes the ``/v1`` prefix) and return
        a :class:`TranscriptResult`; ``job_id`` is set only for ``batch=True``.
        """

        _check_mode_arguments(
            batch,
            keywords=keywords,
            file_format=file_format,
            poll_interval=poll_interval,
            poll_backoff=poll_backoff,
            max_poll_interval=max_poll_interval,
            timeout=timeout,
        )
        if batch:
            return self._transcribe_batch(
                audio_file_path,
                language=language or "ja",
                with_timestamps=with_timestamps,
                style_preference=style_preference,
                poll_interval=1.0 if poll_interval is None else poll_interval,
                poll_backoff=1.5 if poll_backoff is None else poll_backoff,
                max_poll_interval=10.0 if max_poll_interval is None else max_poll_interval,
                timeout=DEFAULT_TIMEOUT if timeout is None else timeout,
            )
        return self._transcribe_one_shot(
            audio_file_path,
            language=language,
            keywords=keywords,
            style_preference=style_preference,
            with_timestamps=with_timestamps,
            file_format=file_format or "other",
        )

    def _transcribe_one_shot(
        self,
        audio_file_path: str | Path,
        *,
        language: str | None,
        keywords: list[str] | None,
        style_preference: TranscriptionStylePreference | dict | None,
        with_timestamps: bool,
        file_format: FileFormat,
    ) -> TranscriptResult:
        self._require_http()
        request = SpeechToTextRequest.build(
            language=language,
            keywords=keywords,
            style_preference=style_preference,
            with_timestamps=with_timestamps,
            file_format=file_format,
        )
        path = Path(audio_file_path)
        with path.open("rb") as f:
            response = self._http.post(
                request.path,
                files={"file": (path.name, f, _guess_content_type(path))},
                data=request.form,
            )
        return parse_speech_to_text(decode_json(response))

    def _transcribe_batch(
        self,
        audio_file_path: str | Path,
        *,
        language: str,
        with_timestamps: bool,
        style_preference: TranscriptionStylePreference | dict | None,
        poll_interval: float,
        poll_backoff: float,
        max_poll_interval: float,
        timeout: float,
    ) -> TranscriptResult:
        job = self.submit_job(
            audio_file_path,
            language=language,
            with_timestamps=with_timestamps,
            style_preference=style_preference,
        )
        deadline = time.monotonic() + timeout
        interval = poll_interval

        while True:
            status = self.get_job(job.job_id)
            if status.state == JobState.done:
                return TranscriptResult(
                    text=status.transcription or "",
                    job_id=job.job_id,
                    # The job API answers [] when the deployment produced no
                    # timings; the one-shot mode says None, so use one convention.
                    segments=status.segments or None,
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

    # ---------- WebSocket -------------------------------------------------

    def stream(
        self,
        *,
        language: str = "ja",
        sample_rate: int = _WS_DEFAULT_SAMPLE_RATE,
        keywords: list[str] | None = None,
        style_preference: TranscriptionStylePreference | dict | None = None,
        turn_detection: ServerVAD | dict | Literal[False] | None = None,
        url: str | None = None,
    ) -> ASRSession:
        """Open a streaming ASR session. Caller drives send_audio / commit."""

        return ASRSession(
            _resolve_ws_url(url),
            language=language,
            sample_rate=sample_rate,
            keywords=keywords,
            style_preference=style_preference,
            turn_detection=turn_detection,
            api_key=self._api_key,
        )

    def _transcribe_file_ws(
        self,
        path: str | Path,
        *,
        language: str = "ja",
        sample_rate: int = _WS_DEFAULT_SAMPLE_RATE,
        keywords: list[str] | None = None,
        style_preference: TranscriptionStylePreference | dict | None = None,
        turn_detection: ServerVAD | dict | Literal[False] | None = None,
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
            style_preference=style_preference,
            turn_detection=turn_detection,
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
        style_preference: TranscriptionStylePreference | dict | None = None,
        turn_detection: ServerVAD | dict | Literal[False] | None = None,
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
            style_preference=style_preference,
            turn_detection=turn_detection,
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
    ) -> None:
        self._http = http
        self._api_key = api_key

    # ---------- REST ------------------------------------------------------

    async def submit_job(
        self,
        audio_file_path: str | Path,
        *,
        language: str = "ja",
        with_timestamps: bool = False,
        style_preference: TranscriptionStylePreference | dict | None = None,
    ) -> JobIDResponse:
        self._require_http()
        path = Path(audio_file_path)
        content_type = _guess_content_type(path)
        data_bytes = await asyncio.to_thread(path.read_bytes)
        data = _job_data(language, with_timestamps, style_preference)
        response = await self._http.post(
            "/transcription_jobs",
            files={"file": (path.name, data_bytes, content_type)},
            data=data,
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

    @overload
    async def transcribe(
        self,
        audio_file_path: str | Path,
        *,
        batch: Literal[False] = False,
        language: str | None = None,
        with_timestamps: bool = False,
        style_preference: TranscriptionStylePreference | dict | None = None,
        keywords: list[str] | None = None,
        file_format: FileFormat | None = None,
    ) -> TranscriptResult: ...

    @overload
    async def transcribe(
        self,
        audio_file_path: str | Path,
        *,
        batch: Literal[True],
        language: str | None = None,
        with_timestamps: bool = False,
        style_preference: TranscriptionStylePreference | dict | None = None,
        poll_interval: float | None = None,
        poll_backoff: float | None = None,
        max_poll_interval: float | None = None,
        timeout: float | None = None,
    ) -> TranscriptResult: ...

    async def transcribe(
        self,
        audio_file_path: str | Path,
        *,
        batch: bool = False,
        language: str | None = None,
        with_timestamps: bool = False,
        style_preference: TranscriptionStylePreference | dict | None = None,
        keywords: list[str] | None = None,
        file_format: FileFormat | None = None,
        poll_interval: float | None = None,
        poll_backoff: float | None = None,
        max_poll_interval: float | None = None,
        timeout: float | None = None,
    ) -> TranscriptResult:
        """Transcribe an audio file over REST.

        ``batch=False`` (default): one synchronous ``POST /v1/speech-to-text``
        with the transcript in the response — no job, no polling — served by
        ``stt`` deployments (fal: ``kotoba-stt``). The server's
        ``max_audio_seconds`` limit (120 s by default) applies, and the request
        runs under the client's per-request timeout (``KotobaClient(timeout=...)``),
        which long files or a cold deployment may need raised. ``language=None``
        uses the server's configured language; ``keywords``, ``style_preference``
        (kana) and ``with_timestamps`` must be enabled on the deployment.
        ``file_format="pcm_s16le_16"`` uploads raw 16-bit LE PCM at 16 kHz mono;
        ``"other"`` (default) is any encoded audio.

        ``batch=True``: the asynchronous job API — ``submit_job()``, then
        ``get_job()`` polled with exponential backoff until the job finishes.
        Self-hosted ``streaming_stt`` deployments only; not available on fal.
        ``language=None`` means ``"ja"``; ``timeout`` is the polling deadline
        (default 20 min) and ``poll_interval`` / ``poll_backoff`` /
        ``max_poll_interval`` shape the polling.

        An argument that belongs to the other mode raises ``ValueError`` before
        any request is sent. Both modes post relative to the session base URL
        (``KOTOBA_ASR_REST_URL``, which includes the ``/v1`` prefix) and return
        a :class:`TranscriptResult`; ``job_id`` is set only for ``batch=True``.
        """

        _check_mode_arguments(
            batch,
            keywords=keywords,
            file_format=file_format,
            poll_interval=poll_interval,
            poll_backoff=poll_backoff,
            max_poll_interval=max_poll_interval,
            timeout=timeout,
        )
        if batch:
            return await self._transcribe_batch(
                audio_file_path,
                language=language or "ja",
                with_timestamps=with_timestamps,
                style_preference=style_preference,
                poll_interval=1.0 if poll_interval is None else poll_interval,
                poll_backoff=1.5 if poll_backoff is None else poll_backoff,
                max_poll_interval=10.0 if max_poll_interval is None else max_poll_interval,
                timeout=DEFAULT_TIMEOUT if timeout is None else timeout,
            )
        return await self._transcribe_one_shot(
            audio_file_path,
            language=language,
            keywords=keywords,
            style_preference=style_preference,
            with_timestamps=with_timestamps,
            file_format=file_format or "other",
        )

    async def _transcribe_one_shot(
        self,
        audio_file_path: str | Path,
        *,
        language: str | None,
        keywords: list[str] | None,
        style_preference: TranscriptionStylePreference | dict | None,
        with_timestamps: bool,
        file_format: FileFormat,
    ) -> TranscriptResult:
        self._require_http()
        request = SpeechToTextRequest.build(
            language=language,
            keywords=keywords,
            style_preference=style_preference,
            with_timestamps=with_timestamps,
            file_format=file_format,
        )
        path = Path(audio_file_path)
        data_bytes = await asyncio.to_thread(path.read_bytes)
        response = await self._http.post(
            request.path,
            files={"file": (path.name, data_bytes, _guess_content_type(path))},
            data=request.form,
        )
        return parse_speech_to_text(decode_json(response))

    async def _transcribe_batch(
        self,
        audio_file_path: str | Path,
        *,
        language: str,
        with_timestamps: bool,
        style_preference: TranscriptionStylePreference | dict | None,
        poll_interval: float,
        poll_backoff: float,
        max_poll_interval: float,
        timeout: float,
    ) -> TranscriptResult:
        job = await self.submit_job(
            audio_file_path,
            language=language,
            with_timestamps=with_timestamps,
            style_preference=style_preference,
        )
        deadline = time.monotonic() + timeout
        interval = poll_interval

        while True:
            status = await self.get_job(job.job_id)
            if status.state == JobState.done:
                return TranscriptResult(
                    text=status.transcription or "",
                    job_id=job.job_id,
                    # The job API answers [] when the deployment produced no
                    # timings; the one-shot mode says None, so use one convention.
                    segments=status.segments or None,
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

    # ---------- WebSocket -------------------------------------------------

    def stream(
        self,
        *,
        language: str = "ja",
        sample_rate: int = _WS_DEFAULT_SAMPLE_RATE,
        keywords: list[str] | None = None,
        style_preference: TranscriptionStylePreference | dict | None = None,
        turn_detection: ServerVAD | dict | Literal[False] | None = None,
        url: str | None = None,
    ) -> AsyncASRSession:
        return AsyncASRSession(
            _resolve_ws_url(url),
            language=language,
            sample_rate=sample_rate,
            keywords=keywords,
            style_preference=style_preference,
            turn_detection=turn_detection,
            api_key=self._api_key,
        )

    async def _transcribe_file_ws(
        self,
        path: str | Path,
        *,
        language: str = "ja",
        sample_rate: int = _WS_DEFAULT_SAMPLE_RATE,
        keywords: list[str] | None = None,
        style_preference: TranscriptionStylePreference | dict | None = None,
        turn_detection: ServerVAD | dict | Literal[False] | None = None,
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
            style_preference=style_preference,
            turn_detection=turn_detection,
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
        style_preference: TranscriptionStylePreference | dict | None = None,
        turn_detection: ServerVAD | dict | Literal[False] | None = None,
        url: str | None = None,
    ) -> AsyncIterator[str]:
        async with self.stream(
            language=language,
            sample_rate=sample_rate,
            keywords=keywords,
            style_preference=style_preference,
            turn_detection=turn_detection,
            url=url,
        ) as session:
            feeder = asyncio.create_task(session.feed(audio))
            try:
                async for event in session:
                    if event.type == "partial_transcript" and event.text:
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
