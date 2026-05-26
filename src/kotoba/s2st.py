"""High-level S2S translation facade.

Same shape as TTS: `stream(...)` for incremental, `translate(path)` for batch.
"""

from __future__ import annotations

from pathlib import Path

from kotoba._pacing import apace, pace
from kotoba._ws_s2st import AsyncS2STSession, S2STSession
from kotoba.audio import load_mono_pcm16_wav, resample_mono_pcm16
from kotoba.models import S2STResult
from kotoba.routing import endpoint_for

_SESSION_SAMPLE_RATE = 24000
_DEFAULT_CHUNK_MS = 40
_DEFAULT_CHUNK_S = _DEFAULT_CHUNK_MS / 1000.0


def _resolve_url(url: str | None, src: str, tgt: str) -> str:
    if url is not None:
        return url
    return endpoint_for("s2st", src, tgt)


def _load_and_resample(path: str | Path) -> bytes:
    audio, sample_rate = load_mono_pcm16_wav(path)
    if sample_rate != _SESSION_SAMPLE_RATE:
        audio = resample_mono_pcm16(audio, sample_rate, _SESSION_SAMPLE_RATE)
    return audio.astype("<i2").tobytes()


def _chunk_iter(pcm16: bytes, chunk_ms: int = _DEFAULT_CHUNK_MS):
    chunk_bytes = int(_SESSION_SAMPLE_RATE * (chunk_ms / 1000.0)) * 2  # int16
    for i in range(0, len(pcm16), chunk_bytes):
        yield pcm16[i : i + chunk_bytes]


class S2STClient:
    """Sync S2S translation client."""

    def __init__(self, api_key: str | None) -> None:
        self._api_key = api_key

    def stream(
        self,
        *,
        src: str,
        tgt: str,
        sample_rate: int = _SESSION_SAMPLE_RATE,
        delay: int | None = None,
        url: str | None = None,
    ) -> S2STSession:
        return S2STSession(
            _resolve_url(url, src, tgt),
            src_language=src,
            tgt_language=tgt,
            sample_rate=sample_rate,
            delay=delay,
            api_key=self._api_key,
        )

    def translate(
        self,
        path: str | Path,
        *,
        src: str,
        tgt: str,
        delay: int | None = None,
        url: str | None = None,
    ) -> S2STResult:
        pcm16 = _load_and_resample(path)
        audio_chunks: list[bytes] = []
        source_text: list[str] = []

        with self.stream(src=src, tgt=tgt, delay=delay, url=url) as session:
            for chunk in pace(_chunk_iter(pcm16), _DEFAULT_CHUNK_S):
                session.send_audio(chunk)
            session.commit()
            for event in session:
                if event.type == "audio_chunk" and event.audio:
                    audio_chunks.append(event.audio)
                elif event.type == "partial_transcript" and event.text:
                    source_text.append(event.text)
                elif event.type == "done":
                    break

        return S2STResult(
            audio=b"".join(audio_chunks),
            sample_rate=_SESSION_SAMPLE_RATE,
            audio_format="pcm16",
            transcript_source="".join(source_text) or None,
            transcript_target=None,
        )


class AsyncS2STClient:
    """Async S2S translation client."""

    def __init__(self, api_key: str | None) -> None:
        self._api_key = api_key

    def stream(
        self,
        *,
        src: str,
        tgt: str,
        sample_rate: int = _SESSION_SAMPLE_RATE,
        delay: int | None = None,
        url: str | None = None,
    ) -> AsyncS2STSession:
        return AsyncS2STSession(
            _resolve_url(url, src, tgt),
            src_language=src,
            tgt_language=tgt,
            sample_rate=sample_rate,
            delay=delay,
            api_key=self._api_key,
        )

    async def translate(
        self,
        path: str | Path,
        *,
        src: str,
        tgt: str,
        delay: int | None = None,
        url: str | None = None,
    ) -> S2STResult:
        pcm16 = _load_and_resample(path)
        audio_chunks: list[bytes] = []
        source_text: list[str] = []

        async with self.stream(src=src, tgt=tgt, delay=delay, url=url) as session:
            async for chunk in apace(_chunk_iter(pcm16), _DEFAULT_CHUNK_S):
                await session.send_audio(chunk)
            await session.commit()
            async for event in session:
                if event.type == "audio_chunk" and event.audio:
                    audio_chunks.append(event.audio)
                elif event.type == "partial_transcript" and event.text:
                    source_text.append(event.text)
                elif event.type == "done":
                    break

        return S2STResult(
            audio=b"".join(audio_chunks),
            sample_rate=_SESSION_SAMPLE_RATE,
            audio_format="pcm16",
            transcript_source="".join(source_text) or None,
            transcript_target=None,
        )
