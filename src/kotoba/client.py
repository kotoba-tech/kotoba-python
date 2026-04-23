"""Public Kotoba client facade.

Two flavors:

- :class:`KotobaClient` — sync, backed by ``requests`` for REST and a daemon
  event loop for WebSocket streaming.
- :class:`AsyncKotobaClient` — async, backed by ``httpx.AsyncClient`` for
  REST and the native asyncio WebSocket client.

Both own sub-clients for each modality::

    client.asr   — REST + WS (POST / poll, or stream)
    client.tts   — WS streaming text-to-speech
    client.s2st  — WS streaming speech-to-speech translation

URL kwargs map 1:1 onto env vars (all optional, read at construction
time if not passed explicitly):

    url                → KOTOBA_ASR_REST_URL       (REST transcription jobs)
    asr_ws_url         → KOTOBA_ASR_URL            (WS live ASR)
    tts_ja_ws_url      → KOTOBA_TTS_JA_URL         (WS Japanese TTS)
    s2st_en_ja_ws_url  → KOTOBA_S2ST_EN_JA_URL     (WS en→ja speech translation)

Usage::

    import os
    import kotoba

    # Pass only what you need — everything else stays unconfigured.
    client = kotoba.KotobaClient(
        api_key=os.environ["KOTOBA_API_KEY"],
        url=os.environ["KOTOBA_ASR_REST_URL"],
    )
    result = client.asr.transcribe("sample.wav", language="ja")
    print(result.text)
"""

from __future__ import annotations

import os
from typing import Any

from kotoba._http import AsyncHttpSession, HttpSession
from kotoba.asr import ASRClient, AsyncASRClient
from kotoba.routing import register_endpoint
from kotoba.s2st import AsyncS2STClient, S2STClient
from kotoba.tts import AsyncTTSClient, TTSClient

_API_KEY_ENV_VAR = "KOTOBA_API_KEY"
_API_URL_ENV_VAR = "KOTOBA_ASR_REST_URL"
_WS_ASR_ENV_VAR = "KOTOBA_ASR_URL"
_WS_TTS_JA_ENV_VAR = "KOTOBA_TTS_JA_URL"
_WS_S2ST_EN_JA_ENV_VAR = "KOTOBA_S2ST_EN_JA_URL"


def _resolve_from_env(name: str) -> str | None:
    value = os.environ.get(name)
    return value or None


def _register_ws_routes(
    asr_ws_url: str | None,
    tts_ja_ws_url: str | None,
    s2st_en_ja_ws_url: str | None,
) -> None:
    """Push explicit WS URLs into the module-level routing registry so
    ``client.asr.stream()`` / ``client.tts.stream()`` / ``client.s2st.stream()``
    pick them up without an explicit ``url=`` kwarg per call."""
    if asr_ws_url:
        register_endpoint("asr", None, None, asr_ws_url)
    if tts_ja_ws_url:
        register_endpoint("tts", None, "ja", tts_ja_ws_url)
    if s2st_en_ja_ws_url:
        register_endpoint("s2st", "en", "ja", s2st_en_ja_ws_url)


class KotobaClient:
    """Entry point for REST + WebSocket access to Kotoba speech APIs.

    Args:
        api_key: Bearer token. Falls back to ``KOTOBA_API_KEY`` env var.
        url: REST API base URL (e.g. ``https://.../v1``). Falls back to
            ``KOTOBA_ASR_REST_URL``.
        asr_ws_url: WebSocket URL for live ASR. Falls back to
            ``KOTOBA_ASR_URL``.
        tts_ja_ws_url: WebSocket URL for Japanese TTS. Falls back to
            ``KOTOBA_TTS_JA_URL``.
        s2st_en_ja_ws_url: WebSocket URL for English→Japanese S2ST. Falls
            back to ``KOTOBA_S2ST_EN_JA_URL``.
        timeout: Per-request HTTP timeout in seconds (REST only).
        max_retries: Retry budget for network errors and transient 5xx/429.

    Each URL can be omitted if the corresponding feature isn't used; the
    SDK only fails when a sub-client is actually invoked without a URL.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        url: str | None = None,
        asr_ws_url: str | None = None,
        tts_ja_ws_url: str | None = None,
        s2st_en_ja_ws_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        api_key = api_key or _resolve_from_env(_API_KEY_ENV_VAR)
        url = url or _resolve_from_env(_API_URL_ENV_VAR)
        asr_ws_url = asr_ws_url or _resolve_from_env(_WS_ASR_ENV_VAR)
        tts_ja_ws_url = tts_ja_ws_url or _resolve_from_env(_WS_TTS_JA_ENV_VAR)
        s2st_en_ja_ws_url = s2st_en_ja_ws_url or _resolve_from_env(
            _WS_S2ST_EN_JA_ENV_VAR
        )

        self._api_key = api_key
        self._url = url
        self._asr_ws_url = asr_ws_url
        self._tts_ja_ws_url = tts_ja_ws_url
        self._s2st_en_ja_ws_url = s2st_en_ja_ws_url

        _register_ws_routes(asr_ws_url, tts_ja_ws_url, s2st_en_ja_ws_url)

        self._http: HttpSession | None = None
        if url:
            self._http = HttpSession(
                base_url=url,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
            )

        self.asr = ASRClient(self._http, api_key=api_key)
        self.tts = TTSClient(api_key)
        self.s2st = S2STClient(api_key)

    @property
    def api_key(self) -> str | None:
        return self._api_key

    @property
    def url(self) -> str | None:
        """REST base URL."""
        return self._url

    @property
    def asr_ws_url(self) -> str | None:
        return self._asr_ws_url

    @property
    def tts_ja_ws_url(self) -> str | None:
        return self._tts_ja_ws_url

    @property
    def s2st_en_ja_ws_url(self) -> str | None:
        return self._s2st_en_ja_ws_url


class AsyncKotobaClient:
    """Async entry point. Same surface as :class:`KotobaClient` with ``await``.

    Usable as a context manager so the underlying HTTP pool is closed::

        async with kotoba.AsyncKotobaClient(...) as client:
            result = await client.asr.transcribe("sample.wav", language="ja")
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        url: str | None = None,
        asr_ws_url: str | None = None,
        tts_ja_ws_url: str | None = None,
        s2st_en_ja_ws_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        api_key = api_key or _resolve_from_env(_API_KEY_ENV_VAR)
        url = url or _resolve_from_env(_API_URL_ENV_VAR)
        asr_ws_url = asr_ws_url or _resolve_from_env(_WS_ASR_ENV_VAR)
        tts_ja_ws_url = tts_ja_ws_url or _resolve_from_env(_WS_TTS_JA_ENV_VAR)
        s2st_en_ja_ws_url = s2st_en_ja_ws_url or _resolve_from_env(
            _WS_S2ST_EN_JA_ENV_VAR
        )

        self._api_key = api_key
        self._url = url
        self._asr_ws_url = asr_ws_url
        self._tts_ja_ws_url = tts_ja_ws_url
        self._s2st_en_ja_ws_url = s2st_en_ja_ws_url

        _register_ws_routes(asr_ws_url, tts_ja_ws_url, s2st_en_ja_ws_url)

        self._http: AsyncHttpSession | None = None
        if url:
            self._http = AsyncHttpSession(
                base_url=url,
                api_key=api_key,
                timeout=timeout,
                max_retries=max_retries,
            )

        self.asr = AsyncASRClient(self._http, api_key=api_key)
        self.tts = AsyncTTSClient(api_key)
        self.s2st = AsyncS2STClient(api_key)

    @property
    def api_key(self) -> str | None:
        return self._api_key

    @property
    def url(self) -> str | None:
        """REST base URL."""
        return self._url

    @property
    def asr_ws_url(self) -> str | None:
        return self._asr_ws_url

    @property
    def tts_ja_ws_url(self) -> str | None:
        return self._tts_ja_ws_url

    @property
    def s2st_en_ja_ws_url(self) -> str | None:
        return self._s2st_en_ja_ws_url

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()

    async def __aenter__(self) -> "AsyncKotobaClient":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()
