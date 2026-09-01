"""HTTP session with retry/backoff shared by all REST sub-clients."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

import httpx
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from kotoba._providers import ProviderConfig, resolve_provider
from kotoba.errors import (
    APIError,
    AuthError,
    ProtocolError,
    TimeoutError,
    WorkerStartupError,
)

_RETRY_STATUS = (429, 500, 502, 503, 504)

logger = logging.getLogger(__name__)


class HttpSession:
    """Thin wrapper around ``requests.Session`` with auth + retry preset.

    Retries network errors and idempotent 5xx/429 responses with exponential
    backoff. Callers that need to interpret non-error statuses (e.g. 202 for
    "still processing", 404 for "not found") pass ``allow_statuses``.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout: float,
        max_retries: int,
        backoff_factor: float = 0.5,
        provider: str | ProviderConfig | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._provider = resolve_provider(provider, base_url)
        self._auth_headers = self._provider.auth_headers(api_key)
        self._warmed = False

        session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            # "POST" is non-idempotent, so we don't retry on POST.
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(self._auth_headers)
        self._session = session

    def post(self, path: str, **kwargs: Any) -> requests.Response:
        return self._request("POST", path, **kwargs)

    def get(self, path: str, **kwargs: Any) -> requests.Response:
        return self._request("GET", path, **kwargs)

    def ensure_ready(self) -> None:
        """Probe the readiness endpoint until it answers 200 (cold start).

        No-op unless the provider has a probe policy (fal). Each probe is a
        short-lived plain request outside the retrying session, so an
        orphaned parked request costs one probe timeout and the next probe
        re-triggers a boot. 401/403 raise immediately; the overall wait is
        bounded by the policy deadline (``WorkerStartupError`` past it).
        """
        policy = self._provider.http_probe
        if policy is None or self._warmed:
            return
        url = f"{self.base_url}{policy.probe_path}"
        started = time.monotonic()
        attempt = 0
        last = "no probe completed"
        while True:
            attempt += 1
            try:
                response = requests.get(
                    url,
                    headers=self._auth_headers,
                    timeout=policy.probe_timeout_s,
                )
            except requests.RequestException as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    self._warmed = True
                    return
                if response.status_code in (401, 403):
                    _raise_for_status(
                        response.status_code, _safe_json(response), url
                    )
                last = f"HTTP {response.status_code}"
            logger.info("Readiness probe %d: %s; app not ready", attempt, last)
            if time.monotonic() - started > policy.deadline_s:
                raise WorkerStartupError(
                    f"App at {url} not ready after {policy.deadline_s:.0f}s "
                    f"({attempt} probes; last: {last})"
                )
            time.sleep(policy.probe_interval_s)

    def _request(
        self,
        method: str,
        path: str,
        *,
        allow_statuses: tuple[int, ...] = (),
        **kwargs: Any,
    ) -> requests.Response:
        url = f"{self.base_url}{path}"
        kwargs.setdefault("timeout", self.timeout)
        try:
            response = self._session.request(method, url, **kwargs)
        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"Request to {url} timed out") from e
        except requests.exceptions.ConnectionError as e:
            raise APIError(f"Failed to reach {url}: {e}") from e
        except requests.RequestException as e:
            raise APIError(f"Request to {url} failed: {e}") from e

        if response.status_code in allow_statuses:
            return response

        if response.status_code < 400:
            return response

        self._raise_for_status(response)
        return response  # unreachable

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        _raise_for_status(response.status_code, _safe_json(response), response.url)


def _safe_json(response: "requests.Response | httpx.Response") -> dict:
    try:
        value = response.json()
    except ValueError:
        value = {"detail": response.text[:512]}
    if not isinstance(value, dict):
        value = {"detail": str(value)[:512]}
    return value


def _raise_for_status(status: int, payload: dict, url: str) -> None:
    message = _extract_message(payload) or f"HTTP {status} from {url}"
    if status in (401, 403):
        raise AuthError(message, status_code=status, payload=payload)
    if 400 <= status < 500:
        raise ProtocolError(message, status_code=status, payload=payload)
    raise APIError(message, status_code=status, payload=payload)


def _extract_message(payload: dict) -> str | None:
    """Pull the human-readable error string out of a server payload.

    Handles the common shapes:

    - FastAPI: ``{"detail": "..."}`` (or a list of validation entries on 422)
    - Flat: ``{"message": "..."}``
    - OpenAI-style envelope: ``{"error": {"message": "...", "code": "..."}}``
    - String error: ``{"error": "..."}``

    For structured (non-string) ``detail``/``error`` payloads — most often
    FastAPI 422 with ``detail`` as a list of validation entries — falls back
    to ``str(value)`` so the diagnostics still surface in the exception
    message instead of a generic ``HTTP <status>`` placeholder.
    """

    detail = payload.get("detail")
    if isinstance(detail, str) and detail:
        return detail
    flat = payload.get("message")
    if isinstance(flat, str) and flat:
        return flat
    err = payload.get("error")
    if isinstance(err, dict):
        nested = err.get("message")
        if isinstance(nested, str) and nested:
            return nested
    if isinstance(err, str) and err:
        return err

    for value in (detail, flat, err):
        if value:
            return str(value)
    return None


class AsyncHttpSession:
    """Async counterpart to :class:`HttpSession`.

    Wraps ``httpx.AsyncClient`` and implements manual exponential-backoff
    retry for network errors and 429/5xx responses. Honors the
    ``Retry-After`` header on 429 when present.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout: float,
        max_retries: int,
        backoff_factor: float = 0.5,
        max_backoff: float = 10.0,
        provider: str | ProviderConfig | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor
        self._max_backoff = max_backoff
        self._provider = resolve_provider(provider, base_url)
        self._warmed = False

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers=self._provider.auth_headers(api_key),
        )

    async def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return await self._request("POST", path, **kwargs)

    async def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return await self._request("GET", path, **kwargs)

    async def ensure_ready(self) -> None:
        """Async mirror of :meth:`HttpSession.ensure_ready`."""
        policy = self._provider.http_probe
        if policy is None or self._warmed:
            return
        started = time.monotonic()
        attempt = 0
        last = "no probe completed"
        while True:
            attempt += 1
            try:
                # Raw client call: probes must bypass the retrying _request
                # wrapper so each one stays a short, bounded attempt.
                response = await self._client.get(
                    policy.probe_path, timeout=policy.probe_timeout_s
                )
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    self._warmed = True
                    return
                if response.status_code in (401, 403):
                    _raise_for_status(
                        response.status_code,
                        _safe_json(response),
                        str(response.url),
                    )
                last = f"HTTP {response.status_code}"
            logger.info("Readiness probe %d: %s; app not ready", attempt, last)
            if time.monotonic() - started > policy.deadline_s:
                raise WorkerStartupError(
                    f"App at {self.base_url}{policy.probe_path} not ready after "
                    f"{policy.deadline_s:.0f}s ({attempt} probes; last: {last})"
                )
            await asyncio.sleep(policy.probe_interval_s)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncHttpSession":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        allow_statuses: tuple[int, ...] = (),
        **kwargs: Any,
    ) -> httpx.Response:
        attempt = 0
        last_exc: Exception | None = None
        while True:
            try:
                response = await self._client.request(method, path, **kwargs)
            except httpx.TimeoutException as e:
                last_exc = TimeoutError(f"Request to {path} timed out")
                last_exc.__cause__ = e
            except httpx.TransportError as e:
                last_exc = APIError(f"Failed to reach {path}: {e}")
                last_exc.__cause__ = e
            except httpx.HTTPError as e:
                last_exc = APIError(f"Request to {path} failed: {e}")
                last_exc.__cause__ = e
            else:
                if (
                    response.status_code in _RETRY_STATUS
                    and attempt < self._max_retries
                ):
                    await self._sleep_for_retry(response, attempt)
                    attempt += 1
                    continue
                if response.status_code in allow_statuses:
                    return response
                if response.status_code < 400:
                    return response
                _raise_for_status(
                    response.status_code, _safe_json(response), str(response.url)
                )

            if attempt >= self._max_retries:
                assert last_exc is not None
                raise last_exc
            await asyncio.sleep(self._backoff_delay(attempt))
            attempt += 1

    def _backoff_delay(self, attempt: int) -> float:
        delay = self._backoff_factor * (2**attempt)
        delay = min(delay, self._max_backoff)
        return delay + random.uniform(0, delay * 0.1)

    async def _sleep_for_retry(self, response: httpx.Response, attempt: int) -> None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                await asyncio.sleep(min(float(retry_after), self._max_backoff))
                return
            except ValueError:
                pass
        await asyncio.sleep(self._backoff_delay(attempt))
