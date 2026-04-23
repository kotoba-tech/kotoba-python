"""HTTP session with retry/backoff shared by all REST sub-clients."""

from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from kotoba.errors import (
    APIError,
    AuthError,
    ProtocolError,
    TimeoutError,
)

_RETRY_STATUS = (429, 500, 502, 503, 504)


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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

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
        if api_key:
            session.headers["Authorization"] = f"Bearer {api_key}"
        self._session = session

    def post(self, path: str, **kwargs: Any) -> requests.Response:
        return self._request("POST", path, **kwargs)

    def get(self, path: str, **kwargs: Any) -> requests.Response:
        return self._request("GET", path, **kwargs)

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
    message = (
        payload.get("detail")
        or payload.get("message")
        or payload.get("error")
        or f"HTTP {status} from {url}"
    )
    if status in (401, 403):
        raise AuthError(message, status_code=status, payload=payload)
    if 400 <= status < 500:
        raise ProtocolError(message, status_code=status, payload=payload)
    raise APIError(message, status_code=status, payload=payload)


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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor
        self._max_backoff = max_backoff

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers=headers,
        )

    async def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return await self._request("POST", path, **kwargs)

    async def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return await self._request("GET", path, **kwargs)

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
