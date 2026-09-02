"""Typed exceptions raised by the Kotoba SDK (REST + WebSocket)."""

from __future__ import annotations


class KotobaError(Exception):
    """Base class for all SDK errors."""


class APIError(KotobaError):
    """Generic transport / server error not covered by a more specific class."""

    def __init__(self, message: str, *, status_code: int | None = None, payload: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


class AuthError(APIError):
    """Authentication or authorization failure (HTTP 401/403)."""


class ProtocolError(APIError):
    """Server returned a structured error frame, malformed request, or 4xx."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status_code: int | None = None,
        payload: dict | None = None,
    ):
        super().__init__(message, status_code=status_code, payload=payload)
        self.code = code


class TimeoutError(APIError):  # noqa: A001 — importable as kotoba.TimeoutError
    """Operation exceeded its deadline (HTTP timeout, polling deadline, etc)."""


class WorkerStartupError(TimeoutError):
    """Provider worker did not become ready within the cold-start deadline."""


class UnsupportedRouteError(KotobaError):
    """Requested (modality, src, tgt) is not in the WebSocket routing table."""
