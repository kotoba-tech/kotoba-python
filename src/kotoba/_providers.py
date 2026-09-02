"""Provider profiles: auth scheme + cold-start policies per backend.

The Kotoba services are reachable either directly (``kotoba`` provider,
today's behavior) or through fal.ai's serverless gateway (``fal`` provider),
which differs in three ways:

- Auth is ``Authorization: Key <api_key>`` instead of ``Bearer <token>``.
- Apps scale to zero, so a first request can land on a cold runner that
  takes minutes to boot.
- Session initialization can be rejected with a capacity error ("No
  available batch slot" etc.) until a slot frees or the autoscaler reacts.

The retry and probe policies here mirror the client guidance validated by
the Fal benchmark campaign (kotoba-realtime-subtitles bench_core):

- WS session-init: capacity rejections are retried with full-jitter
  exponential backoff under a wall-clock deadline (a retry lands after slot
  turnover or an overflow-runner boot, so the budget is time, not attempt
  count). Auth and protocol errors are terminal. Only session establishment
  is retried — a mid-stream failure never is.
- Cold HTTP: never park one long request on a booting app. Probe the
  readiness endpoint with short per-probe timeouts and send the real
  request only once a probe succeeds, bounded by an overall deadline.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

from kotoba.errors import (
    APIError,
    AuthError,
    ProtocolError,
    TimeoutError,
    WorkerStartupError,
)

_PROVIDER_ENV_VAR = "KOTOBA_PROVIDER"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    """Deadline-based WS session-init retry with full-jitter backoff."""

    deadline_s: float = 360.0
    base_delay_s: float = 1.0
    max_delay_s: float = 20.0

    def delay(self, attempt: int, rng: random.Random | None = None) -> float:
        """Full-jitter delay for the given 1-based attempt."""
        cap = min(self.max_delay_s, self.base_delay_s * (2 ** (attempt - 1)))
        return (rng or random).uniform(0, cap)


@dataclass(frozen=True)
class ProbePolicy:
    """Sequential HTTP readiness probing before a cold REST request."""

    probe_timeout_s: float = 30.0
    probe_interval_s: float = 2.0
    deadline_s: float = 600.0
    probe_path: str = "/model_and_cuda_availability"


@dataclass(frozen=True)
class ProviderConfig:
    """How to talk to one backend: auth scheme + cold-start policies."""

    name: str
    auth_scheme: str  # "Bearer" | "Key"
    api_key_env: tuple[str, ...]
    ws_retry: RetryPolicy | None = None  # None => single connect attempt
    http_probe: ProbePolicy | None = None  # None => no readiness probing
    retryable_markers: tuple[str, ...] = ()
    connect_overrides: dict[str, Any] | None = None
    # One-shot synchronous transcription endpoint (POST, multipart) used by
    # ``asr.transcribe()``.
    batch_transcribe_path: str = "/v1/speech-to-text"

    def auth_headers(self, api_key: str | None) -> dict[str, str]:
        if not api_key:
            return {}
        return {"Authorization": f"{self.auth_scheme} {api_key}"}


KOTOBA = ProviderConfig(
    name="kotoba",
    auth_scheme="Bearer",
    api_key_env=("KOTOBA_API_KEY",),
)

# Marker substrings identifying capacity rejections, until the server emits
# a structured retryable code. Conservative on purpose: auth/protocol errors
# must never match.
FAL = ProviderConfig(
    name="fal",
    auth_scheme="Key",
    api_key_env=("FAL_KEY", "KOTOBA_API_KEY"),
    ws_retry=RetryPolicy(),
    http_probe=ProbePolicy(),
    retryable_markers=(
        "no available batch slot",
        "init_success not received",
        "worker_unavailable",
        "try again later",
    ),
    # Cold TCP/WS opens can exceed the websockets default of 10s.
    connect_overrides={"open_timeout": 30},
)

_BY_NAME = {KOTOBA.name: KOTOBA, FAL.name: FAL}


def _is_fal_url(url: str | None) -> bool:
    if not url:
        return False
    host = urlsplit(url).hostname
    return host is not None and (host == "fal.run" or host.endswith(".fal.run"))


def resolve_provider(
    provider: str | ProviderConfig | None,
    url: str | None = None,
) -> ProviderConfig:
    """Resolve the provider for one endpoint.

    Precedence: explicit ``provider`` > ``KOTOBA_PROVIDER`` env var >
    URL-host auto-detection (``fal.run`` / ``*.fal.run``) > ``kotoba``.
    ``FAL_KEY`` being exported never flips the provider by itself.
    """
    if isinstance(provider, ProviderConfig):
        return provider
    name = provider or os.environ.get(_PROVIDER_ENV_VAR) or None
    if name is not None:
        try:
            return _BY_NAME[name.lower()]
        except KeyError:
            raise ValueError(
                f"Unknown provider {name!r}; expected one of {sorted(_BY_NAME)}"
            ) from None
    if _is_fal_url(url):
        return FAL
    return KOTOBA


def is_retryable_session_error(
    exc: BaseException, markers: tuple[str, ...]
) -> bool:
    """Whether a session-init failure is worth retrying.

    Terminal: auth failures and structured server errors that don't match a
    capacity marker. Retryable: capacity rejections, handshake timeouts,
    5xx/429 handshake responses, and connection-layer failures (a runner
    that isn't up yet, or died mid-boot).
    """
    if isinstance(exc, AuthError):
        return False
    message = str(exc).lower()
    if "unauthorized" in message or "forbidden" in message:
        return False
    if any(marker in message for marker in markers):
        return True
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, ProtocolError):
        return False
    if isinstance(exc, APIError):
        if exc.status_code is not None:
            return exc.status_code >= 500 or exc.status_code in (408, 429)
        return True
    return False


async def retry_session(
    connect_once: Callable[[], Awaitable[None]],
    policy: RetryPolicy,
    *,
    retryable: Callable[[BaseException], bool],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
    rng: random.Random | None = None,
) -> None:
    """Run ``connect_once``, retrying retryable failures per ``policy``.

    The budget is wall-clock (capacity rejections resolve on slot turnover
    or an autoscaler boot — minutes, not attempts). Raises
    ``WorkerStartupError`` (with the last failure as ``__cause__``) once the
    next backoff would cross the deadline; non-retryable errors re-raise
    immediately.
    """
    started = clock()
    attempt = 0
    while True:
        attempt += 1
        try:
            await connect_once()
            return
        except Exception as exc:
            if not retryable(exc):
                raise
            delay = policy.delay(attempt, rng)
            if clock() - started + delay > policy.deadline_s:
                raise WorkerStartupError(
                    f"Worker did not become ready within {policy.deadline_s:.0f}s "
                    f"({attempt} attempts; last error: "
                    f"{type(exc).__name__}: {exc})"
                ) from exc
            logger.info(
                "Session init attempt %d failed (%s: %s); retrying in %.1fs",
                attempt,
                type(exc).__name__,
                exc,
                delay,
            )
            await sleep(delay)
