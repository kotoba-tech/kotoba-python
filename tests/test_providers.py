"""Unit tests for the provider layer (auth schemes + cold-start policies).

Pure in-process tests: the retry loop runs against a fake clock/sleep, so
deadline behavior is exercised without real waiting.
"""

from __future__ import annotations

import random

import pytest

from kotoba._providers import (
    FAL,
    KOTOBA,
    ProbePolicy,
    ProviderConfig,
    RetryPolicy,
    _is_fal_url,
    is_retryable_session_error,
    resolve_provider,
    retry_session,
)
from kotoba.errors import (
    APIError,
    AuthError,
    ProtocolError,
    TimeoutError,
    WorkerStartupError,
)


# ---------- auth headers ---------------------------------------------------


def test_auth_headers_schemes():
    assert KOTOBA.auth_headers("tok") == {"Authorization": "Bearer tok"}
    assert FAL.auth_headers("tok") == {"Authorization": "Key tok"}
    assert FAL.auth_headers(None) == {}


# ---------- provider resolution --------------------------------------------


def test_is_fal_url():
    assert _is_fal_url("https://fal.run/team/app")
    assert _is_fal_url("wss://fal.run/team/app/asr")
    assert _is_fal_url("wss://queue.fal.run/team/app")
    assert not _is_fal_url("https://notfal.run/team/app")
    assert not _is_fal_url("https://fal.run.evil.example/x")
    assert not _is_fal_url(None)
    assert not _is_fal_url("")


def test_resolve_provider_precedence(monkeypatch):
    monkeypatch.delenv("KOTOBA_PROVIDER", raising=False)
    # Default.
    assert resolve_provider(None, None) is KOTOBA
    # URL auto-detection.
    assert resolve_provider(None, "wss://fal.run/t/a") is FAL
    # Explicit name beats the URL.
    assert resolve_provider("kotoba", "wss://fal.run/t/a") is KOTOBA
    # A full config passes through untouched.
    custom = ProviderConfig(name="custom", auth_scheme="Key", api_key_env=())
    assert resolve_provider(custom, None) is custom
    # Env var beats auto-detection, loses to the explicit kwarg.
    monkeypatch.setenv("KOTOBA_PROVIDER", "fal")
    assert resolve_provider(None, None) is FAL
    assert resolve_provider("kotoba", "wss://fal.run/t/a") is KOTOBA


def test_resolve_provider_unknown_name():
    with pytest.raises(ValueError, match="Unknown provider"):
        resolve_provider("nope")


def test_fal_key_env_alone_does_not_flip_provider(monkeypatch):
    monkeypatch.delenv("KOTOBA_PROVIDER", raising=False)
    monkeypatch.setenv("FAL_KEY", "secret")
    assert resolve_provider(None, "wss://kotoba.example/asr") is KOTOBA


# ---------- error classification -------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "TTS server error: No available batch slot",
        "ASR server error: {'message': 'init_success not received'}",
        "worker_unavailable",
        "Server busy, try again later",
    ],
)
def test_capacity_markers_are_retryable(message):
    exc = ProtocolError(message)
    assert is_retryable_session_error(exc, FAL.retryable_markers)


def test_terminal_errors_are_not_retryable():
    markers = FAL.retryable_markers
    assert not is_retryable_session_error(AuthError("nope"), markers)
    # Auth words win even when a marker also matches.
    assert not is_retryable_session_error(
        ProtocolError("Unauthorized: no available batch slot"), markers
    )
    assert not is_retryable_session_error(
        APIError("Forbidden", status_code=403), markers
    )
    # A structured server error that isn't a capacity rejection.
    assert not is_retryable_session_error(
        ProtocolError("bad config frame"), markers
    )


def test_connection_layer_errors_are_retryable():
    markers = FAL.retryable_markers
    # TCP refused before the runner boots.
    refused = APIError("Could not reach wss://fal.run/t/a: [Errno 111]")
    refused.__cause__ = OSError(111, "refused")
    assert is_retryable_session_error(refused, markers)
    # Handshake timeout.
    assert is_retryable_session_error(TimeoutError("handshake"), markers)
    # 5xx / 429 rejections at the WS upgrade.
    assert is_retryable_session_error(
        APIError("WebSocket handshake failed", status_code=503), markers
    )
    assert is_retryable_session_error(
        APIError("WebSocket handshake failed", status_code=429), markers
    )
    assert not is_retryable_session_error(
        APIError("WebSocket handshake failed", status_code=404), markers
    )
    # Worker died mid-boot.
    assert is_retryable_session_error(
        APIError("Connection closed during session init"), markers
    )


# ---------- backoff --------------------------------------------------------


def test_full_jitter_delay_bounds():
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=20.0)
    rng = random.Random(42)
    for attempt in range(1, 12):
        cap = min(20.0, 1.0 * 2 ** (attempt - 1))
        for _ in range(20):
            delay = policy.delay(attempt, rng)
            assert 0 <= delay <= cap


# ---------- retry_session --------------------------------------------------


class _FakeTime:
    def __init__(self):
        self.now = 0.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.slept.append(delay)
        self.now += delay


async def test_retry_session_retries_then_succeeds():
    fake = _FakeTime()
    attempts = 0

    async def connect_once():
        nonlocal attempts
        attempts += 1
        if attempts < 4:
            raise ProtocolError("No available batch slot")

    await retry_session(
        connect_once,
        RetryPolicy(deadline_s=360.0),
        retryable=lambda e: is_retryable_session_error(e, FAL.retryable_markers),
        sleep=fake.sleep,
        clock=fake.clock,
        rng=random.Random(7),
    )
    assert attempts == 4
    assert len(fake.slept) == 3


async def test_retry_session_deadline_raises_worker_startup_error():
    fake = _FakeTime()

    async def connect_once():
        fake.now += 5.0  # each attempt burns wall clock
        raise ProtocolError("No available batch slot")

    with pytest.raises(WorkerStartupError) as excinfo:
        await retry_session(
            connect_once,
            RetryPolicy(deadline_s=30.0),
            retryable=lambda e: is_retryable_session_error(
                e, FAL.retryable_markers
            ),
            sleep=fake.sleep,
            clock=fake.clock,
            rng=random.Random(7),
        )
    assert isinstance(excinfo.value.__cause__, ProtocolError)
    # WorkerStartupError stays catchable as the SDK timeout / APIError.
    assert isinstance(excinfo.value, TimeoutError)
    assert isinstance(excinfo.value, APIError)
    # The deadline bounds total elapsed time (attempts + backoff).
    assert fake.now <= 30.0 + 5.0


async def test_retry_session_auth_error_short_circuits():
    fake = _FakeTime()
    attempts = 0

    async def connect_once():
        nonlocal attempts
        attempts += 1
        raise AuthError("WebSocket auth rejected (status 401)")

    with pytest.raises(AuthError):
        await retry_session(
            connect_once,
            RetryPolicy(deadline_s=360.0),
            retryable=lambda e: is_retryable_session_error(
                e, FAL.retryable_markers
            ),
            sleep=fake.sleep,
            clock=fake.clock,
        )
    assert attempts == 1
    assert fake.slept == []


async def test_retry_session_non_retryable_short_circuits():
    async def connect_once():
        raise ProtocolError("bad config frame")

    with pytest.raises(ProtocolError, match="bad config frame"):
        await retry_session(
            connect_once,
            RetryPolicy(deadline_s=360.0),
            retryable=lambda e: is_retryable_session_error(
                e, FAL.retryable_markers
            ),
        )


# ---------- built-in configs -----------------------------------------------


def test_kotoba_profile_has_no_cold_start_machinery():
    assert KOTOBA.ws_retry is None
    assert KOTOBA.http_probe is None
    assert KOTOBA.api_key_env == ("KOTOBA_API_KEY",)


def test_fal_profile():
    assert FAL.auth_scheme == "Key"
    assert FAL.api_key_env == ("FAL_KEY", "KOTOBA_API_KEY")
    assert FAL.ws_retry == RetryPolicy()
    assert FAL.http_probe == ProbePolicy()
    assert FAL.connect_overrides == {"open_timeout": 30}
