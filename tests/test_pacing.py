"""Tests for the realtime pacer used by file-driven WS one-shots."""

from __future__ import annotations

import asyncio
import time

import pytest

from kotoba._pacing import apace, pace


CHUNK_S = 0.04  # 40 ms — same as S2ST default


def _make_chunks(n: int) -> list[bytes]:
    return [b"\x00\x00" * 100 for _ in range(n)]


def test_pace_first_chunk_yields_immediately():
    chunks = _make_chunks(1)
    started = time.monotonic()
    out = list(pace(chunks, CHUNK_S))
    elapsed = time.monotonic() - started

    assert out == chunks
    # Single chunk: no pacing delay required.
    assert elapsed < CHUNK_S / 2


def test_pace_paces_subsequent_chunks():
    n = 5
    chunks = _make_chunks(n)
    started = time.monotonic()
    out = list(pace(chunks, CHUNK_S))
    elapsed = time.monotonic() - started

    assert out == chunks
    # n chunks at CHUNK_S spacing → (n-1) * CHUNK_S minimum elapsed.
    # Allow ~50% headroom for scheduler jitter.
    assert elapsed >= (n - 1) * CHUNK_S * 0.95
    assert elapsed < (n - 1) * CHUNK_S * 2.0


@pytest.mark.asyncio
async def test_apace_paces_subsequent_chunks():
    n = 5
    chunks = _make_chunks(n)
    loop = asyncio.get_running_loop()
    started = loop.time()
    out = [c async for c in apace(chunks, CHUNK_S)]
    elapsed = loop.time() - started

    assert out == chunks
    assert elapsed >= (n - 1) * CHUNK_S * 0.95
    assert elapsed < (n - 1) * CHUNK_S * 2.0


@pytest.mark.asyncio
async def test_apace_accepts_async_iterable():
    async def gen():
        for c in _make_chunks(3):
            yield c

    loop = asyncio.get_running_loop()
    started = loop.time()
    out = [c async for c in apace(gen(), CHUNK_S)]
    elapsed = loop.time() - started

    assert len(out) == 3
    assert elapsed >= 2 * CHUNK_S * 0.95


def test_pace_does_not_drift_on_slow_consumer():
    """If the consumer is slower than realtime, the pacer must not extend the schedule."""
    chunks = _make_chunks(5)
    started = time.monotonic()
    for chunk in pace(chunks, CHUNK_S):
        # Consumer is slower than chunk duration → pacer's sleep delta should be 0.
        time.sleep(CHUNK_S * 1.5)
    elapsed = time.monotonic() - started

    # 5 chunks × 1.5×chunk_s consumer work → ≈ 5 × 60ms = 300ms.
    # The pacer must not add extra delay on top.
    assert elapsed < 5 * CHUNK_S * 1.5 + CHUNK_S
