"""Real-time pacing for file-driven WS one-shots.

The ASR / S2ST WebSocket servers consume audio as a live stream. When a
client reads a finished file and pushes it through a WS session, it must
not blast every chunk at once — the server's input buffer is sized for
realtime input and partial transcripts come out as audio arrives. These
helpers wrap a chunk iterator so each chunk is yielded no faster than
its own duration.

The schedule is anchored to a monotonic start time and the chunk index,
so per-chunk send time doesn't drift the overall rate.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterable, AsyncIterator, Iterable, Iterator


def pace(chunks: Iterable[bytes], chunk_duration_s: float) -> Iterator[bytes]:
    """Yield ``chunks`` paced at roughly realtime.

    The first chunk yields immediately; subsequent chunks wait until
    ``start + i * chunk_duration_s`` of monotonic time. Total elapsed
    iteration time approaches ``n * chunk_duration_s`` from above.
    """

    started = time.monotonic()
    for i, chunk in enumerate(chunks):
        if i > 0:
            delay = (started + i * chunk_duration_s) - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        yield chunk


async def apace(
    chunks: Iterable[bytes] | AsyncIterable[bytes],
    chunk_duration_s: float,
) -> AsyncIterator[bytes]:
    """Async counterpart to :func:`pace`. Accepts sync or async chunk sources."""

    loop = asyncio.get_running_loop()
    started = loop.time()
    i = 0
    if isinstance(chunks, AsyncIterable):
        async for chunk in chunks:
            if i > 0:
                delay = (started + i * chunk_duration_s) - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
            i += 1
            yield chunk
    else:
        for chunk in chunks:
            if i > 0:
                delay = (started + i * chunk_duration_s) - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
            i += 1
            yield chunk
