"""Shared WebSocket session base for TTS and S2ST.

Two layers:

- `AsyncSession` is async-native. It runs a receiver coroutine that drains
  the websocket and pushes typed `StreamEvent`s onto an `asyncio.Queue`.
  Subclasses override `_handshake()` and `_handle_text_frame()`. Iteration
  is `async for event in session` — events are surfaced as soon as the
  receiver decodes them, so playback / display can be incremental.

- `SyncSession` wraps an `AsyncSession` instance with a background event
  loop running on a daemon thread. All sender helpers and the iterator are
  thin `run_coroutine_threadsafe(...).result()` calls. This is the same
  trick OpenAI's Python SDK uses for sync streaming and keeps the public
  API identical between sync and async.

The end goal for callers: `with` and `async with` look the same, examples
look the same, only the `await` keyword changes.
"""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import suppress
from typing import Any, AsyncIterator, Iterator

import websockets
import websockets.exceptions

from kotoba._providers import (
    ProviderConfig,
    is_retryable_session_error,
    resolve_provider,
    retry_session,
)
from kotoba.errors import APIError, AuthError, KotobaError, ProtocolError, TimeoutError
from kotoba.models import StreamEvent

_DONE_SENTINEL: StreamEvent | None = None  # `None` is the "no more events" marker


class AsyncSession:
    """Async-native WebSocket session base. Subclass per modality."""

    # Subclasses set this if they need to wait for a specific frame after open.
    handshake_timeout: float = 15.0
    receive_timeout: float | None = None

    def __init__(
        self,
        url: str,
        *,
        api_key: str | None = None,
        connect_kwargs: dict[str, Any] | None = None,
        provider: str | ProviderConfig | None = None,
    ) -> None:
        self._url = url
        self._api_key = api_key
        self._connect_kwargs = connect_kwargs or {}
        self._provider = resolve_provider(provider, url)
        self._ws: websockets.ClientConnection | None = None
        self._events: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._receiver_task: asyncio.Task[None] | None = None
        self._error: BaseException | None = None
        self._closed = False
        self._session_ready = asyncio.Event()

    # -- lifecycle ---------------------------------------------------------

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def start(self) -> None:
        if self._ws is not None:
            return
        policy = self._provider.ws_retry
        if policy is None:
            await self._connect_once()
            return
        # Cold-start providers (fal): capacity rejections during session
        # init are retried under the policy deadline. Mid-stream failures
        # are never retried.
        await retry_session(
            self._connect_once,
            policy,
            retryable=lambda exc: is_retryable_session_error(
                exc, self._provider.retryable_markers
            ),
        )

    async def _connect_once(self) -> None:
        headers: dict[str, str] = {}
        headers.update(self._provider.auth_headers(self._api_key))

        connect_kwargs: dict[str, Any] = {
            "additional_headers": headers,
            "ping_interval": 20,
            "ping_timeout": 10,
            "close_timeout": 0.5,
            "max_size": 10 * 1024 * 1024,
        }
        if self._provider.connect_overrides:
            connect_kwargs.update(self._provider.connect_overrides)
        connect_kwargs.update(self._connect_kwargs)

        try:
            self._ws = await websockets.connect(self._url, **connect_kwargs)
        except websockets.exceptions.InvalidStatus as exc:
            status = getattr(exc, "response", None)
            code = getattr(status, "status_code", None) if status is not None else None
            if code in (401, 403):
                raise AuthError(f"WebSocket auth rejected by {self._url} (status {code})") from exc
            raise APIError(
                f"WebSocket handshake failed: {exc}", status_code=code
            ) from exc
        except OSError as exc:
            raise APIError(f"Could not reach {self._url}: {exc}") from exc

        self._receiver_task = asyncio.create_task(self._receiver_loop())
        # Race the handshake against the receiver: a server that answers
        # session init with an error frame (or just closes) must surface
        # that error now, not a generic timeout handshake_timeout later.
        handshake_task = asyncio.create_task(self._handshake())
        done, _pending = await asyncio.wait(
            {handshake_task, self._receiver_task},
            timeout=self.handshake_timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if handshake_task in done:
            exc = handshake_task.exception()
            if exc is None:
                return
            await self._abort_attempt()
            if isinstance(exc, KotobaError):
                raise exc
            raise APIError(f"Session handshake failed: {exc!r}") from exc

        handshake_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await handshake_task

        if not done:
            await self._abort_attempt()
            raise TimeoutError(
                f"Timed out waiting for session handshake after {self.handshake_timeout}s"
            )

        # Receiver finished before the handshake: the server rejected the
        # session (error frame -> self._error) or dropped the connection.
        error = self._error
        await self._abort_attempt()
        if error is not None:
            raise error
        raise APIError("Connection closed during session init")

    async def _abort_attempt(self) -> None:
        """Tear down a failed connection attempt so a retry starts pristine.

        Unlike ``close()`` this does not mark the session closed and does
        not enqueue the done-sentinel — the queue and ready-event are
        recreated (not drained) so no stale sentinel or error can leak into
        a later successful attempt.
        """
        if self._receiver_task is not None and not self._receiver_task.done():
            self._receiver_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._receiver_task
        self._receiver_task = None
        if self._ws is not None:
            with suppress(Exception):
                await self._ws.close()
            self._ws = None
        self._events = asyncio.Queue()
        self._error = None
        self._session_ready = asyncio.Event()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._receiver_task and not self._receiver_task.done():
            self._receiver_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._receiver_task

        if self._ws is not None:
            with suppress(Exception):
                await self._ws.close()
            self._ws = None

        # Wake any pending iterator.
        await self._events.put(_DONE_SENTINEL)

    # -- iteration ---------------------------------------------------------

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self

    async def __anext__(self) -> StreamEvent:
        event = await self._events.get()
        if event is _DONE_SENTINEL:
            if self._error is not None:
                raise self._error
            raise StopAsyncIteration
        return event

    # -- subclass hooks ----------------------------------------------------

    async def _handshake(self) -> None:
        """Override to send open frames and wait for the server ack."""

    async def _handle_text_frame(self, payload: dict) -> None:
        """Override to translate one server frame into queued StreamEvents."""

        raise NotImplementedError

    # -- helpers for subclasses --------------------------------------------

    async def _send_json(self, payload: dict) -> None:
        if self._ws is None:
            raise APIError("Session is not started")
        await self._ws.send(json.dumps(payload))

    async def _emit(self, event: StreamEvent) -> None:
        await self._events.put(event)

    async def _emit_done(self) -> None:
        await self._events.put(_DONE_SENTINEL)

    # -- internals ---------------------------------------------------------

    async def _receiver_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    # Both Kotoba protocols use JSON text frames; ignore stray binary.
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                try:
                    if payload.get("type") == "x-fal-error":
                        # Platform-level error frame from the fal gateway
                        # (e.g. a capacity rejection) — same treatment as a
                        # server `error` frame, handled here so every
                        # modality benefits.
                        raise ProtocolError(
                            f"Fal platform error: "
                            f"{payload.get('message', payload)}",
                            code=str(payload.get("code", "x-fal-error")),
                            payload=payload,
                        )
                    await self._handle_text_frame(payload)
                except ProtocolError as exc:
                    self._error = exc
                    await self._emit_done()
                    return
        except websockets.exceptions.ConnectionClosed as exc:
            if exc.code not in (1000, 1001):
                self._error = APIError(
                    f"WebSocket closed unexpectedly: code={exc.code} reason={exc.reason}"
                )
        except Exception as exc:  # pragma: no cover - defensive
            self._error = APIError(f"Receiver loop crashed: {exc!r}")
        finally:
            await self._emit_done()


class SyncSession:
    """Sync facade over an `AsyncSession`, backed by a daemon event loop."""

    def __init__(self, async_session: AsyncSession) -> None:
        self._async = async_session
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self):
        self._start_loop()
        self._run(self._async.start())
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self._run(self._async.close())
        finally:
            self._stop_loop()

    # -- iteration ---------------------------------------------------------

    def __iter__(self) -> Iterator[StreamEvent]:
        return self

    def __next__(self) -> StreamEvent:
        try:
            return self._run(self._async.__anext__())
        except StopAsyncIteration:
            raise StopIteration

    # -- internals ---------------------------------------------------------

    def _start_loop(self) -> None:
        if self._loop is not None:
            return
        ready = threading.Event()
        loop_holder: dict[str, asyncio.AbstractEventLoop] = {}

        def runner() -> None:
            loop = asyncio.new_event_loop()
            loop_holder["loop"] = loop
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        self._thread = threading.Thread(
            target=runner, name="kotoba-sync-loop", daemon=True
        )
        self._thread.start()
        ready.wait()
        self._loop = loop_holder["loop"]

    def _stop_loop(self) -> None:
        if self._loop is None:
            return
        loop, self._loop = self._loop, None
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self, coro):
        if self._loop is None:
            raise RuntimeError("SyncSession is not started")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    # Subclasses (or the modality-specific facade) call this to expose
    # async sender helpers as blocking calls.
    def _call(self, coro_factory, *args, **kwargs):
        return self._run(coro_factory(*args, **kwargs))
