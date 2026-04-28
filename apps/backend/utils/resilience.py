from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")
_stats_lock = asyncio.Lock()
_stats: dict[str, int] = {
    "circuit_open_rejections": 0,
    "circuit_open_events": 0,
    "retry_attempts": 0,
    "retry_failures": 0,
}


async def _inc_stat(name: str, delta: int = 1) -> None:
    async with _stats_lock:
        _stats[name] = int(_stats.get(name, 0)) + int(delta)


async def snapshot_resilience_stats() -> dict[str, int]:
    async with _stats_lock:
        return dict(_stats)


class SimpleCircuitBreaker:
    """Minimal async-safe circuit breaker for external provider calls."""

    def __init__(self, *, fail_threshold: int = 3, open_sec: float = 20.0) -> None:
        self._fail_threshold = max(1, int(fail_threshold))
        self._open_sec = max(1.0, float(open_sec))
        self._fails = 0
        self._open_until = 0.0
        self._lock = asyncio.Lock()

    async def before_call(self) -> bool:
        async with self._lock:
            now = time.monotonic()
            if now < self._open_until:
                await _inc_stat("circuit_open_rejections")
                return False
            if self._open_until > 0:
                # Half-open: allow one call and reset counters.
                self._open_until = 0.0
                self._fails = 0
            return True

    async def on_success(self) -> None:
        async with self._lock:
            self._fails = 0
            self._open_until = 0.0

    async def on_failure(self) -> None:
        async with self._lock:
            self._fails += 1
            if self._fails >= self._fail_threshold:
                self._open_until = time.monotonic() + self._open_sec
                self._fails = 0
                await _inc_stat("circuit_open_events")


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 2,
    initial_backoff_sec: float = 0.15,
) -> T:
    tries = max(1, int(attempts))
    backoff = max(0.01, float(initial_backoff_sec))
    last_exc: Exception | None = None
    for idx in range(tries):
        try:
            return await fn()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if idx == tries - 1:
                break
            await _inc_stat("retry_attempts")
            await asyncio.sleep(backoff)
            backoff *= 2
    assert last_exc is not None
    await _inc_stat("retry_failures")
    raise last_exc
