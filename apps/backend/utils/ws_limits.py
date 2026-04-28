from __future__ import annotations

import asyncio


_lock = asyncio.Lock()
_active: dict[str, int] = {}


async def try_acquire(scope: str, limit: int) -> bool:
    async with _lock:
        cur = int(_active.get(scope, 0))
        if cur >= max(1, int(limit)):
            return False
        _active[scope] = cur + 1
        return True


async def release(scope: str) -> None:
    async with _lock:
        cur = int(_active.get(scope, 0))
        if cur <= 1:
            _active.pop(scope, None)
            return
        _active[scope] = cur - 1
