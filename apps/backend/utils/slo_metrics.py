from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock


_WINDOW = 512
_lock = Lock()
_samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=_WINDOW))


def record_latency(metric: str, value_ms: float) -> None:
    v = float(value_ms)
    if v < 0:
        return
    with _lock:
        _samples[metric].append(v)


def snapshot_percentiles() -> dict[str, dict[str, float | int]]:
    with _lock:
        snap = {k: list(v) for k, v in _samples.items()}
    out: dict[str, dict[str, float | int]] = {}
    for metric, arr in snap.items():
        if not arr:
            continue
        arr.sort()
        out[metric] = {
            "count": len(arr),
            "p50_ms": _percentile(arr, 0.50),
            "p95_ms": _percentile(arr, 0.95),
            "max_ms": round(arr[-1], 2),
        }
    return out


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    idx = int(round((len(values) - 1) * p))
    idx = max(0, min(len(values) - 1, idx))
    return round(values[idx], 2)
