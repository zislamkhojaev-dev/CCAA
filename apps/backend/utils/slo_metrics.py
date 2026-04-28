from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock


_WINDOW = 512
_lock = Lock()
_samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=_WINDOW))
_HIST_BUCKETS_MS: tuple[float, ...] = (
    50.0,
    100.0,
    200.0,
    400.0,
    800.0,
    1200.0,
    1600.0,
    2500.0,
    5000.0,
)


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


def snapshot_histograms() -> dict[str, dict[str, float | int | dict[str, int]]]:
    with _lock:
        snap = {k: list(v) for k, v in _samples.items()}
    out: dict[str, dict[str, float | int | dict[str, int]]] = {}
    for metric, arr in snap.items():
        if not arr:
            continue
        buckets: dict[str, int] = {}
        for b in _HIST_BUCKETS_MS:
            buckets[f"{b:g}"] = sum(1 for v in arr if v <= b)
        buckets["+Inf"] = len(arr)
        out[metric] = {
            "count": len(arr),
            "sum_ms": round(float(sum(arr)), 4),
            "buckets": buckets,
        }
    return out


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    idx = int(round((len(values) - 1) * p))
    idx = max(0, min(len(values) - 1, idx))
    return round(values[idx], 2)
