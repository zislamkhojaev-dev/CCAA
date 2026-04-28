from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from apps.backend.utils.slo_metrics import snapshot_percentiles
from apps.backend.utils.resilience import snapshot_resilience_stats

router = APIRouter()


@router.get("/slo")
async def slo_snapshot() -> dict:
    return {"metrics": snapshot_percentiles()}


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics() -> PlainTextResponse:
    snap = snapshot_percentiles()
    res = await snapshot_resilience_stats()
    lines: list[str] = []
    lines.append("# HELP voiceagent_latency_samples Number of latency samples in window")
    lines.append("# TYPE voiceagent_latency_samples gauge")
    lines.append("# HELP voiceagent_latency_ms Latency percentiles in milliseconds")
    lines.append("# TYPE voiceagent_latency_ms gauge")
    for metric, row in sorted(snap.items()):
        m = metric.replace(".", "_").replace("-", "_")
        count = int(row.get("count", 0))
        p50 = float(row.get("p50_ms", 0.0))
        p95 = float(row.get("p95_ms", 0.0))
        max_v = float(row.get("max_ms", 0.0))
        lines.append(f'voiceagent_latency_samples{{metric="{m}"}} {count}')
        lines.append(f'voiceagent_latency_ms{{metric="{m}",quantile="0.50"}} {p50}')
        lines.append(f'voiceagent_latency_ms{{metric="{m}",quantile="0.95"}} {p95}')
        lines.append(f'voiceagent_latency_ms{{metric="{m}",quantile="1.00"}} {max_v}')
    lines.append("# HELP voiceagent_resilience_total Resilience subsystem counters")
    lines.append("# TYPE voiceagent_resilience_total counter")
    for name, val in sorted(res.items()):
        lines.append(f'voiceagent_resilience_total{{name="{name}"}} {int(val)}')
    return PlainTextResponse("\n".join(lines) + "\n")
