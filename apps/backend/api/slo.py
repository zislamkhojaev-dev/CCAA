from __future__ import annotations

from fastapi import APIRouter

from apps.backend.utils.slo_metrics import snapshot_percentiles

router = APIRouter()


@router.get("/slo")
async def slo_snapshot() -> dict:
    return {"metrics": snapshot_percentiles()}
