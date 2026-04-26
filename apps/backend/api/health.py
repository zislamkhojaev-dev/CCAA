"""Liveness + readiness probes."""

from __future__ import annotations

from fastapi import APIRouter

from apps.backend.config import get_settings

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "app": s.app_name,
        "env": s.environment,
        "providers": {
            "stt": s.stt_provider,
            "llm": s.llm_provider,
            "tts": s.tts_provider,
        },
        "latency_budget_ms": s.latency_budget_ms,
    }
