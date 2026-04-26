"""Глобальные настройки поведения бота (температура, история, приветствия, TTS по умолчанию)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from apps.backend.core.bot_runtime import get_bot_runtime_payload, save_bot_runtime_payload

router = APIRouter()


@router.get("/bot-settings")
async def get_bot_settings() -> dict[str, Any]:
    return await get_bot_runtime_payload()


@router.put("/bot-settings")
async def put_bot_settings(body: dict[str, Any]) -> dict[str, Any]:
    return await save_bot_runtime_payload(body)
