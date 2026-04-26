"""Единая строка настроек бота (админка) + синхронный снимок для Voice Engine."""

from __future__ import annotations

import copy
from typing import Any

from apps.backend.models.db import session_scope
from apps.backend.models.entities import BotRuntimeSettings
from apps.backend.utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_PAYLOAD: dict[str, Any] = {
    "llm_temperature": 0.2,
    "llm_max_tokens": 300,
    "history_max_messages": 24,
    "system_prompt_suffix": "",
    "suppress_repeated_greeting": True,
    "openai_tts_speed": 1.0,
    "elevenlabs_stability": 0.5,
    "elevenlabs_similarity_boost": 0.75,
    "barge_in_rms_threshold": 0.12,
    "barge_in_cooldown_ms": 700,
    "barge_in_hold_frames": 3,
    "stt_voice_rms_threshold": 0.008,
    "stt_min_voiced_seconds": 0.2,
    "stt_duplicate_cooldown_sec": 4.0,
}

_sync_merged: dict[str, Any] = copy.deepcopy(DEFAULT_PAYLOAD)


def invalidate_bot_runtime_cache() -> None:
    global _sync_merged
    _sync_merged = copy.deepcopy(DEFAULT_PAYLOAD)
    log.info("bot_runtime_cache_invalidated")


def get_bot_runtime_payload_sync() -> dict[str, Any]:
    return copy.deepcopy(_sync_merged)


async def get_bot_runtime_payload() -> dict[str, Any]:
    """Загрузка из БД и обновление синхронного снимка (для STT/TTS без async)."""
    global _sync_merged
    merged = copy.deepcopy(DEFAULT_PAYLOAD)
    try:
        async with session_scope() as s:
            row = await s.get(BotRuntimeSettings, 1)
            if row and isinstance(row.payload, dict):
                merged.update(row.payload)
    except Exception as exc:  # noqa: BLE001
        log.warning("bot_runtime_load_failed", error=str(exc))
    _sync_merged = merged
    return copy.deepcopy(merged)


async def save_bot_runtime_payload(payload: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_PAYLOAD)
    async with session_scope() as s:
        row = await s.get(BotRuntimeSettings, 1)
        if row and isinstance(row.payload, dict):
            merged.update(row.payload)
        merged.update(payload)
        if row is None:
            s.add(BotRuntimeSettings(id=1, payload=merged))
        else:
            row.payload = merged
    invalidate_bot_runtime_cache()
    await get_bot_runtime_payload()
    return copy.deepcopy(_sync_merged)
