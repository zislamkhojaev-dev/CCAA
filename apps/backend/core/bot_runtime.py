"""Единая строка настроек бота (админка) + синхронный снимок для Voice Engine."""

from __future__ import annotations

import asyncio
import copy
import time
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
    "ws_voice_max_connections": 50,
    "ws_voice_max_connections_per_ip": 8,
    "ws_voice_max_connections_per_token": 4,
    "ws_agent_assist_max_connections": 100,
    "ws_agent_assist_max_connections_per_ip": 12,
    "ws_agent_assist_max_connections_per_token": 6,
    # VAD (Silero ONNX / WebRTC) — RMS threshold only used by legacy STT-side gates if any
    "stt_voice_rms_threshold": 0.008,
    "vad_silero_speech_threshold": 0.45,
    "vad_webrtc_aggressiveness": 2,
    "vad_prebuffer_sec": 0.3,
    "vad_hangover_sec": 0.25,
    "stt_min_voiced_seconds": 0.2,
    "stt_duplicate_cooldown_sec": 4.0,
    # After STT emits is_final, wait this long for another final before starting a turn
    # (reduces cutting off mid-sentence on streaming ASR). 0 disables debounce.
    "stt_final_debounce_ms": 450,
    "silence_nudge_enabled": True,
    "silence_timeout_sec": 12.0,
    "silence_nudge_cooldown_sec": 20.0,
    "semantic_cache_ttl_sec": 90.0,
    "semantic_cache_max_entries": 256,
    "agent_tool_loop_max_steps": 3,
    "agent_tool_loop_budget_ms": 1200,
    "native_function_calling_enabled": True,
    "rag_hybrid_enabled": True,
    "rag_hybrid_alpha": 0.65,
    "rag_candidate_pool_size": 30,
    "rag_query_embed_cache_ttl_sec": 300.0,
    "rag_query_embed_cache_max_entries": 4096,
    # Semantic router + dialog stability preset
    "semantic_router_fast_enabled": True,
    "semantic_router_embed_enabled": True,
    "router_embed_escalation_cos": 0.42,
    "router_embed_smalltalk_cos": 0.40,
    "router_embed_knowledge_cos": 0.36,
    "router_embed_noise_max_cos": 0.34,
    "semantic_intent_embed_enabled": True,
    "semantic_intent_min_cos": 0.38,
    "semantic_intent_op_min_cos": 0.40,
    "semantic_intent_human_min_cos": 0.58,
    "semantic_intent_human_greeting_ambiguity_max_gap": 0.15,
    "router_embed_escalation_vs_smalltalk_margin": 0.04,
    "router_smalltalk_max_words": 3,
    "router_smalltalk_vs_knowledge_margin": 0.03,
    "low_signal_reply_enabled": True,
    # Earlier TTS kickoff from smaller text batches.
    "tts_micro_batch_chars": 40,
    # Fallback anti-spam
    "fallback_cooldown_sec": 2.5,
    "fallback_message_ru": "",
    "fallback_message_uz": "",
    "fallback_message_soft_ru": "",
    "fallback_message_soft_uz": "",
    "low_signal_message_ru": "",
    "low_signal_message_uz": "",
    "smalltalk_deterministic_enabled": False,
    # Rich semantic router thresholds/policies
    "router_noise_max_words": 2,
    "router_noise_max_chars": 14,
    "router_simple_max_words": 5,
    "router_simple_min_confidence": 0.7,
    "router_escalation_min_confidence": 0.95,
    "router_policy_noise_run_intent": False,
    "router_policy_noise_run_rag": False,
    "router_policy_simple_run_intent": True,
    "router_policy_simple_run_rag": True,
    "router_policy_complex_run_intent": True,
    "router_policy_complex_run_rag": True,
    # Case manager SLA knobs.
    "case_clarification_max_attempts": 2,
    "case_fallback_handoff_limit": 2,
    "case_low_confidence_threshold": 0.45,
    "case_confirm_intent_min_confidence": 0.72,
    # Hybrid phrasing: keep process deterministic, wording dynamic.
    "case_use_dynamic_phrasing": True,
    "case_dynamic_phrase_cache_ttl_sec": 600.0,
    "case_dynamic_phrase_cache_max_entries": 512,
    "case_strict_template_intents": ["block_card", "transfer_money", "personal_data"],
    "case_strict_template_slots": ["card_last4", "customer_id", "source_account", "target_account"],
    "case_intent_confirmation_templates": {
        "default": {
            "ru": "Правильно понимаю, что вопрос по теме «{intent}»?",
            "uz": "To'g'ri tushundimmi, savol «{intent}» mavzusi bo'yichami?",
        },
        "block_card": {
            "ru": "Верно понимаю, что нужно помочь с блокировкой карты?",
            "uz": "To'g'ri tushundimmi, kartani bloklash bo'yicha yordam kerakmi?",
        },
        "transfer_money": {
            "ru": "Правильно понимаю, что проблема связана с переводом денег?",
            "uz": "To'g'ri tushundimmi, muammo pul o'tkazmasi bilan bog'liqmi?",
        },
        "complaint": {
            "ru": "Верно понимаю, что вы хотите оформить жалобу?",
            "uz": "To'g'ri tushundimmi, siz shikoyat qoldirmoqchimisiz?",
        },
    },
    "case_slot_question_templates": {
        "default": {
            "ru": "Уточните, пожалуйста, дополнительную деталь по вашему запросу.",
            "uz": "Iltimos, masalani aniqlashtirish uchun qo'shimcha ma'lumot bering.",
        },
        "card_last4": {
            "ru": "Подскажите, пожалуйста, последние 4 цифры карты.",
            "uz": "Iltimos, kartaning oxirgi 4 raqamini ayting.",
        },
        "incident_time": {
            "ru": "Когда примерно возникла проблема?",
            "uz": "Muammo qachon yuz berdi?",
        },
        "channel": {
            "ru": "Через какой канал была операция: приложение, сайт или банкомат?",
            "uz": "Operatsiya qaysi kanal orqali bo'lgan: ilova, sayt yoki bankomat?",
        },
        "source_account": {
            "ru": "С какого счета выполняли перевод?",
            "uz": "Pul o'tkazma qaysi hisobdan yuborilgan?",
        },
        "target_account": {
            "ru": "На какой счет или карту отправляли перевод?",
            "uz": "Pul qaysi hisob yoki kartaga yuborilgan?",
        },
        "amount": {
            "ru": "Уточните сумму операции.",
            "uz": "Operatsiya summasini ayting.",
        },
        "customer_id": {
            "ru": "Назовите, пожалуйста, номер договора или ID клиента.",
            "uz": "Shartnoma raqami yoki mijoz ID sini ayting.",
        },
        "field_to_change": {
            "ru": "Какие именно данные нужно изменить?",
            "uz": "Qaysi ma'lumotni o'zgartirmoqchisiz?",
        },
        "reason": {
            "ru": "Уточните, пожалуйста, причину обращения.",
            "uz": "Murojaat sababini aniqlashtiring.",
        },
        "topic": {
            "ru": "По какому вопросу хотите оставить обращение?",
            "uz": "Murojaat mavzusini ayting.",
        },
        "details": {
            "ru": "Опишите, пожалуйста, проблему чуть подробнее.",
            "uz": "Muammoni biroz batafsil tasvirlab bering.",
        },
        "product_name": {
            "ru": "Какой именно продукт или тариф вас интересует?",
            "uz": "Qaysi mahsulot yoki tarif sizni qiziqtiryapti?",
        },
    },
}

_sync_merged: dict[str, Any] = copy.deepcopy(DEFAULT_PAYLOAD)
_last_refresh_monotonic = 0.0


def invalidate_bot_runtime_cache() -> None:
    global _sync_merged
    _sync_merged = copy.deepcopy(DEFAULT_PAYLOAD)
    log.info("bot_runtime_cache_invalidated")


def get_bot_runtime_payload_sync() -> dict[str, Any]:
    return copy.deepcopy(_sync_merged)


async def get_bot_runtime_payload() -> dict[str, Any]:
    """Загрузка из БД и обновление синхронного снимка (для STT/TTS без async)."""
    global _sync_merged, _last_refresh_monotonic
    merged = copy.deepcopy(DEFAULT_PAYLOAD)
    try:
        async with session_scope() as s:
            row = await s.get(BotRuntimeSettings, 1)
            if row and isinstance(row.payload, dict):
                merged.update(row.payload)
    except Exception as exc:  # noqa: BLE001
        log.warning("bot_runtime_load_failed", error=str(exc))
    _sync_merged = merged
    _last_refresh_monotonic = time.monotonic()
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


async def periodic_bot_runtime_refresh(stop_event: asyncio.Event) -> None:
    """Periodic DB refresh so each worker converges to the latest runtime config."""
    while not stop_event.is_set():
        try:
            await get_bot_runtime_payload()
        except Exception as exc:  # noqa: BLE001
            log.warning("bot_runtime_periodic_refresh_failed", error=str(exc))
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=5.0)
        except TimeoutError:
            continue
