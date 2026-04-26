"""Dialogue orchestrator: intent + RAG + LLM streaming."""

from __future__ import annotations

import asyncio
import time
from functools import lru_cache
from typing import AsyncIterator

from apps.backend.core.bot_runtime import get_bot_runtime_payload_sync
from apps.backend.core.intent import IntentDetector
from apps.backend.core.prompts import fallback_message, system_prompt
from apps.backend.models.schemas import (
    ChatMessage,
    IntentResult,
    PlaygroundResponse,
    RAGChunk,
)
from apps.backend.rag import RAGService, get_rag_voice
from apps.backend.services import get_llm
from apps.backend.utils.logging import get_logger

log = get_logger(__name__)


class Orchestrator:
    def __init__(self, rag: RAGService | None = None) -> None:
        self._llm = get_llm()
        self._rag = rag or get_rag_voice()
        self._intent = IntentDetector(self._llm)

    async def preview_route(
        self, text: str, *, locale: str = "ru"
    ) -> tuple[IntentResult, list[RAGChunk]]:
        """Intent + RAG без генерации ответа (для Voice Engine / запись транскрипта)."""
        latency: dict[str, float] = {}
        return await self._classify_and_retrieve(text, locale, latency)

    async def stream_answer_after_route(
        self,
        text: str,
        intent: IntentResult,
        sources: list[RAGChunk],
        *,
        locale: str = "ru",
        history: list[ChatMessage] | None = None,
    ) -> AsyncIterator[str]:
        """Поток ответа при уже известном intent/sources (без повторной классификации)."""
        history = history or []
        if intent.requires_human:
            yield self.human_handoff_message(locale)
            return
        if not sources:
            yield fallback_message(locale)
            return
        rt = get_bot_runtime_payload_sync()
        async for delta in self._llm.stream_complete(
            self.build_messages(text, history, sources, locale),
            temperature=float(rt.get("llm_temperature", 0.2)),
            max_tokens=int(rt.get("llm_max_tokens", 300)),
        ):
            yield delta

    async def answer(
        self,
        text: str,
        *,
        locale: str = "ru",
        history: list[ChatMessage] | None = None,
    ) -> PlaygroundResponse:
        history = history or []
        latency: dict[str, float] = {}

        intent, sources = await self._classify_and_retrieve(text, locale, latency)

        if intent.requires_human:
            return PlaygroundResponse(
                answer=self.human_handoff_message(locale),
                intent=intent,
                sources=sources,
                latency_ms=latency,
            )

        if not sources:
            return PlaygroundResponse(
                answer=fallback_message(locale),
                intent=intent,
                sources=[],
                latency_ms=latency,
            )

        rt = get_bot_runtime_payload_sync()
        t0 = time.perf_counter()
        answer = await self._llm.complete(
            self.build_messages(text, history, sources, locale),
            temperature=float(rt.get("llm_temperature", 0.2)),
            max_tokens=int(rt.get("llm_max_tokens", 300)),
        )
        latency["llm_ms"] = (time.perf_counter() - t0) * 1000.0

        return PlaygroundResponse(
            answer=answer or fallback_message(locale),
            intent=intent,
            sources=sources,
            latency_ms=latency,
        )

    async def stream_answer(
        self,
        text: str,
        *,
        locale: str = "ru",
        history: list[ChatMessage] | None = None,
    ) -> AsyncIterator[str]:
        """Плейграунд / совместимость: классификация + поток."""
        history = history or []
        latency: dict[str, float] = {}
        intent, sources = await self._classify_and_retrieve(text, locale, latency)
        async for delta in self.stream_answer_after_route(
            text, intent, sources, locale=locale, history=history
        ):
            yield delta

    @staticmethod
    def human_handoff_message(locale: str) -> str:
        if locale == "uz":
            return "Bu savolni operator hal qiladi. Sizni hozir ulayman."
        return "По этому вопросу вас лучше переключить на оператора. Соединяю."

    async def _classify_and_retrieve(
        self, text: str, locale: str, latency: dict[str, float]
    ) -> tuple[IntentResult, list[RAGChunk]]:
        t0 = time.perf_counter()
        intent_task = asyncio.create_task(self._intent.detect(text, locale=locale))
        rag_task = asyncio.create_task(self._rag.search(text, locale=locale, top_k=5))
        intent, sources = await asyncio.gather(intent_task, rag_task)
        latency["intent_and_rag_ms"] = (time.perf_counter() - t0) * 1000.0
        log.info(
            "orchestrator_routed",
            intent=intent.intent,
            requires_human=intent.requires_human,
            sources=len(sources),
        )
        return intent, sources

    def build_messages(
        self,
        text: str,
        history: list[ChatMessage],
        sources: list[RAGChunk],
        locale: str,
    ) -> list[ChatMessage]:
        rt = get_bot_runtime_payload_sync()
        max_hist = max(2, int(rt.get("history_max_messages", 24)))
        context = self._rag.format_context(sources)
        has_prior_assistant = any(m.role == "assistant" for m in history)
        suppress = bool(rt.get("suppress_repeated_greeting", True)) and has_prior_assistant
        suffix = str(rt.get("system_prompt_suffix") or "")
        messages: list[ChatMessage] = [
            ChatMessage(
                role="system",
                content=system_prompt(
                    locale,
                    context=context,
                    suffix=suffix,
                    suppress_repeated_greeting=suppress,
                ),
            )
        ]
        messages.extend(history[-max_hist:])
        messages.append(ChatMessage(role="user", content=text))
        return messages


@lru_cache(maxsize=1)
def get_orchestrator() -> Orchestrator:
    return Orchestrator(rag=get_rag_voice())
