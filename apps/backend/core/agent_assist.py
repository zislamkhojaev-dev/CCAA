"""Agent Assist (Суфлёр) — real-time prompts for human operators.

Wire format (WebSocket `/ws/agent-assist`):
  client → server   text/JSON   {"type": "utterance", "speaker": "customer|agent", "text": "..."}
                                {"type": "reset"}
  server → client   text/JSON   {"type": "suggestion", "intent": ..., "answer": "...",
                                 "sources": [...], "latency_ms": {...}}
                                {"type": "warning",  "message": "..."}

Suggestions are produced only on customer utterances (we don't suggest
to the agent what they themselves should say back to themselves). When
intent is operational (block_card, transfer_money, etc.) we surface a
warning so the operator handles it personally instead of trusting an
AI-generated answer.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal

from apps.backend.core.intent import IntentDetector
from apps.backend.core.prompts import system_prompt
from apps.backend.models.schemas import ChatMessage, IntentResult, RAGChunk
from apps.backend.rag import get_rag_agent_assist
from apps.backend.services import get_llm
from apps.backend.utils.logging import get_logger

log = get_logger(__name__)

Speaker = Literal["customer", "agent"]


@dataclass(slots=True)
class Suggestion:
    answer: str
    intent: IntentResult
    sources: list[RAGChunk] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    warning: str | None = None


class AgentAssist:
    """Per-call session: keeps the rolling transcript + serves suggestions."""

    def __init__(self) -> None:
        self._llm = get_llm()
        self._rag = get_rag_agent_assist()
        self._intent = IntentDetector(self._llm)
        self._transcript: list[tuple[Speaker, str]] = []

    def reset(self) -> None:
        self._transcript = []

    def add_agent(self, text: str) -> None:
        if text.strip():
            self._transcript.append(("agent", text.strip()))

    async def on_customer_utterance(self, text: str, *, locale: str = "ru") -> Suggestion:
        text = text.strip()
        if not text:
            return Suggestion(
                answer="",
                intent=IntentResult(intent="other", confidence=0.0),
                warning="empty_utterance",
            )

        self._transcript.append(("customer", text))

        latency: dict[str, float] = {}
        t0 = time.perf_counter()
        intent_task = asyncio.create_task(self._intent.detect(text, locale=locale))
        rag_task = asyncio.create_task(self._rag.search(text, locale=locale, top_k=4))
        intent, sources = await asyncio.gather(intent_task, rag_task)
        latency["intent_and_rag_ms"] = (time.perf_counter() - t0) * 1000.0

        if intent.requires_human:
            return Suggestion(
                answer=_handoff_hint(locale, intent.intent),
                intent=intent,
                sources=sources,
                latency_ms=latency,
                warning=(
                    "Операционный запрос — обработайте лично, не зачитывайте подсказку."
                    if locale == "ru"
                    else "Operatsion so‘rov — shaxsan hal qiling, taklifni o‘qimang."
                ),
            )

        if not sources:
            return Suggestion(
                answer="",
                intent=intent,
                sources=[],
                latency_ms=latency,
                warning=(
                    "В базе знаний нет ответа. Уточните вопрос или предложите перевод."
                    if locale == "ru"
                    else "Bilim bazasida javob yo‘q. Savolni aniqlashtiring."
                ),
            )

        t1 = time.perf_counter()
        answer = await self._llm.complete(
            self._build_messages(text, sources, locale),
            temperature=0.2,
            max_tokens=180,
        )
        latency["llm_ms"] = (time.perf_counter() - t1) * 1000.0

        return Suggestion(
            answer=answer.strip(),
            intent=intent,
            sources=sources,
            latency_ms=latency,
        )

    def _build_messages(
        self, text: str, sources: list[RAGChunk], locale: str
    ) -> list[ChatMessage]:
        context = self._rag.format_context(sources)
        sys = system_prompt(locale, context=context)
        sys += "\n" + (
            "Ты помогаешь оператору контакт-центра. Сформулируй короткую (1–2 фразы) "
            "подсказку, которую оператор сможет произнести клиенту. Без markdown."
            if locale == "ru"
            else "Sen call-markaz operatoriga yordam berasan. Operator mijozga aytishi "
            "uchun qisqa (1–2 jumla) tavsiya yoz. Markdown ishlatma."
        )
        msgs: list[ChatMessage] = [ChatMessage(role="system", content=sys)]
        for speaker, t in self._transcript[-6:]:
            role = "user" if speaker == "customer" else "assistant"
            msgs.append(ChatMessage(role=role, content=t))
        msgs.append(ChatMessage(role="user", content=text))
        return msgs


def _handoff_hint(locale: str, intent: str) -> str:
    if locale == "uz":
        return "Operator o‘zi javob bersin — bu shaxsiy yoki moliyaviy operatsiya."
    return "Обработайте сами: операция со счётом или персональными данными клиента."


async def stream_suggestions(
    session: AgentAssist,
    customer_text: str,
    *,
    locale: str = "ru",
) -> AsyncIterator[str]:
    """Optional helper if a UI wants tokens streamed for the suggestion."""
    suggestion = await session.on_customer_utterance(customer_text, locale=locale)
    if suggestion.answer:
        yield suggestion.answer
