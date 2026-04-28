"""Dialogue orchestrator: intent + RAG + LLM streaming."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
import json
import random
import re
import time
from functools import lru_cache
from typing import AsyncIterator

from apps.backend.core.bot_runtime import get_bot_runtime_payload_sync
from apps.backend.core.external_tools import check_payment_status, create_crm_ticket
from apps.backend.core.intent import IntentDetector
from apps.backend.core.route_types import RouteClass, RouteDecision, RoutePolicy
from apps.backend.core.prompts import fallback_message, fallback_message_soft, system_prompt
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

_TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_qdrant",
            "description": "Search knowledge base for supporting instructions",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_to_human",
            "description": "Escalate conversation to human operator",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_payment_status",
            "description": "Check payment status in billing system",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_crm_ticket",
            "description": "Create CRM ticket for follow-up",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "customer_text": {"type": "string"},
                },
                "required": ["reason", "customer_text"],
                "additionalProperties": False,
            },
        },
    },
]


class Orchestrator:
    def __init__(self, rag: RAGService | None = None) -> None:
        self._llm = get_llm()
        self._rag = rag or get_rag_voice()
        self._intent = IntentDetector(self._llm)
        self._semantic_cache: dict[str, tuple[float, IntentResult, list[RAGChunk]]] = {}
        self._workflow = AgentWorkflowGraph()

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
        agent_result = await self._agentic_resolve(
            text=text,
            locale=locale,
            history=history,
            intent=intent,
            sources=sources,
        )
        if agent_result.escalation_reason:
            yield self.human_handoff_message(locale)
            return
        if agent_result.answer:
            yield agent_result.answer
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

        if intent.intent in {"greeting", "thanks", "goodbye"}:
            return PlaygroundResponse(
                answer=self.smalltalk_message(intent.intent, locale),
                intent=intent,
                sources=[],
                latency_ms=latency,
            )

        agent_result = await self._agentic_resolve(
            text=text,
            locale=locale,
            history=history,
            intent=intent,
            sources=sources,
        )
        if agent_result.escalation_reason:
            return PlaygroundResponse(
                answer=self.human_handoff_message(locale),
                intent=intent,
                sources=sources,
                latency_ms=latency,
            )

        if agent_result.answer:
            return PlaygroundResponse(
                answer=agent_result.answer,
                intent=intent,
                sources=sources,
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

    @staticmethod
    def smalltalk_message(intent: str, locale: str) -> str:
        if locale == "uz":
            if intent == "greeting":
                return _pick_variant(
                    [
                        "Assalomu alaykum. Sizni eshitib turibman, qanday yordam bera olaman?",
                        "Salom. Marhamat, savolingizni ayting, yordam beraman.",
                        "Assalomu alaykum. Men shu yerdaman, nimani aniqlashtiraylik?",
                    ]
                )
            if intent == "thanks":
                return _pick_variant(
                    [
                        "Marhamat. Yana savolingiz bo‘lsa, davom etamiz.",
                        "Arzimaydi. Xohlasangiz, yana yordam beraman.",
                        "Mayli, agar yana savol bo‘lsa, birga ko‘rib chiqamiz.",
                    ]
                )
            if intent == "goodbye":
                return _pick_variant(
                    [
                        "Xayr, kuningiz yaxshi o‘tsin.",
                        "Yaxshi kun tilayman. Xayr!",
                        "Xayr, kerak bo‘lsa yana murojaat qiling.",
                    ]
                )
            return "Tushundim, davom etamiz."
        if intent == "greeting":
            return _pick_variant(
                [
                    "Здравствуйте. Чем могу помочь?",
                    "Добрый день. Я на связи, подскажите ваш вопрос.",
                    "Здравствуйте. Готов помочь, о чем хотите узнать?",
                    "Приветствую. Расскажите, пожалуйста, что вас интересует.",
                ]
            )
        if intent == "thanks":
            return _pick_variant(
                [
                    "Пожалуйста. Если нужно, помогу дальше.",
                    "Рад помочь. Если хотите, можем продолжить.",
                    "Всегда пожалуйста. Обращайтесь, если появятся вопросы.",
                ]
            )
        if intent == "goodbye":
            return _pick_variant(
                [
                    "Всего доброго.",
                    "Хорошего дня. До связи.",
                    "До свидания. Если понадобится помощь, я на линии.",
                ]
            )
        return "Понял вас, давайте продолжим."

    async def _classify_and_retrieve(
        self, text: str, locale: str, latency: dict[str, float]
    ) -> tuple[IntentResult, list[RAGChunk]]:
        cached = self._cache_get(text, locale)
        if cached:
            intent, sources = cached
            latency["intent_and_rag_ms"] = 0.0
            log.info(
                "orchestrator_cache_hit",
                intent=intent.intent,
                requires_human=intent.requires_human,
                sources=len(sources),
            )
            return intent, sources

        t0 = time.perf_counter()
        rt = get_bot_runtime_payload_sync()
        if bool(rt.get("semantic_router_fast_enabled", True)):
            route = await asyncio.to_thread(_semantic_router_dispatch, text, locale, rt)
        else:
            route = RouteDecision(
                route_class=RouteClass.COMPLEX,
                confidence=0.5,
                policy=RoutePolicy(run_intent=True, run_rag=True, allow_tooling=True),
                intent_hint=None,
                requires_human=False,
                reason="router_disabled",
            )
        wf = self._workflow.start(route.route_class)
        self._workflow.push(wf, WorkflowNode.ROUTE, note=route.reason)
        if route.route_class == RouteClass.CACHED and cached:
            self._workflow.push(wf, WorkflowNode.CACHED_RESPONSE)
            return cached
        if route.intent_hint is not None and not route.policy.run_intent:
            intent = IntentResult(
                intent=route.intent_hint,
                requires_human=route.requires_human,
                confidence=route.confidence,
            )
            self._workflow.push(wf, WorkflowNode.INTENT_HINT)
        else:
            self._workflow.push(wf, WorkflowNode.INTENT_DETECT)
            intent = await self._intent.detect(text, locale=locale)
        sources: list[RAGChunk] = []
        if route.policy.run_rag:
            self._workflow.push(wf, WorkflowNode.RAG_RETRIEVE)
            sources = await self._search_with_reason_act_observe(text, locale=locale)
        else:
            self._workflow.push(wf, WorkflowNode.NO_RAG)
        latency["intent_and_rag_ms"] = (time.perf_counter() - t0) * 1000.0
        self._cache_put(text, locale, intent, sources)
        self._workflow.push(wf, WorkflowNode.ROUTED_RESULT)
        log.info(
            "orchestrator_routed",
            route_class=route.route_class.value,
            route_confidence=route.confidence,
            intent=intent.intent,
            requires_human=intent.requires_human,
            sources=len(sources),
        )
        return intent, sources

    async def _search_with_reason_act_observe(
        self, text: str, *, locale: str, top_k: int = 5
    ) -> list[RAGChunk]:
        """Reason→Act→Observe mini-loop for RAG.

        1) Base search by user text
        2) If empty, ask LLM for refined search query
        3) Run second search with refined query (single retry)
        """
        sources = await self._rag.search(text, locale=locale, top_k=top_k)
        if sources:
            return sources
        refined = await self._refine_search_query(text=text, locale=locale)
        if not refined or refined.strip().lower() == text.strip().lower():
            return []
        sources2 = await self._rag.search(refined, locale=locale, top_k=top_k)
        if sources2:
            log.info("rag_refined_query_used", original=text[:120], refined=refined[:120], hits=len(sources2))
        return sources2

    async def _refine_search_query(self, *, text: str, locale: str) -> str:
        sys = (
            "You are a semantic query rewriter for a customer-support KB search. "
            "Given a user utterance, either return a better short search query in Russian/Uzbek "
            "or say that clarification is needed. "
            'Return ONLY JSON: {"action":"search"|"clarify","query":"..."}'
        )
        user = (
            f"Locale={locale}\n"
            f"User utterance: {text}\n"
            "If utterance is too vague/noisy, use action=clarify. "
            "If actionable, output concise search query (2-10 words)."
        )
        try:
            raw = await self._llm.complete(
                [ChatMessage(role="system", content=sys), ChatMessage(role="user", content=user)],
                temperature=0.0,
                max_tokens=80,
            )
            data = json.loads(_first_json(raw) or "{}")
            if str(data.get("action", "")).lower() != "search":
                return ""
            q = str(data.get("query", "")).strip()
            return q[:160]
        except Exception:  # noqa: BLE001
            return ""

    def _cache_get(self, text: str, locale: str) -> tuple[IntentResult, list[RAGChunk]] | None:
        rt = get_bot_runtime_payload_sync()
        ttl = max(0.0, float(rt.get("semantic_cache_ttl_sec", 90.0)))
        if ttl <= 0:
            return None
        k = _cache_key(text, locale)
        row = self._semantic_cache.get(k)
        if not row:
            return None
        ts, intent, sources = row
        if (time.monotonic() - ts) > ttl:
            self._semantic_cache.pop(k, None)
            return None
        return intent, sources

    def _cache_put(
        self, text: str, locale: str, intent: IntentResult, sources: list[RAGChunk]
    ) -> None:
        rt = get_bot_runtime_payload_sync()
        ttl = max(0.0, float(rt.get("semantic_cache_ttl_sec", 90.0)))
        if ttl <= 0:
            return
        k = _cache_key(text, locale)
        self._semantic_cache[k] = (time.monotonic(), intent, sources)
        max_entries = max(16, int(rt.get("semantic_cache_max_entries", 256)))
        if len(self._semantic_cache) > max_entries:
            # Drop oldest cache item.
            oldest_k = min(self._semantic_cache.items(), key=lambda kv: kv[1][0])[0]
            self._semantic_cache.pop(oldest_k, None)

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

    async def silence_nudge(
        self,
        *,
        locale: str = "ru",
        history: list[ChatMessage] | None = None,
    ) -> str:
        history = history or []
        rt = get_bot_runtime_payload_sync()
        prompt = (
            "Ты ведешь голосовой диалог поддержки. Система сообщает событие: user_is_silent. "
            "Сформируй короткий живой нудж (1 предложение), чтобы вернуть клиента в диалог. "
            "Без markdown, без списков, без выдуманных деталей."
            if locale != "uz"
            else "Siz ovozli qo‘llab-quvvatlash suhbatini olib boryapsiz. Tizim hodisasi: user_is_silent. "
            "Mijozni suhbatga qaytarish uchun 1 gaplik qisqa nudge yozing. Markdown ishlatmang."
        )
        msgs: list[ChatMessage] = [ChatMessage(role="system", content=prompt)]
        msgs.extend(history[-max(2, int(rt.get("history_max_messages", 24))) :])
        msgs.append(ChatMessage(role="user", content="[event:user_is_silent]"))
        nudge = await self._llm.complete(
            msgs,
            temperature=float(rt.get("llm_temperature", 0.2)),
            max_tokens=64,
        )
        return (nudge or "").strip()

    async def resolve_after_route(
        self,
        text: str,
        intent: IntentResult,
        sources: list[RAGChunk],
        *,
        locale: str = "ru",
        history: list[ChatMessage] | None = None,
    ) -> tuple[str, str | None]:
        history = history or []
        result = await self._agentic_resolve(
            text=text,
            locale=locale,
            history=history,
            intent=intent,
            sources=sources,
        )
        if result.escalation_reason:
            return self.human_handoff_message(locale), result.escalation_reason
        if result.answer:
            return result.answer, None
        if not sources:
            return fallback_message_soft(locale), None
        rt = get_bot_runtime_payload_sync()
        answer = await self._llm.complete(
            self.build_messages(text, history, sources, locale),
            temperature=float(rt.get("llm_temperature", 0.2)),
            max_tokens=int(rt.get("llm_max_tokens", 300)),
        )
        return (answer or fallback_message(locale)), None

    async def _agentic_resolve(
        self,
        *,
        text: str,
        locale: str,
        history: list[ChatMessage],
        intent: IntentResult,
        sources: list[RAGChunk],
    ) -> "_AgentResult":
        if intent.intent in {"greeting", "thanks", "goodbye"}:
            return _AgentResult(answer=self.smalltalk_message(intent.intent, locale))
        if intent.requires_human:
            return _AgentResult(escalation_reason=f"intent:{intent.intent}")
        if not sources:
            return _AgentResult(answer="")
        rt = get_bot_runtime_payload_sync()
        max_steps = max(1, int(rt.get("agent_tool_loop_max_steps", 3)))
        budget_ms = max(100, int(rt.get("agent_tool_loop_budget_ms", 1200)))
        deadline = time.monotonic() + (budget_ms / 1000.0)
        messages = self.build_messages(text, history, sources, locale)
        messages[0] = ChatMessage(
            role="system",
            content=(
                f"{messages[0].content}\n\n"
                "РАБОТА С ИНСТРУМЕНТАМИ:\n"
                'Если нужен инструмент, отвечай СТРОГО JSON: {"tool_call":{"name":"search_qdrant","arguments":{"query":"..."}}}\n'
                'или {"tool_call":{"name":"transfer_to_human","arguments":{"reason":"..."}}}.\n'
                "Если инструмент не нужен, отвечай обычным текстом для клиента.\n"
                "Не смешивай JSON и обычный текст в одном ответе."
            ),
        )
        for _ in range(max_steps):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                log.warning("agent_tool_loop_budget_exceeded", budget_ms=budget_ms)
                return _AgentResult(answer=fallback_message_soft(locale))
            raw = ""
            native_calls: list[dict] = []
            if rt.get("native_function_calling_enabled", True):
                try:
                    raw, native_calls = await asyncio.wait_for(
                        self._llm.complete_with_tools(
                            messages,
                            tools=_TOOL_SCHEMAS,
                            temperature=float(rt.get("llm_temperature", 0.2)),
                            max_tokens=int(rt.get("llm_max_tokens", 300)),
                        ),
                        timeout=remaining,
                    )
                except TimeoutError:
                    log.warning("agent_tool_loop_llm_timeout", budget_ms=budget_ms)
                    return _AgentResult(answer=fallback_message_soft(locale))
            else:
                try:
                    raw = await asyncio.wait_for(
                        self._llm.complete(
                            messages,
                            temperature=float(rt.get("llm_temperature", 0.2)),
                            max_tokens=int(rt.get("llm_max_tokens", 300)),
                        ),
                        timeout=remaining,
                    )
                except TimeoutError:
                    log.warning("agent_tool_loop_llm_timeout", budget_ms=budget_ms)
                    return _AgentResult(answer=fallback_message_soft(locale))
            if native_calls:
                tool_name = str(native_calls[0].get("name") or "")
                args = native_calls[0].get("arguments") or {}
                if not isinstance(args, dict):
                    args = {}
            else:
                tool_call, parse_error = _parse_tool_call(raw)
                if parse_error:
                    messages.append(
                        ChatMessage(
                            role="system",
                            content='TOOL_ERROR: {"error":"invalid_tool_call_json","hint":"return valid JSON tool_call"}',
                        )
                    )
                    continue
                if tool_call is None:
                    return _AgentResult(answer=(raw or "").strip())
                tool_name, args = tool_call
            if tool_name == "transfer_to_human":
                reason = str(args.get("reason") or "llm_transfer")
                if _allow_transfer_to_human(intent=intent, text=text):
                    return _AgentResult(escalation_reason=reason[:256])
                return _AgentResult(answer=fallback_message_soft(locale))
            if tool_name != "search_qdrant":
                if tool_name == "check_payment_status":
                    query = str(args.get("query") or text).strip()
                    result = await check_payment_status(query=query)
                    messages.append(ChatMessage(role="assistant", content=(raw or "").strip()))
                    messages.append(
                        ChatMessage(
                            role="system",
                            content=f"TOOL_RESULT: {json.dumps({'tool': tool_name, 'result': result}, ensure_ascii=False)}",
                        )
                    )
                    continue
                if tool_name == "create_crm_ticket":
                    reason = str(args.get("reason") or "follow_up")
                    customer_text = str(args.get("customer_text") or text)
                    result = await create_crm_ticket(reason=reason, customer_text=customer_text)
                    messages.append(ChatMessage(role="assistant", content=(raw or "").strip()))
                    messages.append(
                        ChatMessage(
                            role="system",
                            content=f"TOOL_RESULT: {json.dumps({'tool': tool_name, 'result': result}, ensure_ascii=False)}",
                        )
                    )
                    continue
                messages.append(
                    ChatMessage(
                        role="system",
                        content=f'{{"error":"unknown_tool","name":"{tool_name}"}}',
                    )
                )
                continue
            query = str(args.get("query") or "").strip()
            if not query:
                messages.append(
                    ChatMessage(
                        role="system",
                        content='{"error":"invalid_arguments","details":"query is required"}',
                    )
                )
                continue
            found = await self._rag.search(query, locale=locale, top_k=5)
            tool_payload = {
                "tool": "search_qdrant",
                "query": query,
                "result_count": len(found),
                "context": self._rag.format_context(found),
            }
            messages.append(ChatMessage(role="assistant", content=raw.strip()))
            messages.append(
                ChatMessage(role="system", content=f"TOOL_RESULT: {json.dumps(tool_payload, ensure_ascii=False)}")
            )
        return _AgentResult(answer=fallback_message_soft(locale))


@lru_cache(maxsize=1)
def get_orchestrator() -> Orchestrator:
    return Orchestrator(rag=get_rag_voice())


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _first_json(text: str) -> str | None:
    m = _JSON_RE.search(text or "")
    return m.group(0) if m else None


def _cache_key(text: str, locale: str) -> str:
    norm = " ".join((text or "").strip().lower().split())
    return f"{locale}:{norm}"


@dataclass(slots=True)
class _AgentResult:
    answer: str = ""
    escalation_reason: str | None = None


def _parse_tool_call(raw: str) -> tuple[tuple[str, dict] | None, bool]:
    text = (raw or "").strip()
    if "tool_call" not in text:
        return None, False
    payload_text = _first_json(text)
    if not payload_text:
        return None, True
    try:
        data = json.loads(payload_text)
    except Exception:  # noqa: BLE001
        return None, True
    tc = data.get("tool_call")
    if not isinstance(tc, dict):
        return None, False
    name = str(tc.get("name") or "").strip()
    args = tc.get("arguments") or {}
    if not name:
        return None, True
    if not isinstance(args, dict):
        args = {}
    return (name, args), False


def _semantic_router_dispatch(text: str, locale: str, runtime: dict) -> RouteDecision:
    """Embedding-based semantic router, then keyword/heuristic fallback."""
    try:
        from apps.backend.core import semantic_local as sem

        if bool(runtime.get("semantic_router_embed_enabled", True)):
            rd = sem.route_by_embedding(text, locale=locale, runtime=runtime)
            if rd is not None:
                return rd
    except Exception as exc:  # noqa: BLE001
        log.warning("semantic_embed_router_failed", error=str(exc))
    return _semantic_router_keywords(text, locale=locale, runtime=runtime)


def _semantic_router_keywords(text: str, *, locale: str, runtime: dict) -> RouteDecision:
    del locale
    t = " ".join((text or "").strip().lower().split())
    noise_max_words = max(1, int(runtime.get("router_noise_max_words", 2)))
    noise_max_chars = max(4, int(runtime.get("router_noise_max_chars", 14)))
    simple_max_words = max(2, int(runtime.get("router_simple_max_words", 5)))
    simple_conf = float(runtime.get("router_simple_min_confidence", 0.7))
    esc_conf = float(runtime.get("router_escalation_min_confidence", 0.95))
    noise_run_intent = bool(runtime.get("router_policy_noise_run_intent", False))
    noise_run_rag = bool(runtime.get("router_policy_noise_run_rag", False))
    simple_run_intent = bool(runtime.get("router_policy_simple_run_intent", True))
    simple_run_rag = bool(runtime.get("router_policy_simple_run_rag", True))
    complex_run_intent = bool(runtime.get("router_policy_complex_run_intent", True))
    complex_run_rag = bool(runtime.get("router_policy_complex_run_rag", True))
    if not t:
        return RouteDecision(
            route_class=RouteClass.NOISE,
            confidence=0.95,
            policy=RoutePolicy(run_intent=noise_run_intent, run_rag=noise_run_rag, allow_tooling=False),
            intent_hint="other",
            requires_human=False,
            reason="empty_or_silence",
        )
    if any(k in t for k in ("оператор", "человек", "human", "agent", "переключи", "позови")):
        return RouteDecision(
            route_class=RouteClass.ESCALATION,
            confidence=esc_conf,
            policy=RoutePolicy(run_intent=False, run_rag=False, allow_tooling=True),
            intent_hint="human_agent",
            requires_human=True,
            reason="explicit_handoff_request",
        )
    if any(k in t for k in ("привет", "здравств", "добрый", "hello", "hi", "salom", "assalomu")):
        return RouteDecision(
            route_class=RouteClass.SIMPLE,
            confidence=0.95,
            policy=RoutePolicy(run_intent=False, run_rag=False, allow_tooling=False),
            intent_hint="greeting",
            requires_human=False,
            reason="smalltalk_greeting",
        )
    if any(k in t for k in ("спасибо", "благодар", "rahmat", "thank")):
        return RouteDecision(
            route_class=RouteClass.SIMPLE,
            confidence=0.95,
            policy=RoutePolicy(run_intent=False, run_rag=False, allow_tooling=False),
            intent_hint="thanks",
            requires_human=False,
            reason="smalltalk_thanks",
        )
    if any(k in t for k in ("пока", "до свид", "goodbye", "bye", "xayr")):
        return RouteDecision(
            route_class=RouteClass.SIMPLE,
            confidence=0.95,
            policy=RoutePolicy(run_intent=False, run_rag=False, allow_tooling=False),
            intent_hint="goodbye",
            requires_human=False,
            reason="smalltalk_goodbye",
        )
    words = len(re.sub(r"[^\w\s]", " ", t).split())
    if words <= noise_max_words and len(t) <= noise_max_chars:
        return RouteDecision(
            route_class=RouteClass.NOISE,
            confidence=0.8,
            policy=RoutePolicy(run_intent=noise_run_intent, run_rag=noise_run_rag, allow_tooling=False),
            intent_hint="other",
            requires_human=False,
            reason="low_signal_short_utterance",
        )
    if words <= simple_max_words:
        return RouteDecision(
            route_class=RouteClass.SIMPLE,
            confidence=simple_conf,
            policy=RoutePolicy(run_intent=simple_run_intent, run_rag=simple_run_rag, allow_tooling=True),
            intent_hint=None,
            requires_human=False,
            reason="simple_consultative_query",
        )
    return RouteDecision(
        route_class=RouteClass.COMPLEX,
        confidence=0.65,
        policy=RoutePolicy(run_intent=complex_run_intent, run_rag=complex_run_rag, allow_tooling=True),
        intent_hint=None,
        requires_human=False,
        reason="complex_query",
    )


class WorkflowNode(str, Enum):
    ROUTE = "route"
    CACHED_RESPONSE = "cached_response"
    INTENT_HINT = "intent_hint"
    INTENT_DETECT = "intent_detect"
    RAG_RETRIEVE = "rag_retrieve"
    NO_RAG = "no_rag"
    ROUTED_RESULT = "routed_result"


@dataclass(slots=True)
class _WorkflowState:
    route_class: RouteClass
    visited: list[WorkflowNode]


class AgentWorkflowGraph:
    def __init__(self) -> None:
        self._edges: dict[WorkflowNode, set[WorkflowNode]] = {
            WorkflowNode.ROUTE: {
                WorkflowNode.CACHED_RESPONSE,
                WorkflowNode.INTENT_HINT,
                WorkflowNode.INTENT_DETECT,
            },
            WorkflowNode.INTENT_HINT: {WorkflowNode.RAG_RETRIEVE, WorkflowNode.NO_RAG},
            WorkflowNode.INTENT_DETECT: {WorkflowNode.RAG_RETRIEVE, WorkflowNode.NO_RAG},
            WorkflowNode.RAG_RETRIEVE: {WorkflowNode.ROUTED_RESULT},
            WorkflowNode.NO_RAG: {WorkflowNode.ROUTED_RESULT},
            WorkflowNode.CACHED_RESPONSE: set(),
            WorkflowNode.ROUTED_RESULT: set(),
        }

    def start(self, route_class: RouteClass) -> _WorkflowState:
        return _WorkflowState(route_class=route_class, visited=[])

    def push(self, state: _WorkflowState, node: WorkflowNode, *, note: str = "") -> None:
        if state.visited:
            prev = state.visited[-1]
            if node not in self._edges.get(prev, set()):
                log.warning(
                    "workflow_edge_violation",
                    from_node=prev.value,
                    to_node=node.value,
                    route_class=state.route_class.value,
                    note=note,
                )
        state.visited.append(node)


def _pick_variant(options: list[str]) -> str:
    if not options:
        return ""
    # Small non-deterministic variation so greetings are less repetitive.
    return random.choice(options)


def _allow_transfer_to_human(*, intent: IntentResult, text: str) -> bool:
    if intent.requires_human or intent.intent in {"human_agent", "block_card", "transfer_money", "personal_data"}:
        return True
    t = (text or "").lower()
    trigger = (
        "оператор",
        "человек",
        "переключи",
        "позови",
        "жалоб",
        "возврат",
        "блокир",
        "персональн",
    )
    return any(k in t for k in trigger)
