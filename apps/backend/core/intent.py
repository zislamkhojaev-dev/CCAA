"""Lightweight intent detector.

A first-pass keyword router gives sub-millisecond latency for the
operational vs. consultative split (FT requirement). Anything ambiguous
is escalated to the LLM for a JSON-shaped classification — still well
within the 1.5-second budget because we run it concurrently with RAG.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from apps.backend.models.schemas import ChatMessage, IntentResult
from apps.backend.services import LLMService


# Operational intents always require a human (account-changing actions).
_OPERATIONAL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "block_card": ("заблокир", "карту утер", "украл", "украдена", "kartani blok"),
    "transfer_money": ("перевод денег", "перевести с", "pul o‘tkaz", "pul jo‘nat"),
    "personal_data": ("сменить паспорт", "паспортные данные", "pasport ma’lumot"),
    "complaint": ("жалоб", "shikoyat"),
    "human_agent": ("оператор", "человек", "operator", "odam"),
}

# Consultative intents the bot is allowed to answer from the KB.
_CONSULTATIVE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tariffs": ("тариф", "стоимост", "комисси", "tarif", "narx"),
    "products": ("продукт", "услуг", "mahsulot", "xizmat"),
    "office_hours": ("график", "режим работ", "ish vaqt"),
    "branches": ("филиал", "отделени", "filial"),
}


@dataclass(slots=True)
class _Match:
    intent: str
    requires_human: bool
    confidence: float


class IntentDetector:
    def __init__(self, llm: LLMService) -> None:
        self._llm = llm

    async def detect(self, text: str, *, locale: str = "ru") -> IntentResult:
        match = self._keyword_match(text.lower())
        if match is not None:
            return IntentResult(
                intent=match.intent,
                confidence=match.confidence,
                requires_human=match.requires_human,
            )
        return await self._llm_classify(text, locale=locale)

    @staticmethod
    def _keyword_match(text: str) -> _Match | None:
        for intent, kws in _OPERATIONAL_KEYWORDS.items():
            if any(kw in text for kw in kws):
                return _Match(intent=intent, requires_human=True, confidence=0.95)
        for intent, kws in _CONSULTATIVE_KEYWORDS.items():
            if any(kw in text for kw in kws):
                return _Match(intent=intent, requires_human=False, confidence=0.85)
        return None

    async def _llm_classify(self, text: str, *, locale: str) -> IntentResult:
        sys = (
            "Classify the user's request into one of: "
            "tariffs, products, office_hours, branches, "
            "block_card, transfer_money, personal_data, complaint, human_agent, other. "
            'Respond ONLY with JSON: {"intent": "...", "requires_human": true|false, "confidence": 0.0-1.0}.'
        )
        try:
            raw = await self._llm.complete(
                [
                    ChatMessage(role="system", content=sys),
                    ChatMessage(role="user", content=text),
                ],
                temperature=0.0,
                max_tokens=80,
            )
            data = json.loads(_first_json_object(raw) or "{}")
            intent = str(data.get("intent", "other"))
            return IntentResult(
                intent=intent,
                confidence=float(data.get("confidence", 0.5)),
                requires_human=bool(data.get("requires_human", False))
                or intent in {"block_card", "transfer_money", "personal_data", "complaint", "human_agent"},
            )
        except Exception:  # noqa: BLE001
            return IntentResult(intent="other", confidence=0.3, requires_human=False)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _first_json_object(text: str) -> str | None:
    m = _JSON_RE.search(text or "")
    return m.group(0) if m else None
