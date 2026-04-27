"""Smoke tests for the keyword-based intent router (no network required)."""

from __future__ import annotations

import pytest

from apps.backend.models.schemas import ChatMessage
from apps.backend.core.intent import IntentDetector
from apps.backend.services.mocks import MockLLM


class _StubLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    async def complete(
        self,
        _messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 80,
    ) -> str:
        return self._response


@pytest.mark.asyncio
async def test_block_card_routed_to_human() -> None:
    detector = IntentDetector(MockLLM())
    result = await detector.detect("Хочу заблокировать карту")
    assert result.intent == "block_card"
    assert result.requires_human is True


@pytest.mark.asyncio
async def test_tariffs_is_consultative() -> None:
    detector = IntentDetector(MockLLM())
    result = await detector.detect("Какие у вас тарифы по вкладам?")
    assert result.intent == "tariffs"
    assert result.requires_human is False


@pytest.mark.asyncio
async def test_human_keyword_escalates() -> None:
    detector = IntentDetector(MockLLM())
    result = await detector.detect("Соедините меня с оператором")
    assert result.requires_human is True


@pytest.mark.asyncio
async def test_llm_other_with_requires_human_does_not_escalate() -> None:
    detector = IntentDetector(
        _StubLLM('{"intent":"other","requires_human":true,"confidence":0.9}')
    )
    result = await detector.detect("Расскажи про компанию")
    assert result.intent == "other"
    assert result.requires_human is False


@pytest.mark.asyncio
async def test_llm_operational_intent_can_escalate() -> None:
    detector = IntentDetector(
        _StubLLM('{"intent":"block_card","requires_human":false,"confidence":0.92}')
    )
    result = await detector.detect("Нужно срочно заблокировать карту")
    assert result.intent == "block_card"
    assert result.requires_human is True
