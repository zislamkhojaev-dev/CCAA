"""Smoke tests for the keyword-based intent router (no network required)."""

from __future__ import annotations

import pytest

from apps.backend.core.intent import IntentDetector
from apps.backend.services.mocks import MockLLM


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
