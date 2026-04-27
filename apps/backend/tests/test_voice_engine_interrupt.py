from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from apps.backend.core.voice_engine import VoiceEngine
from apps.backend.models.schemas import IntentResult, RAGChunk


class _FakeOrchestrator:
    async def preview_route(self, text: str, *, locale: str = "ru"):
        return (
            IntentResult(intent="other", confidence=0.9, requires_human=False),
            [RAGChunk(document_id=uuid4(), chunk_id="c1", text="ctx", score=0.9, title="KB")],
        )

    async def resolve_after_route(
        self,
        text: str,
        intent: IntentResult,
        sources: list[RAGChunk],
        *,
        locale: str = "ru",
        history=None,
    ) -> tuple[str, str | None]:
        return "Первая фраза. Вторая фраза.", None


class _FakeTTS:
    def __init__(self) -> None:
        self.calls = 0

    async def stream_synthesize(
        self,
        text_chunks,
        *,
        voice_id: str,
        locale: str = "ru",
        voice_tts_params: dict | None = None,
    ):
        self.calls += 1
        call_no = self.calls
        async for text in text_chunks:
            if call_no == 1:
                yield text.encode("utf-8")
                return
            await asyncio.sleep(0.5)
            yield text.encode("utf-8")


class _FakeRecorder:
    def __init__(self) -> None:
        self.turns: list[tuple[str, str, dict | None]] = []
        self.escalated_reason: str | None = None
        self.case_state: dict = {}

    async def add_turn(self, role: str, content: str, *, extra: dict | None = None) -> None:
        self.turns.append((role, content, extra))

    async def mark_escalated(self, reason: str | None) -> None:
        self.escalated_reason = reason

    async def build_escalation_packet(self, *, locale: str, reason: str) -> dict:
        return {
            "conversation_id": str(uuid4()),
            "locale": locale,
            "reason": reason,
            "summary": "summary",
            "case_context": self.case_state,
            "turns": [
                {"role": role, "content": content, "extra": extra or {}}
                for role, content, extra in self.turns
            ],
        }

    async def get_case_state(self) -> dict:
        return dict(self.case_state)

    async def upsert_case_state(self, case_state: dict) -> None:
        self.case_state = dict(case_state or {})


@pytest.mark.asyncio
async def test_interrupt_saves_only_spoken_partial(monkeypatch) -> None:
    monkeypatch.setattr("apps.backend.core.voice_engine.get_stt", lambda: object())
    monkeypatch.setattr(
        "apps.backend.core.voice_engine._split_tts_units",
        lambda _text: ["Первая фраза.", "Вторая фраза."],
    )
    fake_tts = _FakeTTS()
    monkeypatch.setattr("apps.backend.core.voice_engine.get_tts", lambda: fake_tts)
    recorder = _FakeRecorder()
    engine = VoiceEngine(_FakeOrchestrator(), conversation_recorder=recorder)  # type: ignore[arg-type]

    out_q: asyncio.Queue[bytes | None] = asyncio.Queue()
    turn_task = asyncio.create_task(
        engine._handle_turn(  # noqa: SLF001
            text="Расскажите",
            locale="ru",
            voice_id="v1",
            audio_out=out_q,
        )
    )
    async with engine._turn_lock:  # noqa: SLF001
        engine._active_turn = turn_task  # noqa: SLF001

    first_packet = await asyncio.wait_for(out_q.get(), timeout=1.0)
    assert first_packet is not None

    await engine.interrupt()
    assert turn_task.cancelled() is True

    assistant_turns = [t for t in recorder.turns if t[0] == "assistant"]
    assert assistant_turns, "partial assistant turn must be recorded on interrupt"
    spoken_partial = assistant_turns[-1][1]
    assert spoken_partial == "Первая фраза."
    assert assistant_turns[-1][2] == {"interrupted": True, "partial": True}


class _EscalateOrchestrator:
    async def preview_route(self, text: str, *, locale: str = "ru"):
        return (
            IntentResult(intent="other", confidence=0.9, requires_human=False),
            [RAGChunk(document_id=uuid4(), chunk_id="c1", text="ctx", score=0.9, title="KB")],
        )

    async def resolve_after_route(
        self,
        text: str,
        intent: IntentResult,
        sources: list[RAGChunk],
        *,
        locale: str = "ru",
        history=None,
    ) -> tuple[str, str | None]:
        return "Соединяю с оператором.", "Клиент просит возврат"


@pytest.mark.asyncio
async def test_transfer_tool_emits_escalation_packet(monkeypatch) -> None:
    monkeypatch.setattr("apps.backend.core.voice_engine.get_stt", lambda: object())
    monkeypatch.setattr("apps.backend.core.voice_engine.get_tts", lambda: _FakeTTS())
    recorder = _FakeRecorder()
    packets: list[dict] = []

    async def on_escalation(pkg: dict) -> None:
        packets.append(pkg)

    engine = VoiceEngine(
        _EscalateOrchestrator(),  # type: ignore[arg-type]
        conversation_recorder=recorder,
        on_escalation=on_escalation,
    )
    out_q: asyncio.Queue[bytes | None] = asyncio.Queue()

    await engine._handle_turn(  # noqa: SLF001
        text="Хочу вернуть деньги",
        locale="ru",
        voice_id="v1",
        audio_out=out_q,
    )

    assert recorder.escalated_reason == "Клиент просит возврат"
    assert len(packets) == 1
    assert packets[0]["reason"] == "Клиент просит возврат"
    assert any(role == "assistant" and content == "Соединяю с оператором." for role, content, _ in recorder.turns)


class _GreetingOrchestrator:
    async def preview_route(self, text: str, *, locale: str = "ru"):
        return IntentResult(intent="greeting", confidence=0.95, requires_human=False), []

    async def resolve_after_route(
        self,
        text: str,
        intent: IntentResult,
        sources: list[RAGChunk],
        *,
        locale: str = "ru",
        history=None,
    ) -> tuple[str, str | None]:
        return "", None


@pytest.mark.asyncio
async def test_greeting_does_not_use_kb_fallback(monkeypatch) -> None:
    monkeypatch.setattr("apps.backend.core.voice_engine.get_stt", lambda: object())
    monkeypatch.setattr("apps.backend.core.voice_engine.get_tts", lambda: _FakeTTS())
    recorder = _FakeRecorder()
    engine = VoiceEngine(_GreetingOrchestrator(), conversation_recorder=recorder)  # type: ignore[arg-type]
    out_q: asyncio.Queue[bytes | dict | None] = asyncio.Queue()

    await engine._handle_turn(  # noqa: SLF001
        text="Привет!",
        locale="ru",
        voice_id="v1",
        audio_out=out_q,
    )

    assistant_turns = [t for t in recorder.turns if t[0] == "assistant"]
    assert assistant_turns
    assert "не нахожу точных данных в базе знаний" not in assistant_turns[-1][1].lower()
    assert assistant_turns[-1][2] == {"smalltalk": True}


class _LowConfidenceOrchestrator:
    async def preview_route(self, text: str, *, locale: str = "ru"):
        return (
            IntentResult(intent="products", confidence=0.3, requires_human=False),
            [RAGChunk(document_id=uuid4(), chunk_id="c1", text="ctx", score=0.9, title="KB")],
        )

    async def resolve_after_route(
        self,
        text: str,
        intent: IntentResult,
        sources: list[RAGChunk],
        *,
        locale: str = "ru",
        history=None,
    ) -> tuple[str, str | None]:
        return "Ответ", None


@pytest.mark.asyncio
async def test_low_confidence_turn_requests_intent_confirmation(monkeypatch) -> None:
    monkeypatch.setattr("apps.backend.core.voice_engine.get_stt", lambda: object())
    monkeypatch.setattr("apps.backend.core.voice_engine.get_tts", lambda: _FakeTTS())
    recorder = _FakeRecorder()
    engine = VoiceEngine(_LowConfidenceOrchestrator(), conversation_recorder=recorder)  # type: ignore[arg-type]
    out_q: asyncio.Queue[bytes | dict | None] = asyncio.Queue()

    await engine._handle_turn(  # noqa: SLF001
        text="Мне нужна помощь",
        locale="ru",
        voice_id="v1",
        audio_out=out_q,
    )

    assistant_turns = [t for t in recorder.turns if t[0] == "assistant"]
    assert assistant_turns
    assert "правильно понимаю" in assistant_turns[-1][1].lower()
    assert assistant_turns[-1][2] == {"intent_confirmation": True, "intent": "products"}


@pytest.mark.asyncio
async def test_case_templates_from_runtime_are_used(monkeypatch) -> None:
    monkeypatch.setattr("apps.backend.core.voice_engine.get_stt", lambda: object())
    monkeypatch.setattr("apps.backend.core.voice_engine.get_tts", lambda: _FakeTTS())
    monkeypatch.setattr(
        "apps.backend.core.case_manager.get_bot_runtime_payload_sync",
        lambda: {
            "case_confirm_intent_min_confidence": 0.8,
            "case_intent_confirmation_templates": {
                "default": {"ru": "Подтвердите intent={intent}", "uz": "intent={intent} tasdiqlang"}
            },
            "case_slot_question_templates": {
                "product_name": {"ru": "Назовите продукт", "uz": "Mahsulot nomini ayting"},
                "default": {"ru": "Уточните деталь", "uz": "Aniqlashtiring"},
            },
            "case_fallback_handoff_limit": 2,
            "case_low_confidence_threshold": 0.45,
            "case_clarification_max_attempts": 2,
        },
    )
    recorder = _FakeRecorder()
    engine = VoiceEngine(_LowConfidenceOrchestrator(), conversation_recorder=recorder)  # type: ignore[arg-type]
    out_q: asyncio.Queue[bytes | dict | None] = asyncio.Queue()

    await engine._handle_turn(  # noqa: SLF001
        text="Мне нужна помощь",
        locale="ru",
        voice_id="v1",
        audio_out=out_q,
    )
    assistant_turns = [t for t in recorder.turns if t[0] == "assistant"]
    assert assistant_turns[-1][1] == "Подтвердите intent=products"


class _TransferIntentOrchestrator:
    async def preview_route(self, text: str, *, locale: str = "ru"):
        return (
            IntentResult(intent="transfer_money", confidence=0.9, requires_human=True),
            [RAGChunk(document_id=uuid4(), chunk_id="c1", text="ctx", score=0.9, title="KB")],
        )

    async def resolve_after_route(
        self,
        text: str,
        intent: IntentResult,
        sources: list[RAGChunk],
        *,
        locale: str = "ru",
        history=None,
    ) -> tuple[str, str | None]:
        return "Соединяю", "intent:transfer_money"


@pytest.mark.asyncio
async def test_transfer_intent_collects_slot_before_handoff(monkeypatch) -> None:
    monkeypatch.setattr("apps.backend.core.voice_engine.get_stt", lambda: object())
    monkeypatch.setattr("apps.backend.core.voice_engine.get_tts", lambda: _FakeTTS())
    recorder = _FakeRecorder()
    engine = VoiceEngine(_TransferIntentOrchestrator(), conversation_recorder=recorder)  # type: ignore[arg-type]
    out_q: asyncio.Queue[bytes | dict | None] = asyncio.Queue()

    await engine._handle_turn(  # noqa: SLF001
        text="Ошибка при переводе",
        locale="ru",
        voice_id="v1",
        audio_out=out_q,
    )
    assistant_turns = [t for t in recorder.turns if t[0] == "assistant"]
    assert assistant_turns
    assert "с какого счета" in assistant_turns[-1][1].lower() or "уточ" in assistant_turns[-1][1].lower()
    assert not any((t[2] or {}).get("escalation") for t in assistant_turns)
