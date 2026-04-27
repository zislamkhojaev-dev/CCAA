from __future__ import annotations

from uuid import uuid4

import pytest

from apps.backend.core.orchestrator import Orchestrator
from apps.backend.models.schemas import ChatMessage, IntentResult, RAGChunk


class _FakeLLM:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)

    async def complete(self, messages, *, temperature: float = 0.2, max_tokens: int = 512) -> str:
        if self._outputs:
            return self._outputs.pop(0)
        return ""

    async def stream_complete(self, messages, *, temperature: float = 0.2, max_tokens: int = 512):
        yield ""

    async def embed(self, texts):
        return [[0.0] * 3 for _ in texts]

    async def complete_with_tools(
        self,
        messages,
        *,
        tools,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> tuple[str, list[dict]]:
        raw = await self.complete(messages, temperature=temperature, max_tokens=max_tokens)
        return raw, []


class _FakeRAG:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(self, query: str, *, locale: str, top_k: int = 5) -> list[RAGChunk]:
        self.queries.append(query)
        return [
            RAGChunk(
                document_id=uuid4(),
                chunk_id="c1",
                text=f"Инструкция по запросу: {query}",
                score=0.9,
                title="KB",
            )
        ]

    @staticmethod
    def format_context(chunks) -> str:
        return "\n".join(c.text for c in chunks)


@pytest.mark.asyncio
async def test_resolve_after_route_transfer_to_human(monkeypatch) -> None:
    monkeypatch.setattr(
        "apps.backend.core.orchestrator.get_llm",
        lambda: _FakeLLM(
            ['{"tool_call":{"name":"transfer_to_human","arguments":{"reason":"Клиент требует возврат"}}}']
        ),
    )
    rag = _FakeRAG()
    orchestrator = Orchestrator(rag=rag)
    intent = IntentResult(intent="other", confidence=0.9, requires_human=False)
    sources = [
        RAGChunk(document_id=uuid4(), chunk_id="x", text="ctx", score=0.8, title="doc"),
    ]

    answer, reason = await orchestrator.resolve_after_route(
        "Верните деньги",
        intent,
        sources,
        locale="ru",
        history=[ChatMessage(role="user", content="Привет")],
    )

    assert "оператора" in answer.lower()
    assert reason == "Клиент требует возврат"


@pytest.mark.asyncio
async def test_resolve_after_route_search_tool_then_answer(monkeypatch) -> None:
    monkeypatch.setattr(
        "apps.backend.core.orchestrator.get_llm",
        lambda: _FakeLLM(
            [
                '{"tool_call":{"name":"search_qdrant","arguments":{"query":"сброс пароля"}}}',
                "Сбросить пароль можно в настройках профиля.",
            ]
        ),
    )
    rag = _FakeRAG()
    orchestrator = Orchestrator(rag=rag)
    intent = IntentResult(intent="other", confidence=0.9, requires_human=False)
    sources = [
        RAGChunk(document_id=uuid4(), chunk_id="x", text="ctx", score=0.8, title="doc"),
    ]

    answer, reason = await orchestrator.resolve_after_route(
        "Как сбросить пароль?",
        intent,
        sources,
        locale="ru",
        history=[],
    )

    assert reason is None
    assert "сбросить пароль" in answer.lower()
    assert rag.queries == ["сброс пароля"]


@pytest.mark.asyncio
async def test_invalid_tool_json_retries_then_answers(monkeypatch) -> None:
    monkeypatch.setattr(
        "apps.backend.core.orchestrator.get_llm",
        lambda: _FakeLLM(
            [
                '{"tool_call":{"name":"search_qdrant","arguments":{"query":"q"}}',
                "Используйте раздел восстановления доступа в профиле.",
            ]
        ),
    )
    rag = _FakeRAG()
    orchestrator = Orchestrator(rag=rag)
    intent = IntentResult(intent="other", confidence=0.9, requires_human=False)
    sources = [RAGChunk(document_id=uuid4(), chunk_id="x", text="ctx", score=0.8, title="doc")]

    answer, reason = await orchestrator.resolve_after_route(
        "Как восстановить доступ?",
        intent,
        sources,
        locale="ru",
        history=[],
    )

    assert reason is None
    assert "восстановления доступа" in answer.lower()
    assert rag.queries == []


@pytest.mark.asyncio
async def test_unknown_tool_exhausts_loop_and_escalates(monkeypatch) -> None:
    monkeypatch.setattr(
        "apps.backend.core.orchestrator.get_llm",
        lambda: _FakeLLM(
            [
                '{"tool_call":{"name":"unknown_tool","arguments":{}}}',
                '{"tool_call":{"name":"unknown_tool","arguments":{}}}',
                '{"tool_call":{"name":"unknown_tool","arguments":{}}}',
            ]
        ),
    )
    rag = _FakeRAG()
    orchestrator = Orchestrator(rag=rag)
    intent = IntentResult(intent="other", confidence=0.9, requires_human=False)
    sources = [RAGChunk(document_id=uuid4(), chunk_id="x", text="ctx", score=0.8, title="doc")]

    answer, reason = await orchestrator.resolve_after_route(
        "Сложный запрос",
        intent,
        sources,
        locale="ru",
        history=[],
    )

    assert "оператора" in answer.lower()
    assert reason == "agent_loop_exhausted"
