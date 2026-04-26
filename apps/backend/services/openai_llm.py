"""OpenAI chat-completion + embeddings client (async)."""

from __future__ import annotations

from typing import AsyncIterator, Sequence

from openai import AsyncOpenAI

from apps.backend.config import get_settings
from apps.backend.models.schemas import ChatMessage
from apps.backend.services.interfaces import LLMService
from apps.backend.utils.logging import get_logger

log = get_logger(__name__)


class OpenAILLM(LLMService):
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            log.warning("openai_api_key_missing")
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model
        self._embed_model = settings.openai_embedding_model

    @staticmethod
    def _to_openai(messages: Sequence[ChatMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=self._to_openai(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    async def stream_complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=self._to_openai(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content or ""
            except (AttributeError, IndexError):
                continue
            if delta:
                yield delta

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self._client.embeddings.create(
            model=self._embed_model, input=list(texts)
        )
        return [d.embedding for d in resp.data]
