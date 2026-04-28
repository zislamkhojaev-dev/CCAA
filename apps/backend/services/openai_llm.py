"""OpenAI chat-completion + embeddings client (async)."""

from __future__ import annotations

from typing import AsyncIterator, Sequence

from openai import AsyncOpenAI

from apps.backend.config import get_settings
from apps.backend.models.schemas import ChatMessage
from apps.backend.services.interfaces import LLMService
from apps.backend.utils.logging import get_logger
from apps.backend.utils.resilience import SimpleCircuitBreaker, retry_async

log = get_logger(__name__)


class OpenAILLM(LLMService):
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            log.warning("openai_api_key_missing")
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model
        self._embed_model = settings.openai_embedding_model
        self._breaker = SimpleCircuitBreaker(fail_threshold=3, open_sec=20.0)

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
        if not await self._breaker.before_call():
            log.warning("openai_llm_circuit_open", operation="complete")
            return ""
        try:
            resp = await retry_async(
                lambda: self._client.chat.completions.create(
                    model=self._model,
                    messages=self._to_openai(messages),
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                attempts=2,
            )
            await self._breaker.on_success()
            return (resp.choices[0].message.content or "").strip()
        except Exception:  # noqa: BLE001
            await self._breaker.on_failure()
            raise

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
        if not await self._breaker.before_call():
            log.warning("openai_llm_circuit_open", operation="embed")
            return []
        try:
            resp = await retry_async(
                lambda: self._client.embeddings.create(
                    model=self._embed_model, input=list(texts)
                ),
                attempts=2,
            )
            await self._breaker.on_success()
            return [d.embedding for d in resp.data]
        except Exception:  # noqa: BLE001
            await self._breaker.on_failure()
            raise

    async def complete_with_tools(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> tuple[str, list[dict]]:
        if not await self._breaker.before_call():
            log.warning("openai_llm_circuit_open", operation="complete_with_tools")
            return "", []
        try:
            resp = await retry_async(
                lambda: self._client.chat.completions.create(
                    model=self._model,
                    messages=self._to_openai(messages),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    tool_choice="auto",
                ),
                attempts=2,
            )
            await self._breaker.on_success()
        except Exception:  # noqa: BLE001
            await self._breaker.on_failure()
            raise
        msg = resp.choices[0].message
        text = (msg.content or "").strip()
        out_calls: list[dict] = []
        for tc in (msg.tool_calls or []):
            if tc.type != "function" or not tc.function:
                continue
            raw_args = tc.function.arguments or "{}"
            try:
                import json

                args = json.loads(raw_args)
            except Exception:  # noqa: BLE001
                args = {}
            if not isinstance(args, dict):
                args = {}
            out_calls.append({"name": tc.function.name, "arguments": args})
        return text, out_calls
