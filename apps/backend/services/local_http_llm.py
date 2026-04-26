"""Локальный / on-prem LLM через OpenAI-совместимый HTTP API (Ollama, LM Studio, vLLM).

Ожидаются эндпоинты:
  POST {base}/v1/chat/completions
  POST {base}/v1/embeddings

Модели задаются в `LOCAL_LLM_CHAT_MODEL` и `LOCAL_LLM_EMBED_MODEL`.
"""

from __future__ import annotations

import json
from typing import AsyncIterator, Sequence

import httpx

from apps.backend.config import get_settings
from apps.backend.models.schemas import ChatMessage
from apps.backend.services.interfaces import LLMService
from apps.backend.utils.logging import get_logger

log = get_logger(__name__)


class LocalHttpLLM(LLMService):
    def __init__(self) -> None:
        s = get_settings()
        base = (s.local_llm_base_url or "").rstrip("/")
        if not base:
            log.warning("local_llm_base_url_missing")
        headers: dict[str, str] = {"content-type": "application/json"}
        if s.local_llm_api_key:
            headers["authorization"] = f"Bearer {s.local_llm_api_key}"
        self._client = httpx.AsyncClient(base_url=base, headers=headers, timeout=120.0)
        self._chat_model = s.local_llm_chat_model
        self._embed_model = s.local_llm_embed_model

    @staticmethod
    def _msgs(messages: Sequence[ChatMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        body = {
            "model": self._chat_model,
            "messages": self._msgs(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = await self._client.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()

    async def stream_complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> AsyncIterator[str]:
        body = {
            "model": self._chat_model,
            "messages": self._msgs(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        async with self._client.stream("POST", "/v1/chat/completions", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                    delta = obj["choices"][0].get("delta", {}).get("content") or ""
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        body = {"model": self._embed_model, "input": list(texts)}
        r = await self._client.post("/v1/embeddings", json=body)
        r.raise_for_status()
        data = r.json()
        return [d["embedding"] for d in data["data"]]

    async def aclose(self) -> None:
        await self._client.aclose()
