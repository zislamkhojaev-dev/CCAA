"""In-process fakes used in tests and when API keys are absent (dev mode)."""

from __future__ import annotations

import asyncio
import hashlib
from typing import AsyncIterator, Sequence

from apps.backend.models.schemas import ChatMessage
from apps.backend.services.interfaces import (
    LLMService,
    STTEvent,
    STTService,
    TTSService,
)


class MockSTT(STTService):
    async def stream_transcribe(
        self,
        audio_chunks: AsyncIterator[bytes],
        *,
        locale: str = "ru",
        sample_rate: int = 16_000,
    ) -> AsyncIterator[STTEvent]:
        total = 0
        async for chunk in audio_chunks:
            total += len(chunk)
        yield STTEvent(text=f"[mock transcript {total}b]", is_final=True, confidence=1.0)

    async def aclose(self) -> None:
        return None


class MockLLM(LLMService):
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        last = messages[-1].content if messages else ""
        return f"[mock-llm] {last[:120]}"

    async def stream_complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> AsyncIterator[str]:
        for token in (await self.complete(messages)).split():
            await asyncio.sleep(0.01)
            yield token + " "

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        from apps.backend.config import get_settings

        dim = get_settings().openai_embedding_dim
        result: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [(b - 127.5) / 127.5 for b in digest]
            while len(vec) < dim:
                vec.extend(vec[: dim - len(vec)])
            result.append(vec[:dim])
        return result


class MockTTS(TTSService):
    async def stream_synthesize(
        self,
        text_chunks: AsyncIterator[str],
        *,
        voice_id: str,
        locale: str = "ru",
        voice_tts_params: dict | None = None,
    ) -> AsyncIterator[bytes]:
        async for delta in text_chunks:
            yield delta.encode("utf-8")

    async def aclose(self) -> None:
        return None
