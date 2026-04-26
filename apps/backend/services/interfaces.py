"""Abstract Strategy interfaces for STT / LLM / TTS providers.

The Voice Engine and Orchestrator only ever depend on these protocols,
which lets us swap to local models (e.g. faster-whisper, vLLM, XTTS)
without touching call-flow code. Required by FT for on-prem migration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Sequence

from apps.backend.models.schemas import ChatMessage


class STTService(ABC):
    """Streaming Speech-to-Text."""

    @abstractmethod
    async def stream_transcribe(
        self,
        audio_chunks: AsyncIterator[bytes],
        *,
        locale: str = "ru",
        sample_rate: int = 16_000,
    ) -> AsyncIterator["STTEvent"]:
        """Consume PCM/Opus chunks and yield interim + final transcripts."""

    @abstractmethod
    async def aclose(self) -> None: ...


class LLMService(ABC):
    """Chat-completion + embedding."""

    @abstractmethod
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str: ...

    @abstractmethod
    async def stream_complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> AsyncIterator[str]: ...

    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class TTSService(ABC):
    """Streaming Text-to-Speech."""

    @abstractmethod
    async def stream_synthesize(
        self,
        text_chunks: AsyncIterator[str],
        *,
        voice_id: str,
        locale: str = "ru",
        voice_tts_params: dict | None = None,
    ) -> AsyncIterator[bytes]:
        """Yield audio chunks (mp3/opus) as soon as they're ready."""

    @abstractmethod
    async def aclose(self) -> None: ...


# ---------- Shared event types ----------
from dataclasses import dataclass


@dataclass(slots=True)
class STTEvent:
    text: str
    is_final: bool
    confidence: float = 0.0
