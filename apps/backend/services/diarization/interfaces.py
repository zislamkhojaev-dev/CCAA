"""Strategy interface: swap LLM / Deepgram / pyannote without touching analytics HTTP."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from apps.backend.models.schemas import DiarizedTurn
from apps.backend.services.diarization.types import TranscriptSegment


class DiarizationService(ABC):
    """Map timestamped ASR segments → speaker-labelled turns."""

    @abstractmethod
    async def diarize(
        self,
        segments: Sequence[TranscriptSegment],
        *,
        full_text: str,
        locale: str = "ru",
        audio_bytes: bytes | None = None,
        audio_filename: str = "audio.wav",
        audio_mime_type: str = "application/octet-stream",
    ) -> list[DiarizedTurn]:
        """Return ordered turns with speaker ∈ {customer, agent, unknown}.

        Провайдеры с доступом к сырому аудио (Deepgram batch, pyannote-воркер)
        используют `audio_*`; LLM-диаризация опирается на `segments` / `full_text`.
        """
