"""Deterministic stub for tests."""

from __future__ import annotations

from typing import Sequence

from apps.backend.models.schemas import DiarizedTurn
from apps.backend.services.diarization.interfaces import DiarizationService
from apps.backend.services.diarization.types import TranscriptSegment


class MockDiarizationService(DiarizationService):
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
        t = (full_text or "").strip()
        return [DiarizedTurn(speaker="unknown", text=t)] if t else []
