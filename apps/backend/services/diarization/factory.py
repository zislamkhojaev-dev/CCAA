"""Diarization provider factory."""

from __future__ import annotations

from functools import lru_cache

from apps.backend.config import get_settings
from apps.backend.services.diarization.interfaces import DiarizationService


@lru_cache(maxsize=1)
def get_diarization() -> DiarizationService:
    s = get_settings()
    if s.diarization_provider == "deepgram":
        from apps.backend.services.diarization.deepgram_diarization import (
            DeepgramDiarizationService,
        )

        return DeepgramDiarizationService()
    if s.diarization_provider == "pyannote":
        from apps.backend.services.diarization.pyannote_diarization import (
            PyannoteDiarizationService,
        )

        return PyannoteDiarizationService()
    if s.diarization_provider == "mock":
        from apps.backend.services.diarization.mock_diarization import MockDiarizationService

        return MockDiarizationService()
    from apps.backend.services.diarization.llm_diarization import LLMDiarizationService

    return LLMDiarizationService()
