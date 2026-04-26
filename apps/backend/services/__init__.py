"""External-AI provider clients implementing Strategy interfaces.

Concrete providers (Deepgram / OpenAI / ElevenLabs) MUST be obtained via
the factory functions below — never imported directly by business code.
This keeps the on-prem migration path open: swap the factory output, no
caller changes.
"""

from apps.backend.services.factory import (
    get_llm,
    get_stt,
    get_tts,
)
from apps.backend.services.interfaces import (
    LLMService,
    STTService,
    TTSService,
)

__all__ = [
    "LLMService",
    "STTService",
    "TTSService",
    "get_llm",
    "get_stt",
    "get_tts",
]
