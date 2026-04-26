"""Provider factory — single place to swap STT/LLM/TTS implementations."""

from __future__ import annotations

from functools import lru_cache

from apps.backend.config import get_settings
from apps.backend.services.interfaces import LLMService, STTService, TTSService


@lru_cache(maxsize=1)
def get_stt() -> STTService:
    settings = get_settings()
    if settings.stt_provider == "deepgram":
        from apps.backend.services.deepgram_stt import DeepgramSTT

        return DeepgramSTT()
    if settings.stt_provider == "openai":
        from apps.backend.services.openai_stt import OpenAIWhisperSTT

        return OpenAIWhisperSTT()
    if settings.stt_provider == "local_http":
        from apps.backend.services.local_http_stt import LocalHttpSTT

        return LocalHttpSTT()
    from apps.backend.services.mocks import MockSTT

    return MockSTT()


@lru_cache(maxsize=1)
def get_llm() -> LLMService:
    settings = get_settings()
    if settings.llm_provider == "openai":
        from apps.backend.services.openai_llm import OpenAILLM

        return OpenAILLM()
    if settings.llm_provider == "local_http":
        from apps.backend.services.local_http_llm import LocalHttpLLM

        return LocalHttpLLM()
    from apps.backend.services.mocks import MockLLM

    return MockLLM()


@lru_cache(maxsize=1)
def get_tts() -> TTSService:
    settings = get_settings()
    if settings.tts_provider == "elevenlabs":
        from apps.backend.services.elevenlabs_tts import ElevenLabsTTS

        return ElevenLabsTTS()
    if settings.tts_provider == "openai":
        from apps.backend.services.openai_tts import OpenAITTS

        return OpenAITTS()
    if settings.tts_provider == "local_http":
        from apps.backend.services.local_http_tts import LocalHttpTTS

        return LocalHttpTTS()
    from apps.backend.services.mocks import MockTTS

    return MockTTS()
