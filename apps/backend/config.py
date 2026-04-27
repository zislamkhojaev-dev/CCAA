"""Application settings loaded from environment variables.

Centralised so that providers can be swapped (Strategy pattern) without
touching business logic. All secrets must come from `.env` / Docker secrets;
nothing is hard-coded.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ----- General -----
    app_name: str = "VoiceAgent"
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # ----- Latency budget (ms) -----
    # Hard requirement from FT: total reply <= 1500ms.
    latency_budget_ms: int = 1500

    # ----- Provider selection (Strategy) -----
    stt_provider: Literal["deepgram", "openai", "local_http", "mock"] = "openai"
    llm_provider: Literal["openai", "local_http", "mock"] = "openai"
    tts_provider: Literal["elevenlabs", "openai", "local_http", "mock"] = "openai"

    # ----- Local HTTP providers (Ollama / LM Studio / vLLM / свой ASR-TTS) -----
    local_llm_base_url: str = ""  # например http://host:11434 — пути /v1/chat/completions
    local_llm_api_key: str = ""
    local_llm_chat_model: str = "llama3.2"
    local_llm_embed_model: str = "nomic-embed-text"
    local_stt_url: str = ""  # POST multipart file=*.wav
    local_tts_url: str = ""  # POST JSON {text, voice_id, locale} → binary audio
    pyannote_worker_url: str = ""  # POST file → JSON turns (см. pyannote_diarization.py)

    # ----- Deepgram (STT) -----
    deepgram_api_key: str = ""
    deepgram_model: str = "nova-2-general"
    deepgram_language: str = "ru"  # supports ru; uz fallback handled in service

    # ----- OpenAI (LLM + embeddings + tts + stt) -----
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dim: int = 1536
    openai_tts_model: str = "tts-1"  # or tts-1-hd
    openai_tts_voice: str = "alloy"  # alloy / echo / fable / onyx / nova / shimmer
    openai_tts_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    openai_stt_model: str = "whisper-1"
    openai_stt_flush_ms: int = 1200  # max buffer size before forcing a flush
    openai_stt_silence_ms: int = 700  # silence gap that triggers a flush

    # ----- ElevenLabs (TTS) -----
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "EXAVITQu4vr4xnSDxMaL"
    elevenlabs_model: str = "eleven_turbo_v2_5"

    # ----- PostgreSQL -----
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "voiceagent"
    postgres_user: str = "voiceagent"
    postgres_password: str = "voiceagent"

    # ----- Qdrant (two KB pools: voice bot vs agent assist суфлёр) -----
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    # Back-compat: `qdrant_collection` mirrors voice pool if you only set one env var.
    qdrant_collection: str = "knowledge_base"
    qdrant_collection_voice: str = "knowledge_base"
    qdrant_collection_agent_assist: str = "knowledge_agent_assist"

    # ----- Diarization (speech analytics; same contract for future batch jobs) -----
    diarization_provider: Literal["llm", "deepgram", "pyannote", "mock"] = "llm"

    # ----- Voice VAD (before STT) -----
    voice_vad_backend: Literal["silero_onnx", "webrtc"] = "silero_onnx"
    silero_vad_onnx_url: str = (
        "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/"
        "src/silero_vad/data/silero_vad.onnx"
    )
    silero_vad_cache_dir: str = ""

    # ----- Locales -----
    supported_locales: list[str] = Field(default_factory=lambda: ["ru", "uz"])
    default_locale: str = "ru"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
