"""Pydantic DTOs used by API + service layers."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ---------- Prompts ----------
class PromptIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    locale: Literal["ru", "uz"] = "ru"
    content: str
    is_active: bool = True


class PromptOut(PromptIn):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    created_at: datetime
    updated_at: datetime


# ---------- Voices ----------
class VoiceIn(BaseModel):
    name: str
    provider: Literal["elevenlabs", "openai", "mock"] = "elevenlabs"
    provider_voice_id: str
    locale: Literal["ru", "uz"] = "ru"
    is_default: bool = False
    tts_params: dict = Field(default_factory=dict)


class VoiceOut(VoiceIn):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    created_at: datetime


class VoiceTtsPatch(BaseModel):
    """Обновление только JSON-параметров синтеза у профиля голоса."""

    tts_params: dict


# ---------- Documents / RAG ----------
class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    title: str
    locale: str
    knowledge_pool: Literal["voice", "agent_assist"] = "voice"
    chunk_count: int
    created_at: datetime


class RAGChunk(BaseModel):
    document_id: UUID
    chunk_id: str
    text: str
    score: float
    title: Optional[str] = None


# ---------- Chat / Playground ----------
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class IntentResult(BaseModel):
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    requires_human: bool = False


class PlaygroundRequest(BaseModel):
    text: str
    locale: Literal["ru", "uz"] = "ru"
    history: list[ChatMessage] = Field(default_factory=list)
    prompt_name: Optional[str] = None


class PlaygroundResponse(BaseModel):
    answer: str
    intent: IntentResult
    sources: list[RAGChunk] = Field(default_factory=list)
    latency_ms: dict[str, float] = Field(default_factory=dict)


# ---------- Speech analytics ----------
class AnalyticsCriterionIn(BaseModel):
    code: str = Field(min_length=1, max_length=64)
    title: str
    description: str = ""
    weight: float = Field(ge=0.0, le=100.0, default=1.0)
    max_score: float = Field(ge=1.0, le=100.0, default=10.0)
    rubric: str = ""


class AnalyticsCriterionOut(AnalyticsCriterionIn):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    set_id: UUID
    created_at: datetime | None = None


class AnalyticsCriteriaSetIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    locale: Literal["ru", "uz"] = "ru"
    is_default: bool = False


class AnalyticsCriteriaSetOut(AnalyticsCriteriaSetIn):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    created_at: datetime
    updated_at: datetime


class DiarizedTurn(BaseModel):
    speaker: Literal["customer", "agent", "unknown"]
    text: str
    start_sec: float | None = None
    end_sec: float | None = None


class CriterionScoreOut(BaseModel):
    code: str
    title: str
    score: float
    max_score: float
    weight: float
    weighted: float
    comment: str = ""


class SpeechAnalysisResult(BaseModel):
    transcript: str
    diarization: list[DiarizedTurn]
    criterion_scores: list[CriterionScoreOut]
    total_weighted: float
    notes: str = ""


# ---------- Voice conversations (transcript + escalation) ----------
class ConversationTurnOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    role: Literal["customer", "assistant", "system"]
    content: str
    seq: int
    extra: dict = Field(default_factory=dict)
    created_at: datetime


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    channel: str
    locale: str
    status: Literal["active", "completed", "escalated"]
    summary: str | None = None
    escalated_reason: str | None = None
    created_at: datetime
    updated_at: datetime
    ended_at: datetime | None = None
    case_state: dict = Field(default_factory=dict)
    turns: list[ConversationTurnOut] = Field(default_factory=list)


class CaseContext(BaseModel):
    intent: str = "other"
    stage: Literal["new", "collecting", "diagnosing", "resolved", "handoff"] = "new"
    intent_confirmed: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    required_slots: list[str] = Field(default_factory=list)
    collected_slots: dict = Field(default_factory=dict)
    missing_slots: list[str] = Field(default_factory=list)
    attempted_steps: list[str] = Field(default_factory=list)
    clarification_attempts: int = 0
    resolution_status: Literal["unknown", "in_progress", "resolved", "handoff"] = "unknown"
    need_handoff: bool = False
    handoff_reason: str | None = None


class EscalationPacket(BaseModel):
    """Пакет для CRM / софтфона при передаче на оператора."""

    conversation_id: UUID
    locale: str
    reason: str
    summary: str
    case_context: CaseContext = Field(default_factory=CaseContext)
    turns: list[ConversationTurnOut]
