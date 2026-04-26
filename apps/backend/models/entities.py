"""ORM entities — only the metadata that has to live in PostgreSQL.

Vector data lives in Qdrant; here we only store source documents,
prompt versions and voice profiles managed via the admin panel.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, SmallInteger, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from apps.backend.models.db import Base


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class BotRuntimeSettings(Base, TimestampMixin):
    """Единственная строка id=1: параметры поведения бота (админка)."""

    __tablename__ = "bot_runtime_settings"

    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True, default=1)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")


class Prompt(Base, TimestampMixin):
    __tablename__ = "prompts"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    locale: Mapped[str] = mapped_column(String(8), nullable=False, default="ru")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)


class Voice(Base, TimestampMixin):
    __tablename__ = "voices"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="elevenlabs")
    provider_voice_id: Mapped[str] = mapped_column(String(128), nullable=False)
    locale: Mapped[str] = mapped_column(String(8), nullable=False, default="ru")
    is_default: Mapped[bool] = mapped_column(default=False, nullable=False)
    # Параметры синтеза (ElevenLabs voice_settings, OpenAI speed, и т.д.) — правит админка.
    tts_params: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)


class Document(Base, TimestampMixin):
    """Source document for RAG; chunks live in Qdrant, payload here."""

    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, default="upload")
    locale: Mapped[str] = mapped_column(String(8), nullable=False, default="ru")
    # Which Qdrant collection this document is indexed into (voice bot vs agent assist).
    knowledge_pool: Mapped[str] = mapped_column(String(32), nullable=False, default="voice")
    chunk_count: Mapped[int] = mapped_column(default=0, nullable=False)
    meta: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)


class AnalyticsCriteriaSet(Base, TimestampMixin):
    """Named set of scoring criteria for speech analytics."""

    __tablename__ = "analytics_criteria_sets"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    locale: Mapped[str] = mapped_column(String(8), nullable=False, default="ru")
    is_default: Mapped[bool] = mapped_column(default=False, nullable=False)


class Conversation(Base, TimestampMixin):
    """Голосовой / текстовый диалог с клиентом: транскрипт по ролям + эскалация."""

    __tablename__ = "conversations"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    channel: Mapped[str] = mapped_column(String(32), nullable=False, default="voice_ws")
    locale: Mapped[str] = mapped_column(String(8), nullable=False, default="ru")
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="active"
    )  # active | completed | escalated
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    escalated_reason: Mapped[str | None] = mapped_column(String(256), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    meta: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    turns: Mapped[list["ConversationTurn"]] = relationship(
        "ConversationTurn",
        back_populates="conversation",
        cascade="all, delete-orphan",
    )


class ConversationTurn(Base, TimestampMixin):
    __tablename__ = "conversation_turns"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # customer | assistant | system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    seq: Mapped[int] = mapped_column(nullable=False)
    extra: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="turns")


class AnalyticsCriterion(Base, TimestampMixin):
    """Single criterion with weight and max score (weighted rubric)."""

    __tablename__ = "analytics_criteria"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    set_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("analytics_criteria_sets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    code: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    weight: Mapped[float] = mapped_column(default=1.0, nullable=False)
    max_score: Mapped[float] = mapped_column(default=10.0, nullable=False)
    rubric: Mapped[str] = mapped_column(Text, nullable=False, default="")
