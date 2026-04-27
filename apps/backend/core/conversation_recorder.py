"""Persist voice dialog: transcript by role, summary, escalation package."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from sqlalchemy import select, update

from apps.backend.models.db import session_scope
from apps.backend.models.entities import Conversation, ConversationTurn
from apps.backend.models.schemas import CaseContext, ChatMessage, ConversationTurnOut, EscalationPacket
from apps.backend.services import get_llm
from apps.backend.utils.logging import get_logger

log = get_logger(__name__)

Role = Literal["customer", "assistant", "system"]


class ConversationRecorder:
    """Per WebSocket session. All DB writes go through `session_scope`."""

    def __init__(self, conversation_id: UUID) -> None:
        self.conversation_id = conversation_id
        self._seq_lock = asyncio.Lock()
        self._llm = get_llm()

    async def _next_seq(self) -> int:
        async with self._seq_lock:
            async with session_scope() as s:
                r = await s.execute(
                    select(ConversationTurn.seq)
                    .where(ConversationTurn.conversation_id == self.conversation_id)
                    .order_by(ConversationTurn.seq.desc())
                    .limit(1)
                )
                last = r.scalar_one_or_none()
                return (last or 0) + 1

    async def add_turn(self, role: Role, content: str, *, extra: dict | None = None) -> None:
        content = (content or "").strip()
        if not content:
            return
        seq = await self._next_seq()
        async with session_scope() as s:
            s.add(
                ConversationTurn(
                    conversation_id=self.conversation_id,
                    role=role,
                    content=content,
                    seq=seq,
                    extra=extra or {},
                )
            )
        log.info(
            "conversation_turn",
            conversation_id=str(self.conversation_id),
            role=role,
            seq=seq,
        )

    async def mark_escalated(self, reason: str | None) -> None:
        async with session_scope() as s:
            await s.execute(
                update(Conversation)
                .where(Conversation.id == self.conversation_id)
                .values(
                    status="escalated",
                    escalated_reason=(reason[:256] if reason else None),
                    updated_at=datetime.now(timezone.utc),
                )
            )

    async def mark_completed(self) -> None:
        now = datetime.now(timezone.utc)
        async with session_scope() as s:
            conv = await s.get(Conversation, self.conversation_id)
            if conv is None:
                return
            if conv.status == "active":
                conv.status = "completed"
                conv.ended_at = now
                conv.updated_at = now

    async def get_case_state(self) -> dict:
        async with session_scope() as s:
            conv = await s.get(Conversation, self.conversation_id)
            if conv is None:
                return {}
            meta = dict(conv.meta or {})
            data = meta.get("case_state")
            return data if isinstance(data, dict) else {}

    async def upsert_case_state(self, case_state: dict) -> None:
        async with session_scope() as s:
            conv = await s.get(Conversation, self.conversation_id)
            if conv is None:
                return
            meta = dict(conv.meta or {})
            meta["case_state"] = dict(case_state or {})
            conv.meta = meta
            conv.updated_at = datetime.now(timezone.utc)

    async def load_turns(self) -> list[ConversationTurn]:
        async with session_scope() as s:
            r = await s.execute(
                select(ConversationTurn)
                .where(ConversationTurn.conversation_id == self.conversation_id)
                .order_by(ConversationTurn.seq.asc())
            )
            return list(r.scalars().all())

    async def summarize(self, *, locale: str) -> str:
        turns = await self.load_turns()
        if not turns:
            return ""
        lines = "\n".join(f"{t.role}: {t.content}" for t in turns)
        sys = (
            "Сожми диалог в 3–6 коротких предложений для оператора: тема, ключевые факты, "
            "что уже сказал бот, что хочет клиент. Без markdown."
            if locale != "uz"
            else "Dialogni operator uchun 3–6 qisqa jumlada qisqartiring."
        )
        msg = await self._llm.complete(
            [
                ChatMessage(role="system", content=sys),
                ChatMessage(role="user", content=lines),
            ],
            temperature=0.2,
            max_tokens=400,
        )
        summary = (msg or "").strip()
        async with session_scope() as s:
            await s.execute(
                update(Conversation)
                .where(Conversation.id == self.conversation_id)
                .values(summary=summary, updated_at=datetime.now(timezone.utc))
            )
        return summary

    async def build_escalation_packet(self, *, locale: str, reason: str) -> dict[str, Any]:
        summary = await self.summarize(locale=locale)
        case_state = await self.get_case_state()
        case_context = CaseContext.model_validate(case_state or {})
        turns_orm = await self.load_turns()
        turns_out = [
            ConversationTurnOut(
                id=t.id,
                role=t.role,  # type: ignore[arg-type]
                content=t.content,
                seq=t.seq,
                extra=t.extra or {},
                created_at=t.created_at,
            )
            for t in turns_orm
        ]
        pkt = EscalationPacket(
            conversation_id=self.conversation_id,
            locale=locale,
            reason=reason,
            summary=summary,
            case_context=case_context,
            turns=turns_out,
        )
        return pkt.model_dump(mode="json")


async def create_conversation(*, channel: str = "voice_ws", locale: str = "ru") -> UUID:
    cid = uuid4()
    async with session_scope() as s:
        s.add(
            Conversation(
                id=cid,
                channel=channel,
                locale=locale,
                status="active",
                meta={},
            )
        )
    log.info("conversation_created", conversation_id=str(cid))
    return cid
