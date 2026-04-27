"""REST: просмотр записанных диалогов и транскрипта по ролям."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.backend.models import ConversationOut, ConversationTurnOut, get_session
from apps.backend.models.entities import Conversation, ConversationTurn

router = APIRouter()


@router.get("/{conversation_id}", response_model=ConversationOut)
async def get_conversation(
    conversation_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> ConversationOut:
    conv = await session.get(Conversation, conversation_id)
    if conv is None:
        raise HTTPException(404, "Conversation not found")
    r = await session.execute(
        select(ConversationTurn)
        .where(ConversationTurn.conversation_id == conversation_id)
        .order_by(ConversationTurn.seq.asc())
    )
    turns_orm = list(r.scalars().all())
    turns = [
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
    return ConversationOut(
        id=conv.id,
        channel=conv.channel,
        locale=conv.locale,
        status=conv.status,  # type: ignore[arg-type]
        summary=conv.summary,
        escalated_reason=conv.escalated_reason,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        ended_at=conv.ended_at,
        case_state=dict((conv.meta or {}).get("case_state") or {}),
        turns=turns,
    )
