"""CRUD over TTS voice profiles."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from apps.backend.models import Voice, VoiceIn, VoiceOut, VoiceTtsPatch, get_session

router = APIRouter()


@router.get("", response_model=list[VoiceOut])
async def list_voices(session: AsyncSession = Depends(get_session)) -> list[Voice]:
    result = await session.execute(select(Voice).order_by(Voice.created_at.desc()))
    return list(result.scalars().all())


@router.post("", response_model=VoiceOut, status_code=status.HTTP_201_CREATED)
async def create_voice(
    body: VoiceIn, session: AsyncSession = Depends(get_session)
) -> Voice:
    if body.is_default:
        await session.execute(update(Voice).values(is_default=False))
    voice = Voice(**body.model_dump())
    session.add(voice)
    await session.flush()
    return voice


@router.patch("/{voice_id}", response_model=VoiceOut)
async def patch_voice_tts(
    voice_id: UUID,
    body: VoiceTtsPatch,
    session: AsyncSession = Depends(get_session),
) -> Voice:
    voice = await session.get(Voice, voice_id)
    if voice is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Voice not found")
    voice.tts_params = body.tts_params
    await session.flush()
    return voice


@router.delete("/{voice_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_voice(
    voice_id: UUID, session: AsyncSession = Depends(get_session)
) -> None:
    voice = await session.get(Voice, voice_id)
    if voice is None:
        raise HTTPException(404, "Voice not found")
    await session.delete(voice)
