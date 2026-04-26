"""CRUD over system prompts shown in the admin panel."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.backend.models import Prompt, PromptIn, PromptOut, get_session

router = APIRouter()


@router.get("", response_model=list[PromptOut])
async def list_prompts(session: AsyncSession = Depends(get_session)) -> list[Prompt]:
    result = await session.execute(select(Prompt).order_by(Prompt.updated_at.desc()))
    return list(result.scalars().all())


@router.post("", response_model=PromptOut, status_code=status.HTTP_201_CREATED)
async def create_prompt(
    body: PromptIn, session: AsyncSession = Depends(get_session)
) -> Prompt:
    prompt = Prompt(**body.model_dump())
    session.add(prompt)
    await session.flush()
    return prompt


@router.put("/{prompt_id}", response_model=PromptOut)
async def update_prompt(
    prompt_id: UUID,
    body: PromptIn,
    session: AsyncSession = Depends(get_session),
) -> Prompt:
    prompt = await session.get(Prompt, prompt_id)
    if prompt is None:
        raise HTTPException(404, "Prompt not found")
    for key, value in body.model_dump().items():
        setattr(prompt, key, value)
    await session.flush()
    return prompt


@router.delete(
    "/{prompt_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,
)
async def delete_prompt(
    prompt_id: UUID, session: AsyncSession = Depends(get_session)
) -> Response:
    prompt = await session.get(Prompt, prompt_id)
    if prompt is None:
        raise HTTPException(404, "Prompt not found")
    await session.delete(prompt)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
