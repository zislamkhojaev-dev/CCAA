"""Speech analytics: configurable criteria + recording analysis."""

from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from apps.backend.analytics import SpeechAnalyticsService
from apps.backend.models.db import get_session
from apps.backend.models.entities import AnalyticsCriterion, AnalyticsCriteriaSet
from apps.backend.models.schemas import (
    AnalyticsCriteriaSetIn,
    AnalyticsCriteriaSetOut,
    AnalyticsCriterionIn,
    AnalyticsCriterionOut,
    SpeechAnalysisResult,
)

router = APIRouter()


# ---------- Criteria sets ----------
@router.get("/criteria-sets", response_model=list[AnalyticsCriteriaSetOut])
async def list_criteria_sets(
    session: AsyncSession = Depends(get_session),
) -> list[AnalyticsCriteriaSet]:
    r = await session.execute(
        select(AnalyticsCriteriaSet).order_by(AnalyticsCriteriaSet.created_at.desc())
    )
    return list(r.scalars().all())


@router.post(
    "/criteria-sets",
    response_model=AnalyticsCriteriaSetOut,
    status_code=status.HTTP_201_CREATED,
)
async def create_criteria_set(
    body: AnalyticsCriteriaSetIn,
    session: AsyncSession = Depends(get_session),
) -> AnalyticsCriteriaSet:
    if body.is_default:
        await session.execute(update(AnalyticsCriteriaSet).values(is_default=False))
    row = AnalyticsCriteriaSet(**body.model_dump())
    session.add(row)
    await session.flush()
    return row


@router.delete("/criteria-sets/{set_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_criteria_set(
    set_id: UUID, session: AsyncSession = Depends(get_session)
) -> None:
    row = await session.get(AnalyticsCriteriaSet, set_id)
    if row is None:
        raise HTTPException(404, "Criteria set not found")
    await session.delete(row)


# ---------- Criteria ----------
@router.get(
    "/criteria-sets/{set_id}/criteria",
    response_model=list[AnalyticsCriterionOut],
)
async def list_criteria(
    set_id: UUID, session: AsyncSession = Depends(get_session)
) -> list[AnalyticsCriterion]:
    parent = await session.get(AnalyticsCriteriaSet, set_id)
    if parent is None:
        raise HTTPException(404, "Criteria set not found")
    r = await session.execute(
        select(AnalyticsCriterion)
        .where(AnalyticsCriterion.set_id == set_id)
        .order_by(AnalyticsCriterion.code)
    )
    return list(r.scalars().all())


@router.post(
    "/criteria-sets/{set_id}/criteria",
    response_model=AnalyticsCriterionOut,
    status_code=status.HTTP_201_CREATED,
)
async def add_criterion(
    set_id: UUID,
    body: AnalyticsCriterionIn,
    session: AsyncSession = Depends(get_session),
) -> AnalyticsCriterion:
    parent = await session.get(AnalyticsCriteriaSet, set_id)
    if parent is None:
        raise HTTPException(404, "Criteria set not found")
    row = AnalyticsCriterion(id=uuid4(), set_id=set_id, **body.model_dump())
    session.add(row)
    await session.flush()
    return row


@router.delete("/criteria/{criterion_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_criterion(
    criterion_id: UUID, session: AsyncSession = Depends(get_session)
) -> None:
    row = await session.get(AnalyticsCriterion, criterion_id)
    if row is None:
        raise HTTPException(404, "Criterion not found")
    await session.delete(row)


# ---------- Analyze ----------
@router.post("/analyze", response_model=SpeechAnalysisResult)
async def analyze_recording(
    file: UploadFile = File(...),
    criteria_set_id: UUID | None = Form(default=None),
    locale: str = Form(default="ru"),
    session: AsyncSession = Depends(get_session),
) -> SpeechAnalysisResult:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    svc = SpeechAnalyticsService()
    return await svc.analyze_recording(
        audio_bytes=raw,
        filename=file.filename or "recording.wav",
        session=session,
        criteria_set_id=criteria_set_id,
        locale=locale,
    )
