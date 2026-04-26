"""Knowledge-base management — separate Qdrant pools for voice vs agent assist."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.backend.models import Document, DocumentOut, get_session
from apps.backend.rag import extract_text, rag_for_pool

router = APIRouter()

KnowledgePoolParam = Literal["voice", "agent_assist"]


@router.get("", response_model=list[DocumentOut])
async def list_documents(
    pool: KnowledgePoolParam | None = Query(
        default=None,
        description="Фильтр по пулу: voice | agent_assist; без параметра — все",
    ),
    session: AsyncSession = Depends(get_session),
) -> list[Document]:
    q = select(Document).order_by(Document.created_at.desc())
    if pool:
        q = q.where(Document.knowledge_pool == pool)
    result = await session.execute(q)
    return list(result.scalars().all())


@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    locale: str = Form(default="ru"),
    knowledge_pool: KnowledgePoolParam = Form(
        default="voice",
        description="voice — база голосового бота; agent_assist — отдельная база суфлёра",
    ),
    session: AsyncSession = Depends(get_session),
) -> Document:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    try:
        text = extract_text(file.filename or "upload.txt", raw)
    except ValueError as exc:
        raise HTTPException(415, str(exc)) from exc
    if not text.strip():
        raise HTTPException(422, "No extractable text")

    doc = Document(
        title=title or file.filename or "untitled",
        locale=locale,
        knowledge_pool=knowledge_pool,
    )
    session.add(doc)
    await session.flush()

    rag = rag_for_pool(knowledge_pool)
    chunks = await rag.index_document(
        document_id=doc.id, title=doc.title, text=text, locale=locale
    )
    doc.chunk_count = chunks
    await session.flush()
    return doc


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: UUID, session: AsyncSession = Depends(get_session)
) -> None:
    doc = await session.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    pool: KnowledgePoolParam = (
        doc.knowledge_pool
        if doc.knowledge_pool in ("voice", "agent_assist")
        else "voice"
    )
    await rag_for_pool(pool).delete_document(document_id)
    await session.delete(doc)
