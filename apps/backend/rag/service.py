"""High-level RAG facade backed by Qdrant.

Two logical pools (separate Qdrant collections):
  * **voice** — голосовой бот / оркестратор / playground voice.
  * **agent_assist** — суфлёр оператора (может быть уже и по другим правилам).

Hard rule from FT: the bot MUST NOT invent answers outside the
knowledge base. We enforce this in two places:
  1. `search` returns scored chunks; callers fall back to the
     "transfer to human" branch when nothing crosses the threshold.
  2. The default system prompt (see `core/prompts.py`) instructs the
     model to refuse on empty context.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from functools import lru_cache
from typing import Literal, Sequence

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from apps.backend.config import get_settings
from apps.backend.models.schemas import RAGChunk
from apps.backend.rag.chunking import chunk_text
from apps.backend.services import get_llm
from apps.backend.utils.logging import get_logger

log = get_logger(__name__)

KnowledgePool = Literal["voice", "agent_assist"]


class RAGService:
    """One Qdrant collection per instance — never mix voice / assist vectors."""

    def __init__(self, *, collection_name: str) -> None:
        settings = get_settings()
        self._client = AsyncQdrantClient(
            host=settings.qdrant_host, port=settings.qdrant_port
        )
        self._collection = collection_name
        self._dim = settings.openai_embedding_dim
        self._llm = get_llm()

    @property
    def collection_name(self) -> str:
        return self._collection

    async def ensure_collection(self) -> None:
        existing = await self._client.get_collections()
        names = {c.name for c in existing.collections}
        if self._collection in names:
            return
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qm.VectorParams(size=self._dim, distance=qm.Distance.COSINE),
        )
        log.info("qdrant_collection_created", name=self._collection)

    async def index_document(
        self,
        *,
        document_id: uuid.UUID,
        title: str,
        text: str,
        locale: str = "ru",
    ) -> int:
        await self.ensure_collection()
        chunks = chunk_text(text)
        if not chunks:
            return 0

        vectors = await self._llm.embed(chunks)
        points = [
            qm.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "document_id": str(document_id),
                    "title": title,
                    "locale": locale,
                    "chunk_index": idx,
                    "text": chunk,
                },
            )
            for idx, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]
        await self._client.upsert(collection_name=self._collection, points=points)
        log.info(
            "document_indexed",
            document_id=str(document_id),
            chunks=len(points),
            collection=self._collection,
        )
        return len(points)

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        locale: str | None = None,
        score_threshold: float = 0.55,
    ) -> list[RAGChunk]:
        await self.ensure_collection()
        if not query.strip():
            return []
        [vector] = await self._llm.embed([query])

        flt: qm.Filter | None = None
        if locale:
            flt = qm.Filter(
                must=[qm.FieldCondition(key="locale", match=qm.MatchValue(value=locale))]
            )

        hits = await self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=flt,
        )
        return [
            RAGChunk(
                document_id=uuid.UUID(h.payload["document_id"]),
                chunk_id=str(h.id),
                text=h.payload.get("text", ""),
                score=float(h.score),
                title=h.payload.get("title"),
            )
            for h in hits
        ]

    async def delete_document(self, document_id: uuid.UUID) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="document_id",
                            match=qm.MatchValue(value=str(document_id)),
                        )
                    ]
                )
            ),
        )

    async def list_indexed_documents(self) -> list[dict]:
        """Best-effort document list reconstructed from Qdrant payload.

        Нужен как fallback, когда `documents` в PostgreSQL пусты/потеряны,
        а вектора в Qdrant уже существуют.
        """
        await self.ensure_collection()
        docs: dict[str, dict] = {}
        offset = None
        while True:
            points, next_offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=None,
                with_payload=True,
                with_vectors=False,
                limit=512,
                offset=offset,
            )
            for p in points:
                payload = p.payload or {}
                doc_id = str(payload.get("document_id") or "").strip()
                if not doc_id:
                    continue
                row = docs.get(doc_id)
                if row is None:
                    title = str(payload.get("title") or "Indexed document")
                    locale = str(payload.get("locale") or "ru")
                    row = {
                        "id": doc_id,
                        "title": title,
                        "locale": locale,
                        "chunk_count": 0,
                        # В Qdrant нет created_at документа; используем deterministic fallback.
                        "created_at": datetime.now(timezone.utc),
                    }
                    docs[doc_id] = row
                row["chunk_count"] += 1
            if next_offset is None:
                break
            offset = next_offset
        return list(docs.values())

    @staticmethod
    def format_context(chunks: Sequence[RAGChunk]) -> str:
        if not chunks:
            return ""
        return "\n\n".join(
            f"[Источник: {c.title or c.chunk_id}]\n{c.text}" for c in chunks
        )


@lru_cache(maxsize=1)
def get_rag_voice() -> RAGService:
    s = get_settings()
    name = s.qdrant_collection_voice or s.qdrant_collection
    return RAGService(collection_name=name)


@lru_cache(maxsize=1)
def get_rag_agent_assist() -> RAGService:
    s = get_settings()
    return RAGService(collection_name=s.qdrant_collection_agent_assist)


def get_rag() -> RAGService:
    """Backward-compatible alias for the voice-bot knowledge pool."""
    return get_rag_voice()


def rag_for_pool(pool: KnowledgePool) -> RAGService:
    if pool == "agent_assist":
        return get_rag_agent_assist()
    return get_rag_voice()
