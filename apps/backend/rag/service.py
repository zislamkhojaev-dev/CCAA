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

import math
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from typing import Literal, Sequence

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from apps.backend.config import get_settings
from apps.backend.core.bot_runtime import get_bot_runtime_payload_sync
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
        rt = get_bot_runtime_payload_sync()
        [vector] = await self._llm.embed([query])

        flt: qm.Filter | None = None
        if locale:
            flt = qm.Filter(
                must=[qm.FieldCondition(key="locale", match=qm.MatchValue(value=locale))]
            )

        candidate_pool = max(top_k, int(rt.get("rag_candidate_pool_size", 20)))
        hits = await self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=max(top_k * 2, candidate_pool),
            score_threshold=score_threshold,
            query_filter=flt,
        )
        candidates = [
            RAGChunk(
                document_id=uuid.UUID(h.payload["document_id"]),
                chunk_id=str(h.id),
                text=h.payload.get("text", ""),
                score=float(h.score),
                title=h.payload.get("title"),
            )
            for h in hits
        ]
        if not bool(rt.get("rag_hybrid_enabled", True)):
            return candidates[:top_k]
        alpha = float(rt.get("rag_hybrid_alpha", 0.65))
        alpha = max(0.0, min(1.0, alpha))
        return self._hybrid_rerank(query=query, chunks=candidates, top_k=top_k, alpha=alpha)

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

    @staticmethod
    def _hybrid_rerank(query: str, chunks: list[RAGChunk], *, top_k: int, alpha: float) -> list[RAGChunk]:
        if not chunks:
            return []
        bm25 = _bm25_scores(query, [c.text for c in chunks])
        vec_scores = [float(c.score) for c in chunks]
        vec_min = min(vec_scores)
        vec_max = max(vec_scores)
        bm_min = min(bm25) if bm25 else 0.0
        bm_max = max(bm25) if bm25 else 0.0

        def norm(v: float, lo: float, hi: float) -> float:
            if hi <= lo:
                return 0.0
            return (v - lo) / (hi - lo)

        ranked = []
        for idx, ch in enumerate(chunks):
            v = norm(float(ch.score), vec_min, vec_max)
            b = norm(bm25[idx], bm_min, bm_max)
            fused = alpha * v + (1.0 - alpha) * b
            ranked.append((fused, ch))
        ranked.sort(key=lambda x: x[0], reverse=True)
        out = [c for _, c in ranked[:top_k]]
        return out


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


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _bm25_scores(query: str, docs: list[str], *, k1: float = 1.5, b: float = 0.75) -> list[float]:
    q_terms = _tokenize(query)
    if not q_terms or not docs:
        return [0.0 for _ in docs]
    doc_tokens = [_tokenize(d) for d in docs]
    doc_lens = [len(toks) for toks in doc_tokens]
    avgdl = (sum(doc_lens) / len(doc_lens)) if doc_lens else 0.0
    if avgdl <= 0:
        return [0.0 for _ in docs]
    term_df: Counter[str] = Counter()
    for toks in doc_tokens:
        for t in set(toks):
            term_df[t] += 1
    n_docs = len(doc_tokens)
    q_tf = Counter(q_terms)
    scores: list[float] = []
    for toks in doc_tokens:
        tf = Counter(toks)
        dl = len(toks)
        s = 0.0
        for term, qf in q_tf.items():
            df = term_df.get(term, 0)
            if df <= 0:
                continue
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            f = tf.get(term, 0)
            denom = f + k1 * (1.0 - b + b * (dl / avgdl))
            if denom <= 0:
                continue
            s += idf * ((f * (k1 + 1.0)) / denom) * qf
        scores.append(s)
    return scores
