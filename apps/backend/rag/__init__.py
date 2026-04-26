"""RAG layer: chunking, embedding, indexing, semantic search (Qdrant)."""

from apps.backend.rag.chunking import chunk_text
from apps.backend.rag.parser import extract_text
from apps.backend.rag.service import (
    KnowledgePool,
    RAGService,
    get_rag,
    get_rag_agent_assist,
    get_rag_voice,
    rag_for_pool,
)

__all__ = [
    "KnowledgePool",
    "RAGService",
    "chunk_text",
    "extract_text",
    "get_rag",
    "get_rag_agent_assist",
    "get_rag_voice",
    "rag_for_pool",
]
