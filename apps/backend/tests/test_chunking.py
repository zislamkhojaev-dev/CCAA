from __future__ import annotations

from apps.backend.rag.chunking import chunk_text


def test_chunk_text_returns_non_empty() -> None:
    text = "Параграф один. " * 200
    chunks = chunk_text(text, max_tokens=100, overlap=20)
    assert chunks, "chunker must produce at least one chunk"
    assert all(c.strip() for c in chunks)


def test_chunk_text_handles_empty() -> None:
    assert chunk_text("") == []
    assert chunk_text("   ") == []
