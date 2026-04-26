"""Token-aware chunking with overlap.

Falls back to a character-based heuristic if `tiktoken` is unavailable
(e.g. on-prem deployment without internet to download the encoder).
"""

from __future__ import annotations

from typing import Iterable

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # noqa: BLE001
    _ENC = None


def chunk_text(text: str, *, max_tokens: int = 400, overlap: int = 60) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    if _ENC is None:
        return _char_chunks(text, max_chars=max_tokens * 4, overlap=overlap * 4)

    tokens = _ENC.encode(text)
    chunks: list[str] = []
    step = max(1, max_tokens - overlap)
    for start in range(0, len(tokens), step):
        window = tokens[start : start + max_tokens]
        if not window:
            break
        chunks.append(_ENC.decode(window).strip())
        if start + max_tokens >= len(tokens):
            break
    return [c for c in chunks if c]


def _char_chunks(text: str, *, max_chars: int, overlap: int) -> list[str]:
    out: list[str] = []
    step = max(1, max_chars - overlap)
    for i in range(0, len(text), step):
        out.append(text[i : i + max_chars].strip())
        if i + max_chars >= len(text):
            break
    return [c for c in out if c]


def join_chunks(chunks: Iterable[str], *, sep: str = "\n---\n") -> str:
    return sep.join(c for c in chunks if c)
