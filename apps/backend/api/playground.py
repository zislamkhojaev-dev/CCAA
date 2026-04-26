"""Text-mode playground for QA (no audio).

Two endpoints:
  * `POST /chat`        — single-shot, full response with metadata
  * `POST /chat/stream` — Server-Sent Events: meta first, then text deltas
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from apps.backend.core import get_orchestrator
from apps.backend.core.bot_runtime import get_bot_runtime_payload_sync
from apps.backend.core.intent import IntentDetector
from apps.backend.core.prompts import fallback_message
from apps.backend.models.schemas import PlaygroundRequest, PlaygroundResponse
from apps.backend.rag import get_rag_voice
from apps.backend.services import get_llm

router = APIRouter()


@router.post("/chat", response_model=PlaygroundResponse)
async def playground_chat(req: PlaygroundRequest) -> PlaygroundResponse:
    orchestrator = get_orchestrator()
    return await orchestrator.answer(
        req.text, locale=req.locale, history=req.history
    )


@router.post("/chat/stream")
async def playground_chat_stream(req: PlaygroundRequest) -> StreamingResponse:
    """Stream the answer as SSE.

    Frame protocol (UTF-8, `data:` SSE format):
      event: meta   data: {"intent": ..., "sources": [...], "latency_ms": {...}}
      event: token  data: "<text fragment>"
      event: done   data: {}
    """
    llm = get_llm()
    rag = get_rag_voice()
    orchestrator = get_orchestrator()
    intent_detector = IntentDetector(llm)

    async def gen():
        intent_task = asyncio.create_task(
            intent_detector.detect(req.text, locale=req.locale)
        )
        rag_task = asyncio.create_task(
            rag.search(req.text, locale=req.locale, top_k=5)
        )
        intent, sources = await asyncio.gather(intent_task, rag_task)

        meta = {
            "intent": intent.model_dump(),
            "sources": [s.model_dump(mode="json") for s in sources],
        }
        yield _sse("meta", meta)

        if intent.requires_human:
            yield _sse("token", _handoff(req.locale))
            yield _sse("done", {})
            return
        if not sources:
            yield _sse("token", fallback_message(req.locale))
            yield _sse("done", {})
            return

        messages = orchestrator.build_messages(
            req.text, req.history, sources, req.locale
        )
        rt = get_bot_runtime_payload_sync()
        try:
            async for delta in llm.stream_complete(
                messages,
                temperature=float(rt.get("llm_temperature", 0.2)),
                max_tokens=int(rt.get("llm_max_tokens", 300)),
            ):
                yield _sse("token", delta)
        except Exception as exc:  # noqa: BLE001
            yield _sse("error", {"message": str(exc)})
        yield _sse("done", {})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(event: str, data) -> bytes:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def _handoff(locale: str) -> str:
    if locale == "uz":
        return "Bu savol bo‘yicha sizni operatorga ulayman."
    return "По этому вопросу вас лучше переключить на оператора. Соединяю."
