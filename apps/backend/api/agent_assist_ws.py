"""WebSocket: real-time suggestions for the operator dashboard."""

from __future__ import annotations

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from apps.backend.core import AgentAssist
from apps.backend.utils.logging import get_logger

router = APIRouter()
log = get_logger(__name__)


@router.websocket("/ws/agent-assist")
async def agent_assist_ws(ws: WebSocket) -> None:
    await ws.accept()
    locale = ws.query_params.get("locale", "ru")
    session = AgentAssist()

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "invalid_json"})
                continue

            kind = msg.get("type")
            if kind == "reset":
                session.reset()
                await ws.send_json({"type": "reset_ack"})
                continue

            if kind != "utterance":
                await ws.send_json({"type": "error", "message": f"unknown_type:{kind}"})
                continue

            speaker = msg.get("speaker", "customer")
            text = (msg.get("text") or "").strip()
            if not text:
                continue

            if speaker == "agent":
                session.add_agent(text)
                continue

            suggestion = await session.on_customer_utterance(text, locale=locale)
            payload = {
                "type": "suggestion",
                "answer": suggestion.answer,
                "intent": suggestion.intent.model_dump(),
                "sources": [s.model_dump(mode="json") for s in suggestion.sources],
                "latency_ms": suggestion.latency_ms,
                "warning": suggestion.warning,
            }
            await ws.send_json(payload)

    except WebSocketDisconnect:
        log.info("agent_assist_disconnected")
    finally:
        try:
            await ws.close()
        except RuntimeError:
            pass
