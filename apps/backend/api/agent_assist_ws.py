"""WebSocket: real-time suggestions for the operator dashboard."""

from __future__ import annotations

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from apps.backend.core.bot_runtime import get_bot_runtime_payload_sync
from apps.backend.core import AgentAssist
from apps.backend.utils.logging import get_logger
from apps.backend.utils.ws_limits import release as ws_release
from apps.backend.utils.ws_limits import try_acquire as ws_try_acquire

router = APIRouter()
log = get_logger(__name__)


@router.websocket("/ws/agent-assist")
async def agent_assist_ws(ws: WebSocket) -> None:
    rt = get_bot_runtime_payload_sync()
    acquired_scopes: list[str] = []
    max_global = max(1, int(rt.get("ws_agent_assist_max_connections", 100)))
    ip = (ws.client.host if ws.client else "") or "unknown"
    token = str(ws.query_params.get("token") or ws.query_params.get("session_token") or "").strip()
    scopes: list[tuple[str, int]] = [("agent_assist_ws", max_global)]
    scopes.append(
        (
            f"agent_assist_ws:ip:{ip}",
            max(1, int(rt.get("ws_agent_assist_max_connections_per_ip", 12))),
        )
    )
    if token:
        scopes.append(
            (
                f"agent_assist_ws:token:{token}",
                max(1, int(rt.get("ws_agent_assist_max_connections_per_token", 6))),
            )
        )
    for scope, limit in scopes:
        ok = await ws_try_acquire(scope, limit)
        if ok:
            acquired_scopes.append(scope)
            continue
        for acq in reversed(acquired_scopes):
            await ws_release(acq)
        await ws.accept()
        await ws.send_json({"type": "error", "message": "too_many_connections"})
        await ws.close(code=1013)
        return
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
        for scope in reversed(acquired_scopes):
            await ws_release(scope)
        try:
            await ws.close()
        except RuntimeError:
            pass
