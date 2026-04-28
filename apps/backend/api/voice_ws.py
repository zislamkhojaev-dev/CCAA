"""WebSocket: голосовой канал + запись транскрипта + эскалация."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from apps.backend.config import get_settings
from apps.backend.core.bot_runtime import get_bot_runtime_payload_sync
from apps.backend.core import get_orchestrator
from apps.backend.core.conversation_recorder import ConversationRecorder, create_conversation
from apps.backend.core.voice_engine import VoiceEngine
from apps.backend.models.db import session_scope
from apps.backend.models.entities import Voice
from apps.backend.utils.logging import get_logger
from apps.backend.utils.ws_limits import release as ws_release
from apps.backend.utils.ws_limits import try_acquire as ws_try_acquire

router = APIRouter()
log = get_logger(__name__)


@router.websocket("/ws/voice")
async def voice_ws(ws: WebSocket) -> None:
    rt = get_bot_runtime_payload_sync()
    acquired_scopes: list[str] = []
    max_global = max(1, int(rt.get("ws_voice_max_connections", 50)))
    ip = (ws.client.host if ws.client else "") or "unknown"
    token = str(ws.query_params.get("token") or ws.query_params.get("session_token") or "").strip()
    scopes: list[tuple[str, int]] = [("voice_ws", max_global)]
    scopes.append(
        (
            f"voice_ws:ip:{ip}",
            max(1, int(rt.get("ws_voice_max_connections_per_ip", 8))),
        )
    )
    if token:
        scopes.append(
            (
                f"voice_ws:token:{token}",
                max(1, int(rt.get("ws_voice_max_connections_per_token", 4))),
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
    voice_id = ws.query_params.get("voice_id")
    sample_rate = int(ws.query_params.get("sample_rate", "16000"))

    conv_id = await create_conversation(channel="voice_ws", locale=locale)
    recorder = ConversationRecorder(conv_id)

    audio_in_q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=256)
    audio_out_q: asyncio.Queue[bytes | dict | None] = asyncio.Queue(maxsize=256)

    async def send_event(name: str, payload: dict | None = None) -> None:
        data = payload or {}
        body = {"type": name, **data, "v": 1, "event": name, "data": data}
        try:
            await ws.send_json(body)
        except (RuntimeError, WebSocketDisconnect):
            pass

    async def emit_escalation(pkg: dict) -> None:
        await send_event("escalation_packet", pkg)

    async def emit_state(state_evt: dict) -> None:
        await send_event("state", state_evt)

    effective_voice_id: str | None = None
    voice_tts: dict = {}
    async with session_scope() as s:
        vrow = None
        if voice_id:
            try:
                vrow = await s.get(Voice, UUID(voice_id))
            except ValueError:
                vrow = None
        if vrow is None:
            r = await s.execute(select(Voice).where(Voice.is_default.is_(True)).limit(1))
            vrow = r.scalar_one_or_none()
        if vrow is not None:
            effective_voice_id = vrow.provider_voice_id
            voice_tts = dict(vrow.tts_params or {})
    if not effective_voice_id:
        st = get_settings()
        if st.tts_provider == "elevenlabs":
            effective_voice_id = st.elevenlabs_voice_id
        elif st.tts_provider == "openai":
            effective_voice_id = st.openai_tts_voice
        else:
            effective_voice_id = voice_id or "default"

    engine = VoiceEngine(
        get_orchestrator(),
        conversation_recorder=recorder,
        on_escalation=emit_escalation,
        on_state_change=emit_state,
        voice_tts_params=voice_tts,
    )

    async def reader() -> None:
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data:
                    await audio_in_q.put(data)
                    continue
                text = msg.get("text")
                if text:
                    try:
                        ctrl = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    t = ctrl.get("type")
                    if t == "end":
                        break
                    if t == "interrupt":
                        await engine.interrupt()
                        await send_event("interrupted")
                        continue
                    if t == "escalate":
                        await engine.interrupt()
                        reason = str(ctrl.get("reason") or "manual")
                        await recorder.mark_escalated(reason)
                        pkg = await recorder.build_escalation_packet(
                            locale=locale, reason=reason
                        )
                        await emit_escalation(pkg)
                        continue
                    if t == "audio_played_ack":
                        try:
                            idx = int(ctrl.get("index"))
                        except Exception:  # noqa: BLE001
                            continue
                        await engine.mark_audio_played(idx)
                        continue
        finally:
            await audio_in_q.put(None)

    async def writer() -> None:
        while True:
            chunk = await audio_out_q.get()
            if chunk is None:
                break
            if isinstance(chunk, dict):
                if "type" in chunk:
                    name = str(chunk.get("type") or "event")
                    payload = dict(chunk)
                    payload.pop("type", None)
                    await send_event(name, payload)
                else:
                    await send_event("event", dict(chunk))
                continue
            try:
                await ws.send_bytes(chunk)
            except (RuntimeError, WebSocketDisconnect):
                break

    async def audio_in_iter() -> AsyncIterator[bytes]:
        while True:
            chunk = await audio_in_q.get()
            if chunk is None:
                return
            yield chunk

    try:
        await send_event("session", {"conversation_id": str(conv_id)})
    except Exception:  # noqa: BLE001
        pass

    reader_task = asyncio.create_task(reader(), name="ws-reader")
    writer_task = asyncio.create_task(writer(), name="ws-writer")
    engine_task = asyncio.create_task(
        engine.run(
            audio_in=audio_in_iter(),
            audio_out=audio_out_q,
            locale=locale,
            voice_id=effective_voice_id,
            sample_rate=sample_rate,
        ),
        name="ws-engine",
    )

    try:
        await asyncio.wait(
            {reader_task, writer_task, engine_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
    except WebSocketDisconnect:
        log.info("voice_ws_client_disconnected")
    finally:
        for scope in reversed(acquired_scopes):
            await ws_release(scope)
        for task in (reader_task, writer_task, engine_task):
            task.cancel()
        for task in (reader_task, writer_task, engine_task):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await recorder.mark_completed()
        except Exception as exc:  # noqa: BLE001
            log.warning("conversation_finalize_failed", error=str(exc))
        try:
            await ws.close()
        except RuntimeError:
            pass
