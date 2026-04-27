from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi.testclient import TestClient

from apps.backend.main import create_app


class _FakeVoiceEngine:
    def __init__(self, _orch, **kwargs) -> None:
        self._on_state_change = kwargs.get("on_state_change")

    async def interrupt(self) -> None:
        if self._on_state_change:
            await self._on_state_change({"state": "interrupted", "reason": "manual"})

    async def run(self, *, audio_in, audio_out, locale: str, voice_id: str | None = None, sample_rate: int = 16000):
        if self._on_state_change:
            await self._on_state_change({"state": "listening", "reason": ""})
        async for _ in audio_in:
            await audio_out.put({"type": "audio_segment_start", "index": 0})
            await audio_out.put(b"fake-audio")
            await audio_out.put({"type": "audio_segment_end", "index": 0})
            break
        await audio_out.put(None)


class _InterruptVoiceEngine(_FakeVoiceEngine):
    async def run(
        self, *, audio_in, audio_out, locale: str, voice_id: str | None = None, sample_rate: int = 16000
    ):
        if self._on_state_change:
            await self._on_state_change({"state": "speaking", "reason": ""})
        async for _ in audio_in:
            break
        await audio_out.put(None)


class _SilenceVoiceEngine(_FakeVoiceEngine):
    async def run(
        self, *, audio_in, audio_out, locale: str, voice_id: str | None = None, sample_rate: int = 16000
    ):
        if self._on_state_change:
            await self._on_state_change({"state": "listening", "reason": ""})
        await audio_out.put({"type": "silence_nudge", "text": "Вы еще на линии?"})
        await audio_out.put(None)


class _DummyRecorder:
    def __init__(self, _cid) -> None:
        pass

    async def mark_escalated(self, _reason: str) -> None:
        return None

    async def build_escalation_packet(self, *, locale: str, reason: str) -> dict:
        return {"conversation_id": str(uuid4()), "locale": locale, "reason": reason, "summary": "", "turns": []}

    async def mark_completed(self) -> None:
        return None


@asynccontextmanager
async def _fake_session_scope():
    class _S:
        async def get(self, *args, **kwargs):
            return None

        async def execute(self, *args, **kwargs):
            class _R:
                def scalar_one_or_none(self):
                    return None

            return _R()

    yield _S()


def _patch_ws_deps(monkeypatch, engine_cls) -> None:
    from apps.backend.api import voice_ws as vw

    monkeypatch.setattr(vw, "VoiceEngine", engine_cls)
    monkeypatch.setattr(vw, "ConversationRecorder", _DummyRecorder)
    monkeypatch.setattr(vw, "create_conversation", lambda **kwargs: asyncio.sleep(0, result=uuid4()))
    monkeypatch.setattr(vw, "session_scope", _fake_session_scope)
    monkeypatch.setattr(vw, "get_orchestrator", lambda: object())


def test_voice_ws_emits_state_and_audio(monkeypatch) -> None:
    _patch_ws_deps(monkeypatch, _FakeVoiceEngine)
    app = create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/ws/voice?locale=ru") as ws:
            ws.send_bytes(b"\x00\x01")
            got = [ws.receive_json() for _ in range(5)]
            types = {m.get("type") for m in got}
            assert "state" in types
            assert "audio_segment_start" in types


def test_voice_ws_interrupt_event(monkeypatch) -> None:
    _patch_ws_deps(monkeypatch, _InterruptVoiceEngine)
    app = create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/ws/voice?locale=ru") as ws:
            ws.send_text('{"type":"interrupt"}')
            got = [ws.receive_json() for _ in range(5)]
            types = {m.get("type") for m in got}
            assert "interrupted" in types


def test_voice_ws_silence_event(monkeypatch) -> None:
    _patch_ws_deps(monkeypatch, _SilenceVoiceEngine)
    app = create_app()
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/ws/voice?locale=ru") as ws:
            got = [ws.receive_json() for _ in range(4)]
            types = {m.get("type") for m in got}
            assert "silence_nudge" in types


def test_slo_snapshot_endpoint() -> None:
    app = create_app()
    with TestClient(app) as client:
        r = client.get("/api/v1/slo")
        assert r.status_code == 200
        body = r.json()
        assert "metrics" in body
