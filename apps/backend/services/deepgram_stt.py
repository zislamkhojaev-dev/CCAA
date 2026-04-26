"""Deepgram streaming STT client.

Uses the official `deepgram-sdk` v3 async interface. We keep a single
WebSocket connection per call and forward PCM/Opus chunks straight from
the user. Transcripts are surfaced as `STTEvent` items so the engine can
react to interim results (and start LLM generation early).
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveOptions,
    LiveTranscriptionEvents,
)

from apps.backend.config import get_settings
from apps.backend.services.interfaces import STTEvent, STTService
from apps.backend.utils.logging import get_logger

log = get_logger(__name__)


class DeepgramSTT(STTService):
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.deepgram_api_key:
            log.warning("deepgram_api_key_missing")
        self._client = DeepgramClient(
            settings.deepgram_api_key,
            DeepgramClientOptions(options={"keepalive": "true"}),
        )
        self._model = settings.deepgram_model

    async def stream_transcribe(
        self,
        audio_chunks: AsyncIterator[bytes],
        *,
        locale: str = "ru",
        sample_rate: int = 16_000,
    ) -> AsyncIterator[STTEvent]:
        connection = self._client.listen.asyncwebsocket.v("1")
        out_queue: asyncio.Queue[STTEvent | None] = asyncio.Queue()

        async def on_transcript(_self, result, **_kwargs):
            try:
                alt = result.channel.alternatives[0]
                text = (alt.transcript or "").strip()
                if not text:
                    return
                await out_queue.put(
                    STTEvent(
                        text=text,
                        is_final=bool(getattr(result, "is_final", False)),
                        confidence=float(getattr(alt, "confidence", 0.0) or 0.0),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                log.error("deepgram_event_error", error=str(exc))

        async def on_error(_self, error, **_kwargs):
            log.error("deepgram_error", error=str(error))
            await out_queue.put(None)

        async def on_close(_self, *_args, **_kwargs):
            await out_queue.put(None)

        connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
        connection.on(LiveTranscriptionEvents.Error, on_error)
        connection.on(LiveTranscriptionEvents.Close, on_close)

        options = LiveOptions(
            model=self._model,
            language=locale if locale in {"ru", "en"} else "ru",
            encoding="linear16",
            channels=1,
            sample_rate=sample_rate,
            interim_results=True,
            smart_format=True,
            punctuate=True,
        )

        if not await connection.start(options):
            raise RuntimeError("Failed to start Deepgram connection")

        async def pump_audio() -> None:
            try:
                async for chunk in audio_chunks:
                    if not chunk:
                        continue
                    await connection.send(chunk)
            finally:
                await connection.finish()

        pump_task = asyncio.create_task(pump_audio(), name="deepgram-pump")

        try:
            while True:
                event = await out_queue.get()
                if event is None:
                    break
                yield event
        finally:
            pump_task.cancel()
            try:
                await pump_task
            except (asyncio.CancelledError, Exception):
                pass

    async def aclose(self) -> None:
        # SDK manages per-call connections; nothing global to close here.
        return None
