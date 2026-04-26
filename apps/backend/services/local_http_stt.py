"""Локальный ASR через HTTP: отправка WAV/PCM чанков на ваш сервис.

Ожидается `LOCAL_STT_URL` (полный URL, например http://whisper:8080/transcribe).
POST multipart: поле `file` — WAV mono 16 kHz (как у OpenAI Whisper batch).

Ответ: JSON `{"text":"..."}` или OpenAI-стиль `{"text": "..."}` из поля text/transcript.
"""

from __future__ import annotations

import asyncio
import io
import time
import wave
from typing import AsyncIterator

import httpx

from apps.backend.config import get_settings
from apps.backend.services.interfaces import STTEvent, STTService
from apps.backend.utils.logging import get_logger

log = get_logger(__name__)


class LocalHttpSTT(STTService):
    """Буферизация PCM → WAV, как у OpenAI Whisper STT, но POST на локальный URL."""

    def __init__(self) -> None:
        s = get_settings()
        self._url = (s.local_stt_url or "").strip()
        self._flush_ms = s.openai_stt_flush_ms
        self._silence_ms = s.openai_stt_silence_ms
        # Полный URL в `LOCAL_STT_URL` (без базового префикса клиента).
        self._client = httpx.AsyncClient(timeout=120.0)

    async def stream_transcribe(
        self,
        audio_chunks: AsyncIterator[bytes],
        *,
        locale: str = "ru",
        sample_rate: int = 16_000,
    ) -> AsyncIterator[STTEvent]:
        if not self._url:
            log.error("local_stt_url_missing")
            yield STTEvent(text="[local stt: URL не задан]", is_final=True, confidence=0.0)
            return

        out_q: asyncio.Queue[STTEvent | None] = asyncio.Queue()
        buf = bytearray()
        last_chunk_at = time.monotonic()

        async def flush_buf(reason: str) -> None:
            if len(buf) < sample_rate:
                return
            wav = _pcm16_to_wav(bytes(buf), sample_rate=sample_rate)
            buf.clear()
            try:
                files = {"file": ("speech.wav", wav, "audio/wav")}
                r = await self._client.post(self._url, files=files, data={"language": locale})
                r.raise_for_status()
                data = r.json()
                text = (
                    (data.get("text") or data.get("transcript") or data.get("result") or "")
                    .strip()
                )
                if text:
                    log.info("local_stt_flush", reason=reason, len=len(text))
                    await out_q.put(STTEvent(text=text, is_final=True, confidence=0.9))
            except Exception as exc:  # noqa: BLE001
                log.error("local_stt_error", error=str(exc))

        async def consumer() -> None:
            nonlocal buf, last_chunk_at
            try:
                async for chunk in audio_chunks:
                    if not chunk:
                        continue
                    buf.extend(chunk)
                    last_chunk_at = time.monotonic()
                    if len(buf) >= sample_rate * 2 * (self._flush_ms / 1000.0):
                        await flush_buf("size")
            finally:
                await flush_buf("eos")
                await out_q.put(None)

        async def silence_watcher() -> None:
            while True:
                await asyncio.sleep(self._silence_ms / 1000.0 / 2)
                if buf and (time.monotonic() - last_chunk_at) * 1000.0 >= self._silence_ms:
                    await flush_buf("silence")

        cons = asyncio.create_task(consumer())
        watch = asyncio.create_task(silence_watcher())
        try:
            while True:
                ev = await out_q.get()
                if ev is None:
                    break
                yield ev
        finally:
            watch.cancel()
            cons.cancel()
            for t in (watch, cons):
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

    async def aclose(self) -> None:
        await self._client.aclose()


def _pcm16_to_wav(pcm: bytes, *, sample_rate: int) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return out.getvalue()
