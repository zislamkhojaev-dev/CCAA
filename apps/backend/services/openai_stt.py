"""OpenAI Whisper STT provider with chunked buffering.

Whisper REST is batch-only, so we approximate streaming:
  * accumulate inbound PCM16 audio into a rolling buffer;
  * every `flush_interval_ms` (default 1.2 s) we flush the buffer as a
    16 kHz mono WAV to `audio.transcriptions.create()`;
  * we emit the result as a final `STTEvent`.

This loses sub-second latency vs. Deepgram, but lets the prototype be
tested end-to-end with just an OpenAI key. The Strategy interface keeps
the rest of the system identical.
"""

from __future__ import annotations

import asyncio
import io
import time
import wave
from typing import AsyncIterator

from openai import AsyncOpenAI

from apps.backend.config import get_settings
from apps.backend.services.interfaces import STTEvent, STTService
from apps.backend.utils.logging import get_logger

log = get_logger(__name__)


class OpenAIWhisperSTT(STTService):
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            log.warning("openai_api_key_missing")
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_stt_model
        self._flush_ms = settings.openai_stt_flush_ms
        self._silence_ms = settings.openai_stt_silence_ms

    async def stream_transcribe(
        self,
        audio_chunks: AsyncIterator[bytes],
        *,
        locale: str = "ru",
        sample_rate: int = 16_000,
    ) -> AsyncIterator[STTEvent]:
        out_q: asyncio.Queue[STTEvent | None] = asyncio.Queue()
        buf = bytearray()
        last_chunk_at = time.monotonic()

        async def flush(reason: str) -> None:
            nonlocal buf
            if len(buf) < sample_rate:  # < ~0.5 s of audio: skip
                return
            wav_bytes = _pcm16_to_wav(bytes(buf), sample_rate=sample_rate)
            buf = bytearray()
            try:
                resp = await self._client.audio.transcriptions.create(
                    model=self._model,
                    file=("speech.wav", wav_bytes, "audio/wav"),
                    language=locale if locale in {"ru", "en"} else "ru",
                    response_format="json",
                    temperature=0,
                )
                text = (resp.text or "").strip()
                if text:
                    log.info("whisper_flush", reason=reason, text_len=len(text))
                    await out_q.put(STTEvent(text=text, is_final=True, confidence=0.9))
            except Exception as exc:  # noqa: BLE001
                log.error("whisper_error", error=str(exc))

        async def consumer() -> None:
            nonlocal buf, last_chunk_at
            try:
                async for chunk in audio_chunks:
                    if not chunk:
                        continue
                    buf.extend(chunk)
                    last_chunk_at = time.monotonic()
                    if len(buf) >= sample_rate * 2 * (self._flush_ms / 1000.0):
                        await flush("size")
            finally:
                await flush("eos")
                await out_q.put(None)

        async def silence_watcher() -> None:
            while True:
                await asyncio.sleep(self._silence_ms / 1000.0 / 2)
                if buf and (time.monotonic() - last_chunk_at) * 1000.0 >= self._silence_ms:
                    await flush("silence")

        cons_task = asyncio.create_task(consumer(), name="whisper-consumer")
        watch_task = asyncio.create_task(silence_watcher(), name="whisper-watcher")

        try:
            while True:
                event = await out_q.get()
                if event is None:
                    break
                yield event
        finally:
            watch_task.cancel()
            cons_task.cancel()
            for t in (watch_task, cons_task):
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

    async def aclose(self) -> None:
        return None


def _pcm16_to_wav(pcm: bytes, *, sample_rate: int) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return out.getvalue()
