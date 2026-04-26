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
import re
import time
import wave
from array import array
from typing import AsyncIterator

from openai import AsyncOpenAI

from apps.backend.config import get_settings
from apps.backend.core.bot_runtime import get_bot_runtime_payload_sync
from apps.backend.services.interfaces import STTEvent, STTService
from apps.backend.utils.logging import get_logger

log = get_logger(__name__)

_JUNK_PATTERNS = [
    re.compile(r"редактор\s+субтитров", re.IGNORECASE),
    re.compile(r"корректор\s+[а-яa-z]\.", re.IGNORECASE),
    re.compile(r"продолжение\s+следует", re.IGNORECASE),
    re.compile(r"с\s+вами\s+был", re.IGNORECASE),
]


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
        voiced_bytes = 0
        last_emit_text = ""
        last_emit_at = 0.0

        async def flush(reason: str) -> None:
            nonlocal buf, voiced_bytes, last_emit_at, last_emit_text
            rt = get_bot_runtime_payload_sync()
            min_voiced_sec = max(0.05, float(rt.get("stt_min_voiced_seconds", 0.2)))
            duplicate_cooldown = max(0.0, float(rt.get("stt_duplicate_cooldown_sec", 4.0)))
            if len(buf) < sample_rate:  # < ~0.5 s of audio: skip
                return
            if voiced_bytes < int(sample_rate * 2 * min_voiced_sec):
                # Почти нет речи, это фон/шум — не шлём в Whisper.
                buf = bytearray()
                voiced_bytes = 0
                return
            wav_bytes = _pcm16_to_wav(bytes(buf), sample_rate=sample_rate)
            buf = bytearray()
            voiced_bytes = 0
            try:
                resp = await self._client.audio.transcriptions.create(
                    model=self._model,
                    file=("speech.wav", wav_bytes, "audio/wav"),
                    language=locale if locale in {"ru", "en"} else "ru",
                    response_format="json",
                    temperature=0,
                )
                text = _normalize_text(resp.text or "")
                if text:
                    if _is_junk_text(text):
                        log.info("whisper_discarded_junk", text=text[:120])
                        return
                    now = time.monotonic()
                    if text == last_emit_text and (now - last_emit_at) < duplicate_cooldown:
                        log.info("whisper_discarded_duplicate", text=text[:120])
                        return
                    last_emit_text = text
                    last_emit_at = now
                    log.info("whisper_flush", reason=reason, text_len=len(text))
                    await out_q.put(STTEvent(text=text, is_final=True, confidence=0.9))
            except Exception as exc:  # noqa: BLE001
                log.error("whisper_error", error=str(exc))

        async def consumer() -> None:
            nonlocal buf, last_chunk_at, voiced_bytes
            try:
                async for chunk in audio_chunks:
                    if not chunk:
                        continue
                    rt = get_bot_runtime_payload_sync()
                    rms_threshold = max(0.001, float(rt.get("stt_voice_rms_threshold", 0.008)))
                    rms = _pcm_rms(chunk)
                    if rms >= rms_threshold:
                        buf.extend(chunk)
                        voiced_bytes += len(chunk)
                        last_chunk_at = time.monotonic()
                    elif buf:
                        # Сохраняем короткие "хвосты" тишины внутри уже начавшейся фразы.
                        buf.extend(chunk)
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


def _pcm_rms(pcm: bytes) -> float:
    if len(pcm) < 2:
        return 0.0
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return 0.0
    acc = 0.0
    for s in samples:
        v = s / 32768.0
        acc += v * v
    return (acc / len(samples)) ** 0.5


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _is_junk_text(text: str) -> bool:
    t = _normalize_text(text)
    if len(t) < 2:
        return True
    if not any(ch.isalnum() for ch in t):
        return True
    for rx in _JUNK_PATTERNS:
        if rx.search(t):
            return True
    return False
