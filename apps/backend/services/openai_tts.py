"""OpenAI TTS provider (`tts-1` / `tts-1-hd`).

Why not stream byte-by-byte from OpenAI?
  The HTTP endpoint is *progressive* (you get audio as it's encoded) but
  not *generative* — request granularity is one synthesis call. To keep
  perceived latency low we still split incoming text on sentence
  boundaries and start playback as soon as the first sentence is ready.
  Audio is forwarded in small chunks via the SDK's streaming response.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from openai import AsyncOpenAI

from apps.backend.config import get_settings
from apps.backend.core.bot_runtime import get_bot_runtime_payload_sync
from apps.backend.services.interfaces import TTSService
from apps.backend.utils.logging import get_logger
from apps.backend.utils.resilience import SimpleCircuitBreaker

log = get_logger(__name__)


class OpenAITTS(TTSService):
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            log.warning("openai_api_key_missing")
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_tts_model
        self._format = settings.openai_tts_format
        self._breaker = SimpleCircuitBreaker(fail_threshold=3, open_sec=20.0)

    async def stream_synthesize(
        self,
        text_chunks: AsyncIterator[str],
        *,
        voice_id: str,
        locale: str = "ru",
        voice_tts_params: dict | None = None,
    ) -> AsyncIterator[bytes]:
        flush_marks = {".", "!", "?", "…", "\n"}
        buf: list[str] = []

        async for delta in text_chunks:
            if not delta:
                continue
            buf.append(delta)
            if any(delta.rstrip().endswith(m) for m in flush_marks) and len("".join(buf)) > 12:
                async for chunk in self._synth_one("".join(buf), voice_id, voice_tts_params):
                    yield chunk
                buf.clear()

        tail = "".join(buf).strip()
        if tail:
            async for chunk in self._synth_one(tail, voice_id, voice_tts_params):
                yield chunk

    async def _synth_one(
        self,
        text: str,
        voice_id: str,
        voice_tts_params: dict | None,
    ) -> AsyncIterator[bytes]:
        rt = get_bot_runtime_payload_sync()
        v = voice_tts_params or {}
        try:
            speed = float(v.get("speed", rt.get("openai_tts_speed", 1.0)))
        except (TypeError, ValueError):
            speed = 1.0
        speed = max(0.25, min(4.0, speed))
        if not await self._breaker.before_call():
            log.warning("openai_tts_circuit_open")
            return
        try:
            async with self._client.audio.speech.with_streaming_response.create(
                model=self._model,
                voice=voice_id,  # alloy / echo / fable / onyx / nova / shimmer
                input=text,
                response_format=self._format,
                speed=speed,
            ) as resp:
                had_chunks = False
                async for chunk in resp.iter_bytes(chunk_size=4096):
                    if chunk:
                        had_chunks = True
                        yield chunk
                if had_chunks:
                    await self._breaker.on_success()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            await self._breaker.on_failure()
            log.warning("openai_tts_error", error=str(exc))
            return

    async def aclose(self) -> None:
        return None
