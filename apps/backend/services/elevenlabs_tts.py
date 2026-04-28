"""ElevenLabs streaming TTS client.

We post text deltas to the streaming endpoint and pipe back audio
chunks as soon as they arrive. The Voice Engine forwards them straight
to the user — first-byte latency, not last-byte, is what matters.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import httpx

from apps.backend.config import get_settings
from apps.backend.core.bot_runtime import get_bot_runtime_payload_sync
from apps.backend.services.interfaces import TTSService
from apps.backend.utils.logging import get_logger
from apps.backend.utils.resilience import SimpleCircuitBreaker

log = get_logger(__name__)

_API = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"


class ElevenLabsTTS(TTSService):
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.elevenlabs_api_key:
            log.warning("elevenlabs_api_key_missing")
        self._api_key = settings.elevenlabs_api_key
        self._model = settings.elevenlabs_model
        self._client = httpx.AsyncClient(timeout=30.0)
        self._breaker = SimpleCircuitBreaker(fail_threshold=3, open_sec=20.0)

    async def stream_synthesize(
        self,
        text_chunks: AsyncIterator[str],
        *,
        voice_id: str,
        locale: str = "ru",
        voice_tts_params: dict | None = None,
    ) -> AsyncIterator[bytes]:
        # Aggregate small deltas into sentence-sized payloads to reduce
        # round trips while still keeping perceived latency low.
        sentence_buf: list[str] = []
        flush_marks = {".", "!", "?", "…"}

        async def iter_sentences() -> AsyncIterator[str]:
            async for delta in text_chunks:
                if not delta:
                    continue
                sentence_buf.append(delta)
                if any(delta.rstrip().endswith(m) for m in flush_marks):
                    text = "".join(sentence_buf).strip()
                    sentence_buf.clear()
                    if text:
                        yield text
            tail = "".join(sentence_buf).strip()
            if tail:
                yield tail

        async for sentence in iter_sentences():
            async for chunk in self._synth_one(sentence, voice_id, voice_tts_params):
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
            stability = float(v.get("stability", rt.get("elevenlabs_stability", 0.5)))
        except (TypeError, ValueError):
            stability = 0.5
        try:
            similarity = float(
                v.get("similarity_boost", rt.get("elevenlabs_similarity_boost", 0.75))
            )
        except (TypeError, ValueError):
            similarity = 0.75
        stability = max(0.0, min(1.0, stability))
        similarity = max(0.0, min(1.0, similarity))
        url = _API.format(voice_id=voice_id)
        headers = {
            "xi-api-key": self._api_key,
            "accept": "audio/mpeg",
            "content-type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": self._model,
            "voice_settings": {"stability": stability, "similarity_boost": similarity},
        }
        if not await self._breaker.before_call():
            log.warning("elevenlabs_tts_circuit_open")
            return
        try:
            for attempt in range(2):
                had_chunks = False
                async with self._client.stream(
                    "POST", url, headers=headers, json=payload
                ) as resp:
                    if resp.status_code >= 500 and attempt == 0:
                        # Retry once on transient provider failures before yielding audio.
                        continue
                    if resp.status_code >= 400:
                        body = await resp.aread()
                        log.error(
                            "tts_http_error",
                            status=resp.status_code,
                            body=body[:200].decode("utf-8", "replace"),
                        )
                        await self._breaker.on_failure()
                        return
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            had_chunks = True
                            yield chunk
                if had_chunks:
                    await self._breaker.on_success()
                return
        except asyncio.CancelledError:
            raise
        except httpx.HTTPError as exc:
            await self._breaker.on_failure()
            log.warning("tts_stream_interrupted", error=str(exc))
            return

    async def aclose(self) -> None:
        await self._client.aclose()
