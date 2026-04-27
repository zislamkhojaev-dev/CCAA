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
            payload = await self._synth_one(sentence, voice_id, voice_tts_params)
            if payload:
                yield payload

    async def _synth_one(self, text: str, voice_id: str, voice_tts_params: dict | None) -> bytes:
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
        try:
            out = bytearray()
            async with self._client.stream(
                "POST", url, headers=headers, json=payload
            ) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    log.error(
                        "tts_http_error",
                        status=resp.status_code,
                        body=body[:200].decode("utf-8", "replace"),
                    )
                    return b""
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        out.extend(chunk)
            return bytes(out)
        except asyncio.CancelledError:
            raise
        except httpx.HTTPError as exc:
            log.warning("tts_stream_interrupted", error=str(exc))
            return b""

    async def aclose(self) -> None:
        await self._client.aclose()
