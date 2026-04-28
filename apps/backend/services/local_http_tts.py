"""Локальный TTS через HTTP.

`LOCAL_TTS_URL` — POST JSON вида `{"text":"...","voice_id":"...","locale":"ru"}`.
Ответ: бинарное тело (audio/mpeg или audio/wav) целиком; мы режем на чанки для API.
"""

from __future__ import annotations

from typing import AsyncIterator

import httpx

from apps.backend.config import get_settings
from apps.backend.services.interfaces import TTSService
from apps.backend.utils.logging import get_logger
from apps.backend.utils.resilience import SimpleCircuitBreaker, retry_async

log = get_logger(__name__)


class LocalHttpTTS(TTSService):
    def __init__(self) -> None:
        s = get_settings()
        self._url = (s.local_tts_url or "").strip()
        self._client = httpx.AsyncClient(timeout=120.0)
        self._breaker = SimpleCircuitBreaker(fail_threshold=3, open_sec=15.0)

    async def stream_synthesize(
        self,
        text_chunks: AsyncIterator[str],
        *,
        voice_id: str,
        locale: str = "ru",
        voice_tts_params: dict | None = None,
    ) -> AsyncIterator[bytes]:
        if not self._url:
            log.error("local_tts_url_missing")
            return
        buf: list[str] = []
        async for delta in text_chunks:
            if delta:
                buf.append(delta)
        text = "".join(buf).strip()
        if not text:
            return
        try:
            if not await self._breaker.before_call():
                log.warning("local_tts_circuit_open")
                return
            body: dict = {"text": text, "voice_id": voice_id, "locale": locale}
            if voice_tts_params:
                body["tts_params"] = voice_tts_params
            r = await retry_async(
                lambda: self._client.post(self._url, json=body),
                attempts=2,
            )
            r.raise_for_status()
            data = r.content
            step = 4096
            for i in range(0, len(data), step):
                yield data[i : i + step]
            await self._breaker.on_success()
        except Exception as exc:  # noqa: BLE001
            await self._breaker.on_failure()
            log.warning("local_tts_error", error=str(exc))

    async def aclose(self) -> None:
        await self._client.aclose()
