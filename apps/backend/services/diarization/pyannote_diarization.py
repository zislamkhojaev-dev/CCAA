"""Внешний pyannote / GPU-воркер по HTTP (очередь не в этом сервисе).

Тяжёлые модели pyannote.audio обычно крутят в **отдельном** контейнере/воркере
с GPU: бэкенд VoiceAgent только кладёт файл в очередь или шлёт multipart POST
и получает готовые метки спикеров. Так проще масштабировать и не тащить CUDA
в основной API-сборке.

Контракт воркера (`PYANNOTE_WORKER_URL`, POST multipart):

  - поле `file` — то же аудио, что и в аналитике (wav/mp3);
  - ответ JSON: `{"turns":[{"speaker":"customer"|"agent"|"unknown","text":"...",
    "start_sec":0.0,"end_sec":1.0}]}`.

Если URL не задан или ответ невалиден — возвращаем один блок `unknown` по `full_text`,
чтобы пайплайн аналитики не ломался.
"""

from __future__ import annotations

from typing import Any, Sequence

import httpx

from apps.backend.config import get_settings
from apps.backend.models.schemas import DiarizedTurn
from apps.backend.services.diarization.interfaces import DiarizationService
from apps.backend.services.diarization.types import TranscriptSegment
from apps.backend.utils.logging import get_logger

log = get_logger(__name__)


class PyannoteDiarizationService(DiarizationService):
    async def diarize(
        self,
        segments: Sequence[TranscriptSegment],
        *,
        full_text: str,
        locale: str = "ru",
        audio_bytes: bytes | None = None,
        audio_filename: str = "audio.wav",
        audio_mime_type: str = "application/octet-stream",
    ) -> list[DiarizedTurn]:
        url = (get_settings().pyannote_worker_url or "").strip()
        t = (full_text or "").strip()
        if not url or not audio_bytes:
            log.info(
                "pyannote_diarization_skipped",
                has_url=bool(url),
                has_audio=bool(audio_bytes),
            )
            return [DiarizedTurn(speaker="unknown", text=t)] if t else []

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                files = {"file": (audio_filename or "audio.wav", audio_bytes, audio_mime_type)}
                r = await client.post(url, files=files)
                r.raise_for_status()
                data = r.json()
        except Exception as exc:  # noqa: BLE001
            log.warning("pyannote_worker_error", error=str(exc))
            return [DiarizedTurn(speaker="unknown", text=t)] if t else []

        turns = data.get("turns") if isinstance(data, dict) else None
        if not isinstance(turns, list):
            return [DiarizedTurn(speaker="unknown", text=t)] if t else []

        out: list[DiarizedTurn] = []
        for item in turns:
            if not isinstance(item, dict):
                continue
            sp = str(item.get("speaker") or "unknown")
            if sp not in ("customer", "agent", "unknown"):
                sp = "unknown"
            tx = str(item.get("text", "")).strip()
            if not tx:
                continue
            s_sec = _f(item.get("start_sec"))
            e_sec = _f(item.get("end_sec"))
            out.append(DiarizedTurn(speaker=sp, text=tx, start_sec=s_sec, end_sec=e_sec))
        return out or ([DiarizedTurn(speaker="unknown", text=t)] if t else [])


def _f(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
