"""Deepgram pre-recorded `/v1/listen` с diarize + utterances → DiarizedTurn.

Маппинг спикеров (эвристика под моно-звонок): speaker **0** в ответе Deepgram
часто совпадает с первым говорящим (оператор), **1** — клиент. При одном
спикере все реплики получают `unknown`, чтобы не вводить в заблуждение.
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

_LISTEN = "https://api.deepgram.com/v1/listen"


def _map_speaker(dg_speaker: int | None) -> str:
    """Deepgram integer speaker → роль (0 часто оператор, 1 — клиент)."""
    if dg_speaker is None:
        return "unknown"
    i = int(dg_speaker)
    if i == 0:
        return "agent"
    if i == 1:
        return "customer"
    return "unknown"


class DeepgramDiarizationService(DiarizationService):
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
        s = get_settings()
        key = (s.deepgram_api_key or "").strip()
        if not key or not audio_bytes:
            log.warning(
                "deepgram_diarization_fallback",
                reason="no_key_or_audio" if not audio_bytes else "no_key",
            )
            return self._from_text_only(full_text)

        params = {
            "model": s.deepgram_model,
            "language": locale,
            "diarize": "true",
            "utterances": "true",
            "punctuate": "true",
        }
        headers = {"Authorization": f"Token {key}", "Content-Type": audio_mime_type}
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(_LISTEN, params=params, content=audio_bytes, headers=headers)
                r.raise_for_status()
                data = r.json()
        except Exception as exc:  # noqa: BLE001
            log.error("deepgram_diarization_http_error", error=str(exc))
            return self._from_text_only(full_text)

        utterances = _extract_utterances(data)
        if not utterances:
            return self._from_text_only(full_text)

        speakers_present = {u.get("speaker") for u in utterances if "speaker" in u}
        if len(speakers_present) <= 1:
            text = " ".join(str(u.get("transcript", "")).strip() for u in utterances).strip()
            return [DiarizedTurn(speaker="unknown", text=text or (full_text or "").strip())]

        out: list[DiarizedTurn] = []
        for u in utterances:
            tx = str(u.get("transcript", "")).strip()
            if not tx:
                continue
            sp = u.get("speaker")
            try:
                sp_i = int(sp) if sp is not None else None
            except (TypeError, ValueError):
                sp_i = None
            role = _map_speaker(sp_i)
            start = u.get("start")
            end = u.get("end")
            try:
                s_sec = float(start) if start is not None else None
            except (TypeError, ValueError):
                s_sec = None
            try:
                e_sec = float(end) if end is not None else None
            except (TypeError, ValueError):
                e_sec = None
            out.append(
                DiarizedTurn(speaker=role, text=tx, start_sec=s_sec, end_sec=e_sec)
            )
        return out or self._from_text_only(full_text)

    @staticmethod
    def _from_text_only(full_text: str) -> list[DiarizedTurn]:
        t = (full_text or "").strip()
        if not t:
            return []
        return [DiarizedTurn(speaker="unknown", text=t)]


def _extract_utterances(data: dict[str, Any]) -> list[dict[str, Any]]:
    results = data.get("results") or {}
    raw = results.get("utterances")
    if isinstance(raw, list):
        return [u for u in raw if isinstance(u, dict)]
    return []
