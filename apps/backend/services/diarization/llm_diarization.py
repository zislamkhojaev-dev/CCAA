"""LLM-based diarization (same heuristic as the original analytics prototype)."""

from __future__ import annotations

import json
import re
from typing import Sequence

from openai import AsyncOpenAI

from apps.backend.config import get_settings
from apps.backend.models.schemas import DiarizedTurn
from apps.backend.services.diarization.interfaces import DiarizationService
from apps.backend.services.diarization.types import TranscriptSegment
from apps.backend.utils.logging import get_logger

log = get_logger(__name__)


class LLMDiarizationService(DiarizationService):
    def __init__(self) -> None:
        s = get_settings()
        self._client = AsyncOpenAI(api_key=s.openai_api_key)
        self._model = s.openai_model

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
        text = (full_text or "").strip()
        if not text:
            return []
        lines = [
            f"[{s.start_sec:.1f}-{s.end_sec:.1f}] {s.text}"
            for s in segments[:80]
            if s.text.strip()
        ]
        if not lines:
            lines = [text]
        sys = (
            "You label a mono call-center recording. Each line is a timestamped ASR segment. "
            "Assign each line to speaker: customer (client) or agent (operator). "
            'Return ONLY JSON: {"turns":[{"speaker":"customer"|"agent"|"unknown","text":"..."}]} '
            "Merge adjacent lines from the same speaker into one turn. "
            "Use Russian context clues (Клиент often asks questions; Оператор greets with company name)."
            if locale != "uz"
            else "Use Uzbek/Russian mixed context. Same JSON schema."
        )
        user = "\n".join(lines[:120])
        raw = await self._chat_json(
            [{"role": "system", "content": sys}, {"role": "user", "content": user}],
            max_tokens=2000,
        )
        turns = raw.get("turns") if isinstance(raw, dict) else None
        if not isinstance(turns, list):
            return [DiarizedTurn(speaker="unknown", text=text)]
        out: list[DiarizedTurn] = []
        for t in turns:
            if not isinstance(t, dict):
                continue
            sp = t.get("speaker", "unknown")
            if sp not in ("customer", "agent", "unknown"):
                sp = "unknown"
            tx = str(t.get("text", "")).strip()
            if tx:
                out.append(DiarizedTurn(speaker=sp, text=tx))
        return out or [DiarizedTurn(speaker="unknown", text=text)]

    async def _chat_json(self, messages: list[dict], *, max_tokens: int) -> dict:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.1,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            return json.loads(m.group(0)) if m else {}
