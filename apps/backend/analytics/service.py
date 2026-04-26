"""Pipeline: transcribe (Whisper) → diarize (`DiarizationService`) → score (LLM rubric).

* **Transcription** — OpenAI `whisper-1` with `verbose_json` + segment timestamps.
* **Diarization** — `apps.backend.services.diarization.get_diarization()` (по умолчанию LLM;
  `DIARIZATION_PROVIDER=deepgram|pyannote` — заглушки под реализацию без смены HTTP/UI).
* **Оценка качества** — критерии и веса из PostgreSQL; LLM-рубрика; взвешенный итог.
"""

from __future__ import annotations

import json
import mimetypes
import re
from dataclasses import dataclass
from typing import Any, Sequence
from uuid import UUID

from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.backend.config import get_settings
from apps.backend.models.entities import AnalyticsCriterion, AnalyticsCriteriaSet
from apps.backend.models.schemas import (
    CriterionScoreOut,
    DiarizedTurn,
    SpeechAnalysisResult,
)
from apps.backend.services.diarization import TranscriptSegment, get_diarization
from apps.backend.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class _WhisperSegment:
    start: float
    end: float
    text: str


class SpeechAnalyticsService:
    def __init__(self) -> None:
        s = get_settings()
        self._client = AsyncOpenAI(api_key=s.openai_api_key)
        self._stt_model = s.openai_stt_model

    async def analyze_recording(
        self,
        *,
        audio_bytes: bytes,
        filename: str,
        session: AsyncSession,
        criteria_set_id: UUID | None = None,
        locale: str = "ru",
    ) -> SpeechAnalysisResult:
        segments, full_text = await self._transcribe_segments(audio_bytes, filename)
        diar = get_diarization()
        seg_dto = [
            TranscriptSegment(start_sec=s.start, end_sec=s.end, text=s.text)
            for s in segments
        ]
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        diarization = await diar.diarize(
            seg_dto,
            full_text=full_text,
            locale=locale,
            audio_bytes=audio_bytes,
            audio_filename=filename,
            audio_mime_type=mime,
        )
        crit_rows, sset = await self._load_criteria(session, criteria_set_id)
        scores, total, notes = await self._llm_score(
            full_text=full_text,
            diarization=diarization,
            criteria=crit_rows,
            locale=locale,
            set_name=sset.name if sset else "default",
        )
        return SpeechAnalysisResult(
            transcript=full_text,
            diarization=diarization,
            criterion_scores=scores,
            total_weighted=total,
            notes=notes,
        )

    async def _transcribe_segments(
        self, audio_bytes: bytes, filename: str
    ) -> tuple[list[_WhisperSegment], str]:
        file_tuple = (filename or "call.wav", audio_bytes, "application/octet-stream")
        kwargs: dict = {
            "model": self._stt_model,
            "file": file_tuple,
            "response_format": "verbose_json",
        }
        try:
            resp = await self._client.audio.transcriptions.create(
                **kwargs,
                timestamp_granularities=["segment"],
            )
        except Exception:  # noqa: BLE001 — старые версии SDK без granularities
            resp = await self._client.audio.transcriptions.create(**kwargs)

        if hasattr(resp, "model_dump"):
            data = resp.model_dump()
        else:
            data = dict(resp) if isinstance(resp, dict) else {}
        full = (getattr(resp, "text", None) or data.get("text") or "").strip()
        segs_raw = data.get("segments") or getattr(resp, "segments", None) or []
        segments: list[_WhisperSegment] = []
        for s in segs_raw:
            if hasattr(s, "model_dump"):
                s = s.model_dump()
            if not isinstance(s, dict):
                continue
            try:
                segments.append(
                    _WhisperSegment(
                        start=float(s.get("start", 0)),
                        end=float(s.get("end", 0)),
                        text=(s.get("text") or "").strip(),
                    )
                )
            except (TypeError, ValueError):
                continue
        if not full and segments:
            full = " ".join(s.text for s in segments if s.text)
        if not segments and full:
            segments.append(_WhisperSegment(start=0.0, end=0.0, text=full))
        return segments, full

    async def _load_criteria(
        self, session: AsyncSession, criteria_set_id: UUID | None
    ) -> tuple[list[AnalyticsCriterion], AnalyticsCriteriaSet | None]:
        if criteria_set_id:
            sset = await session.get(AnalyticsCriteriaSet, criteria_set_id)
            if sset is None:
                return [], None
            q = await session.execute(
                select(AnalyticsCriterion)
                .where(AnalyticsCriterion.set_id == criteria_set_id)
                .order_by(AnalyticsCriterion.code)
            )
            return list(q.scalars().all()), sset
        q = await session.execute(
            select(AnalyticsCriteriaSet).where(AnalyticsCriteriaSet.is_default.is_(True))
        )
        sset = q.scalar_one_or_none()
        if sset is None:
            q2 = await session.execute(select(AnalyticsCriteriaSet).limit(1))
            sset = q2.scalar_one_or_none()
        if sset is None:
            return [], None
        q3 = await session.execute(
            select(AnalyticsCriterion)
            .where(AnalyticsCriterion.set_id == sset.id)
            .order_by(AnalyticsCriterion.code)
        )
        return list(q3.scalars().all()), sset

    async def _llm_score(
        self,
        *,
        full_text: str,
        diarization: list[DiarizedTurn],
        criteria: Sequence[AnalyticsCriterion],
        locale: str,
        set_name: str,
    ) -> tuple[list[CriterionScoreOut], float, str]:
        if not criteria:
            return [], 0.0, "Нет критериев в наборе — добавьте их в разделе «Речевая аналитика»."

        spec = [
            {
                "code": c.code,
                "title": c.title,
                "description": c.description,
                "weight": c.weight,
                "max_score": c.max_score,
                "rubric": c.rubric or c.description,
            }
            for c in criteria
        ]
        dialog = "\n".join(f"{t.speaker}: {t.text}" for t in diarization)
        sys = (
            f"You are a QA evaluator for contact-center calls. Criteria set: {set_name}. "
            "Score each criterion from 0 up to its max_score using the rubric. "
            'Return ONLY JSON: {"scores":[{"code":"...","score":number,"comment":"..."}],'
            '"summary":"..."}. Be strict but fair.'
        )
        user = (
            f"Locale context: {locale}\n\nFULL TRANSCRIPT:\n{full_text}\n\n"
            f"DIALOG (diarized):\n{dialog}\n\nCRITERIA:\n{json.dumps(spec, ensure_ascii=False)}"
        )
        raw = await self._chat_json(
            [{"role": "system", "content": sys}, {"role": "user", "content": user}],
            max_tokens=1500,
        )
        scores_raw = raw.get("scores") if isinstance(raw, dict) else None
        summary = (raw.get("summary") if isinstance(raw, dict) else "") or ""
        by_code: dict[str, dict[str, Any]] = {}
        if isinstance(scores_raw, list):
            for item in scores_raw:
                if isinstance(item, dict) and item.get("code"):
                    by_code[str(item["code"])] = item

        out: list[CriterionScoreOut] = []
        total_w = 0.0
        sum_weighted = 0.0
        for c in criteria:
            item = by_code.get(c.code, {})
            try:
                sc = float(item.get("score", 0))
            except (TypeError, ValueError):
                sc = 0.0
            sc = max(0.0, min(float(c.max_score), sc))
            w = float(c.weight)
            weighted = w * (sc / float(c.max_score)) if c.max_score else 0.0
            sum_weighted += weighted
            total_w += w
            out.append(
                CriterionScoreOut(
                    code=c.code,
                    title=c.title,
                    score=sc,
                    max_score=float(c.max_score),
                    weight=w,
                    weighted=weighted,
                    comment=str(item.get("comment", "")),
                )
            )
        denom = total_w if total_w > 0 else 1.0
        return out, sum_weighted / denom, summary

    async def _chat_json(self, messages: list[dict], *, max_tokens: int) -> dict:
        resp = await self._client.chat.completions.create(
            model=get_settings().openai_model,
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
