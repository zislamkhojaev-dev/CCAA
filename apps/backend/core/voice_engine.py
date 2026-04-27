"""Voice Engine — duplex pipeline + transcript + escalation."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import AsyncIterator

from apps.backend.config import get_settings
from apps.backend.core.bot_runtime import get_bot_runtime_payload_sync
from apps.backend.core.case_manager import CaseManager
from apps.backend.core.conversation_recorder import ConversationRecorder
from apps.backend.core.orchestrator import Orchestrator
from apps.backend.core.vad_audio_gate import pcm_vad_gate_stream
from apps.backend.core.prompts import fallback_message, fallback_message_soft, silence_nudge_message
from apps.backend.models.schemas import ChatMessage
from apps.backend.services import get_stt, get_tts
from apps.backend.utils.slo_metrics import record_latency
from apps.backend.utils.logging import get_logger, timed_stage

log = get_logger(__name__)


async def _once(text: str) -> AsyncIterator[str]:
    yield text


class VoiceEngine:
    def __init__(
        self,
        orchestrator: Orchestrator,
        *,
        conversation_recorder: ConversationRecorder | None = None,
        on_escalation: Callable[[dict], Awaitable[None]] | None = None,
        on_state_change: Callable[[dict], Awaitable[None]] | None = None,
        voice_tts_params: dict | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._stt = get_stt()
        self._tts = get_tts()
        self._settings = get_settings()
        self._voice_tts_params: dict = dict(voice_tts_params or {})
        self._history: list[ChatMessage] = []
        self._turn_lock = asyncio.Lock()
        self._active_turn: asyncio.Task | None = None
        self._recorder = conversation_recorder
        self._on_escalation = on_escalation
        self._on_state_change = on_state_change
        self._case_manager = CaseManager()
        self._last_accepted_utterance = ""
        self._last_accepted_at = 0.0
        self._last_fallback_at = 0.0
        self._fallback_count = 0
        self._last_silence_nudge_at = 0.0
        self._state = CallState.LISTENING
        self._segment_lock = asyncio.Lock()
        self._segment_text_by_index: dict[int, str] = {}
        self._played_segment_indices: set[int] = set()
        self._allowed_transitions: dict[CallState, set[CallState]] = {
            CallState.LISTENING: {CallState.THINKING, CallState.NUDGING, CallState.ENDED},
            CallState.THINKING: {CallState.SPEAKING, CallState.ESCALATED, CallState.INTERRUPTED, CallState.LISTENING},
            CallState.SPEAKING: {CallState.LISTENING, CallState.INTERRUPTED, CallState.ESCALATED},
            CallState.NUDGING: {CallState.LISTENING, CallState.INTERRUPTED},
            CallState.INTERRUPTED: {CallState.LISTENING, CallState.THINKING},
            CallState.ESCALATED: {CallState.LISTENING, CallState.ENDED},
            CallState.ENDED: set(),
        }

    def _default_voice_id(self) -> str:
        provider = self._settings.tts_provider
        if provider == "elevenlabs":
            return self._settings.elevenlabs_voice_id
        if provider == "openai":
            return self._settings.openai_tts_voice
        return "default"

    async def interrupt(self) -> None:
        async with self._turn_lock:
            task = self._active_turn
            self._active_turn = None
        if task is None or task.done():
            return
        await self._set_state(CallState.INTERRUPTED, reason="barge_in")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            log.info("voice_turn_cancelled_barge_in")
        await self._set_state(CallState.LISTENING)

    async def run(
        self,
        *,
        audio_in: AsyncIterator[bytes],
        audio_out: "asyncio.Queue[bytes | dict | None]",
        locale: str = "ru",
        voice_id: str | None = None,
        sample_rate: int = 16_000,
    ) -> None:
        voice_id = voice_id or self._default_voice_id()
        await self._set_state(CallState.LISTENING)
        self._last_accepted_at = time.monotonic()
        stop_event = asyncio.Event()

        utterance_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def consume_stt() -> None:
            """Buffer finals with a debounce window so the user can finish a thought.

            Streaming STT often emits several finals in quick succession; without a
            short wait we may start the LLM on the first fragment.
            """
            stt_state: dict[str, object] = {"seq": 0, "pending": None, "task": None}

            async def _commit_if_ready(text: str, conf: float) -> None:
                text = text.strip()
                if not _is_meaningful_utterance(text):
                    log.info("stt_ignored_noise", text=text)
                    return
                now = time.monotonic()
                if (
                    text == self._last_accepted_utterance
                    and now - self._last_accepted_at < 1.5
                ):
                    log.info("stt_ignored_duplicate", text=text)
                    return
                self._last_accepted_utterance = text
                self._last_accepted_at = now
                log.info("stt_final", text=text, conf=conf)
                await utterance_queue.put(text)

            async def _schedule_final(text: str, conf: float) -> None:
                rt = get_bot_runtime_payload_sync()
                ms = max(0, int(rt.get("stt_final_debounce_ms", 450)))
                stt_state["pending"] = (text, conf)
                stt_state["seq"] = int(stt_state["seq"]) + 1
                my_seq = int(stt_state["seq"])
                prev = stt_state.get("task")
                if isinstance(prev, asyncio.Task) and not prev.done():
                    prev.cancel()
                if ms <= 0:
                    stt_state["pending"] = None
                    await _commit_if_ready(text, conf)
                    return

                async def _after_quiet() -> None:
                    try:
                        await asyncio.sleep(ms / 1000.0)
                    except asyncio.CancelledError:
                        return
                    if int(stt_state["seq"]) != my_seq:
                        return
                    pending = stt_state.get("pending")
                    if not isinstance(pending, tuple):
                        return
                    t, c = pending[0], float(pending[1])
                    stt_state["pending"] = None
                    await _commit_if_ready(t, c)

                stt_state["task"] = asyncio.create_task(_after_quiet(), name="stt-final-debounce")

            try:
                async for event in self._stt.stream_transcribe(
                    self._vad_gate_audio(audio_in, sample_rate=sample_rate),
                    locale=locale,
                    sample_rate=sample_rate,
                ):
                    if event.is_final and event.text.strip():
                        await _schedule_final(
                            event.text.strip(),
                            float(event.confidence or 0.0),
                        )
            finally:
                prev = stt_state.get("task")
                if isinstance(prev, asyncio.Task) and not prev.done():
                    prev.cancel()
                    try:
                        await prev
                    except asyncio.CancelledError:
                        pass
                pending = stt_state.get("pending")
                stt_state["pending"] = None
                stt_state["seq"] = int(stt_state["seq"]) + 1
                if isinstance(pending, tuple):
                    t, c = pending[0], float(pending[1])
                    if _is_meaningful_utterance(t):
                        now = time.monotonic()
                        if not (
                            t == self._last_accepted_utterance
                            and now - self._last_accepted_at < 1.5
                        ):
                            self._last_accepted_utterance = t
                            self._last_accepted_at = now
                            log.info("stt_final_flush", text=t, conf=c)
                            await utterance_queue.put(t)
                await utterance_queue.put(None)

        async def respond_loop() -> None:
            while True:
                text = await utterance_queue.get()
                if text is None:
                    async with self._turn_lock:
                        tail = self._active_turn
                    if tail and not tail.done():
                        try:
                            await tail
                        except asyncio.CancelledError:
                            pass
                    break
                await self.interrupt()
                await self._set_state(CallState.THINKING)
                async with self._turn_lock:
                    self._active_turn = asyncio.create_task(
                        self._handle_turn(
                            text=text,
                            locale=locale,
                            voice_id=voice_id,
                            audio_out=audio_out,
                        ),
                        name=f"ve-turn-{text[:20]!r}",
                    )
            await audio_out.put(None)
            stop_event.set()
            await self._set_state(CallState.ENDED)

        async def silence_nudge_loop() -> None:
            while not stop_event.is_set():
                await asyncio.sleep(1.0)
                rt = get_bot_runtime_payload_sync()
                if not bool(rt.get("silence_nudge_enabled", True)):
                    continue
                now = time.monotonic()
                timeout_sec = max(3.0, float(rt.get("silence_timeout_sec", 12.0)))
                cooldown_sec = max(5.0, float(rt.get("silence_nudge_cooldown_sec", 20.0)))
                if now - self._last_accepted_at < timeout_sec:
                    continue
                if now - self._last_silence_nudge_at < cooldown_sec:
                    continue
                async with self._turn_lock:
                    active = self._active_turn
                if active and not active.done():
                    continue
                await self._set_state(CallState.NUDGING)
                try:
                    nudge = (await self._orchestrator.silence_nudge(locale=locale, history=self._history)).strip()
                except Exception:  # noqa: BLE001
                    nudge = ""
                if not nudge:
                    nudge = silence_nudge_message(locale)
                async for packet in self._stream_tts_packets(_once(nudge), voice_id=voice_id, locale=locale):
                    await audio_out.put(packet)
                if self._recorder:
                    await self._recorder.add_turn("assistant", nudge, extra={"silence_nudge": True})
                self._history.append(ChatMessage(role="assistant", content=nudge))
                self._trim_history()
                self._last_silence_nudge_at = now
                await self._set_state(CallState.LISTENING)

        stt_task = asyncio.create_task(consume_stt(), name="ve-stt")
        resp_task = asyncio.create_task(respond_loop(), name="ve-respond")
        nudge_task = asyncio.create_task(silence_nudge_loop(), name="ve-silence-nudge")
        try:
            await asyncio.gather(stt_task, resp_task, nudge_task)
        finally:
            for t in (stt_task, resp_task, nudge_task):
                t.cancel()
            await self.interrupt()

    async def _handle_turn(
        self,
        *,
        text: str,
        locale: str,
        voice_id: str,
        audio_out: "asyncio.Queue[bytes | dict | None]",
    ) -> None:
        spoken_answer = ""
        t_turn_start = time.perf_counter()
        try:
            intent, sources = await self._orchestrator.preview_route(text, locale=locale)
            if self._recorder:
                await self._recorder.add_turn("customer", text)
            prior_case_state = await self._recorder.get_case_state() if self._recorder else {}
            case_decision = await self._case_manager.evaluate_turn(
                text=text,
                locale=locale,
                intent=intent,
                prior=prior_case_state,
                fallback_count=self._fallback_count,
            )
            if self._recorder:
                await self._recorder.upsert_case_state(case_decision.context.model_dump(mode="json"))

            if case_decision.intent_confirmation:
                confirm = case_decision.intent_confirmation
                if self._recorder:
                    await self._recorder.add_turn(
                        "assistant",
                        confirm,
                        extra={"intent_confirmation": True, "intent": intent.intent},
                    )
                async for packet in self._stream_tts_packets(_once(confirm), voice_id=voice_id, locale=locale):
                    await audio_out.put(packet)
                self._history.append(ChatMessage(role="user", content=text))
                self._history.append(ChatMessage(role="assistant", content=confirm))
                self._trim_history()
                return

            if case_decision.clarification_question:
                question = case_decision.clarification_question
                if self._recorder:
                    await self._recorder.add_turn(
                        "assistant",
                        question,
                        extra={
                            "clarification": True,
                            "intent": intent.intent,
                            "missing_slots": case_decision.context.missing_slots,
                        },
                    )
                async for packet in self._stream_tts_packets(_once(question), voice_id=voice_id, locale=locale):
                    await audio_out.put(packet)
                self._history.append(ChatMessage(role="user", content=text))
                self._history.append(ChatMessage(role="assistant", content=question))
                self._trim_history()
                return

            if case_decision.should_handoff:
                handoff_reason = case_decision.handoff_reason or f"intent:{intent.intent}"
                await self._emit_escalated_reply(
                    text=text,
                    locale=locale,
                    voice_id=voice_id,
                    audio_out=audio_out,
                    assistant_text=Orchestrator.human_handoff_message(locale),
                    reason=handoff_reason,
                    extra_meta={"intent": intent.intent, "escalation": True, "case_manager": True},
                )
                return

            if intent.intent in {"greeting", "thanks", "goodbye"}:
                smalltalk = Orchestrator.smalltalk_message(intent.intent, locale)
                if self._recorder:
                    await self._recorder.add_turn("assistant", smalltalk, extra={"smalltalk": True})
                async for packet in self._stream_tts_packets(
                    _once(smalltalk),
                    voice_id=voice_id,
                    locale=locale,
                ):
                    await audio_out.put(packet)
                self._history.append(ChatMessage(role="user", content=text))
                self._history.append(ChatMessage(role="assistant", content=smalltalk))
                self._fallback_count = 0
                self._trim_history()
                return

            if not sources:
                now = time.monotonic()
                rt = get_bot_runtime_payload_sync()
                fb_cooldown = max(0.0, float(rt.get("fallback_cooldown_sec", 2.0)))
                if now - self._last_fallback_at < fb_cooldown:
                    log.info("fallback_skipped_cooldown", text=text)
                    return
                self._last_fallback_at = now
                if bool(rt.get("low_signal_reply_enabled", True)) and _is_low_signal_text(text):
                    fb = _low_signal_message(locale)
                    extra = {"low_signal": True}
                else:
                    self._fallback_count += 1
                    fb = fallback_message(locale) if self._fallback_count == 1 else fallback_message_soft(locale)
                    extra = {"fallback": True}
                if self._recorder:
                    await self._recorder.add_turn("assistant", fb, extra=extra)
                async for packet in self._stream_tts_packets(
                    _once(fb),
                    voice_id=voice_id,
                    locale=locale,
                ):
                    await audio_out.put(packet)
                self._history.append(ChatMessage(role="user", content=text))
                self._history.append(ChatMessage(role="assistant", content=fb))
                self._trim_history()
                return

            with timed_stage("turn", text_len=len(text)) as ctx:
                text_q: asyncio.Queue[str | None] = asyncio.Queue()
                escalation_reason: str | None = None
                resolved_answer: str = ""

                async def llm_producer() -> None:
                    nonlocal escalation_reason
                    nonlocal resolved_answer
                    try:
                        with timed_stage("llm"):
                            answer, escalation_reason = await self._orchestrator.resolve_after_route(
                                text,
                                intent,
                                sources,
                                locale=locale,
                                history=self._history,
                            )
                            resolved_answer = (answer or "").strip()
                            if resolved_answer and not escalation_reason:
                                rt = get_bot_runtime_payload_sync()
                                batch_chars = max(20, int(rt.get("tts_micro_batch_chars", 48)))
                                for batch in _to_micro_batches(resolved_answer, batch_chars=batch_chars):
                                    await text_q.put(batch)
                    except asyncio.CancelledError:
                        raise
                    finally:
                        await text_q.put(None)

                async def text_iter() -> AsyncIterator[str]:
                    buf: list[str] = []
                    while True:
                        delta = await text_q.get()
                        if delta is None:
                            break
                        buf.append(delta)
                        yield delta
                    ctx["answer"] = "".join(buf)

                async def tts_consumer() -> None:
                    nonlocal spoken_answer
                    with timed_stage("tts"):
                        await self._set_state(CallState.SPEAKING)
                        await self._reset_segment_tracking()
                        spoken_units: list[str] = []
                        seg_idx = 0
                        first_audio_sent_ms: float | None = None
                        async for chunk in text_iter():
                            for unit in _split_tts_units(chunk):
                                await self._register_segment_text(seg_idx, unit)
                                await audio_out.put({"type": "audio_segment_start", "index": seg_idx})
                                async for packet in self._stream_tts_packets(
                                    _once(unit),
                                    voice_id=voice_id,
                                    locale=locale,
                                ):
                                    await audio_out.put(packet)
                                    if first_audio_sent_ms is None:
                                        first_audio_sent_ms = (time.perf_counter() - t_turn_start) * 1000.0
                                await audio_out.put({"type": "audio_segment_end", "index": seg_idx})
                                spoken_units.append(unit)
                                seg_idx += 1
                        spoken_answer = "".join(spoken_units).strip()
                        if first_audio_sent_ms is not None:
                            log.info("slo_first_audio_sent_ms", value=round(first_audio_sent_ms, 2))
                            record_latency("first_audio_sent_ms", first_audio_sent_ms)
                        turn_total_ms = (time.perf_counter() - t_turn_start) * 1000.0
                        log.info(
                            "slo_turn_total_ms",
                            value=round(turn_total_ms, 2),
                            state=self._state.value,
                        )
                        record_latency("turn_total_ms", turn_total_ms)
                        await self._set_state(CallState.LISTENING)

                await asyncio.gather(llm_producer(), tts_consumer())

                answer = (ctx.get("answer") or "").strip()
                if escalation_reason:
                    await self._set_state(CallState.ESCALATED, reason=escalation_reason)
                    await self._emit_escalated_reply(
                        text=text,
                        locale=locale,
                        voice_id=voice_id,
                        audio_out=audio_out,
                        assistant_text=resolved_answer or Orchestrator.human_handoff_message(locale),
                        reason=escalation_reason,
                        extra_meta={
                            "intent": intent.intent,
                            "escalation": True,
                            "escalation_source": "tool",
                        },
                    )
                    return
                if self._recorder and answer:
                    await self._recorder.add_turn("assistant", answer)
                    state = await self._recorder.get_case_state()
                    if state:
                        state["stage"] = "resolved"
                        state["resolution_status"] = "resolved"
                        steps = list(state.get("attempted_steps") or [])
                        steps.append("resolved_with_answer")
                        state["attempted_steps"] = steps[-12:]
                        await self._recorder.upsert_case_state(state)
                self._history.append(ChatMessage(role="user", content=text))
                if answer:
                    self._history.append(ChatMessage(role="assistant", content=answer))
                    self._fallback_count = 0
                self._trim_history()
        except asyncio.CancelledError:
            self._history.append(ChatMessage(role="user", content=text))
            partial = (await self._compose_played_text()).strip()
            if partial:
                self._history.append(ChatMessage(role="assistant", content=partial))
                if self._recorder:
                    await asyncio.shield(
                        self._recorder.add_turn(
                            "assistant",
                            partial,
                            extra={"interrupted": True, "partial": True},
                        )
                    )
            self._trim_history()
            raise

    async def mark_audio_played(self, segment_index: int) -> None:
        async with self._segment_lock:
            self._played_segment_indices.add(int(segment_index))

    async def _emit_escalated_reply(
        self,
        *,
        text: str,
        locale: str,
        voice_id: str,
        audio_out: "asyncio.Queue[bytes | dict | None]",
        assistant_text: str,
        reason: str,
        extra_meta: dict,
    ) -> None:
        if self._recorder:
            await self._recorder.add_turn(
                "assistant", assistant_text, extra=extra_meta
            )
            await self._recorder.mark_escalated(reason)
            if self._on_escalation:
                pkg = await self._recorder.build_escalation_packet(
                    locale=locale, reason=reason
                )
                await self._on_escalation(pkg)
        async for packet in self._stream_tts_packets(
            _once(assistant_text),
            voice_id=voice_id,
            locale=locale,
        ):
            await audio_out.put(packet)
        self._history.append(ChatMessage(role="user", content=text))
        self._history.append(ChatMessage(role="assistant", content=assistant_text))
        self._fallback_count = 0
        self._trim_history()

    async def _set_state(self, state: "CallState", *, reason: str | None = None) -> None:
        if self._state == state and not reason:
            return
        allowed = self._allowed_transitions.get(self._state, set())
        if state not in allowed and self._state != state:
            log.warning(
                "state_transition_forced",
                from_state=self._state.value,
                to_state=state.value,
                reason=reason or "",
            )
        self._state = state
        if self._on_state_change:
            try:
                await self._on_state_change({"state": state.value, "reason": reason or ""})
            except Exception:  # noqa: BLE001
                pass

    async def _reset_segment_tracking(self) -> None:
        async with self._segment_lock:
            self._segment_text_by_index.clear()
            self._played_segment_indices.clear()

    async def _register_segment_text(self, idx: int, text: str) -> None:
        async with self._segment_lock:
            self._segment_text_by_index[int(idx)] = text

    async def _compose_played_text(self) -> str:
        async with self._segment_lock:
            ordered = sorted(self._played_segment_indices)
            parts = [self._segment_text_by_index.get(i, "") for i in ordered]
        return "".join(p for p in parts if p).strip()

    def _trim_history(self) -> None:
        rt = get_bot_runtime_payload_sync()
        n = max(2, int(rt.get("history_max_messages", 24)))
        self._history = self._history[-n:]

    async def _stream_tts_packets(
        self,
        text_chunks: AsyncIterator[str],
        *,
        voice_id: str,
        locale: str,
    ) -> AsyncIterator[bytes]:
        """Yield synthesized audio packets ready for immediate websocket send."""
        async for chunk in self._tts.stream_synthesize(
            text_chunks,
            voice_id=voice_id,
            locale=locale,
            voice_tts_params=self._voice_tts_params,
        ):
            if chunk:
                yield chunk

    async def _vad_gate_audio(
        self,
        audio_in: AsyncIterator[bytes],
        *,
        sample_rate: int,
    ) -> AsyncIterator[bytes]:
        """Server-side VAD before STT: Silero ONNX (default) or WebRTC VAD."""
        backend = self._settings.voice_vad_backend
        async for chunk in pcm_vad_gate_stream(
            audio_in,
            sample_rate=sample_rate,
            backend=backend,
            runtime_get=get_bot_runtime_payload_sync,
        ):
            yield chunk


_ONLY_PUNCT_OR_EMOJI_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def _is_meaningful_utterance(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 2:
        return False
    # Ignore pure emoji / punctuation / symbols.
    if _ONLY_PUNCT_OR_EMOJI_RE.fullmatch(t):
        return False
    # Need at least one alnum character to avoid noisy transcripts.
    return any(ch.isalnum() for ch in t)


def _split_tts_units(text: str) -> list[str]:
    s = (text or "").strip()
    if not s:
        return []
    parts = re.split(r"(?<=[\.\!\?\u2026])\s+", s)
    units = [p.strip() for p in parts if p.strip()]
    return units or [s]


def _to_micro_batches(text: str, *, batch_chars: int = 48) -> list[str]:
    s = (text or "").strip()
    if not s:
        return []
    out: list[str] = []
    cur = ""
    for token in s.split():
        nxt = (cur + " " + token).strip()
        if cur and len(nxt) > batch_chars and not cur.endswith((".", "!", "?", "…", ",")):
            out.append(cur + " ")
            cur = token
        else:
            cur = nxt
    if cur:
        out.append(cur)
    return out


def _is_low_signal_text(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    hold_words = ("подожди", "погоди", "сек", "секунду", "минут", "алло", "угу", "ага")
    if any(w in t for w in hold_words):
        return True
    actionable = ("карт", "плат", "перевод", "блок", "тариф", "услуг", "компан", "деньг")
    if any(w in t for w in actionable):
        return False
    compact = re.sub(r"[^\w\s]", " ", t).split()
    return len(compact) <= 2 and len(t) <= 14


def _low_signal_message(locale: str) -> str:
    if locale == "uz":
        return "Mayli, kutaman. Qisqacha ayting, qaysi savol bo‘yicha yordam kerak."
    return "Хорошо, я на линии. Коротко подскажите, с каким вопросом помочь?"


class CallState(str, Enum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    NUDGING = "nudging"
    INTERRUPTED = "interrupted"
    ESCALATED = "escalated"
    ENDED = "ended"
