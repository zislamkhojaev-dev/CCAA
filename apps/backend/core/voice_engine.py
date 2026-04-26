"""Voice Engine — duplex pipeline + transcript + escalation."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from typing import AsyncIterator

from apps.backend.config import get_settings
from apps.backend.core.bot_runtime import get_bot_runtime_payload_sync
from apps.backend.core.conversation_recorder import ConversationRecorder
from apps.backend.core.orchestrator import Orchestrator
from apps.backend.core.prompts import fallback_message
from apps.backend.models.schemas import ChatMessage
from apps.backend.services import get_stt, get_tts
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
        self._last_accepted_utterance = ""
        self._last_accepted_at = 0.0
        self._last_fallback_at = 0.0

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
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            log.info("voice_turn_cancelled_barge_in")

    async def run(
        self,
        *,
        audio_in: AsyncIterator[bytes],
        audio_out: "asyncio.Queue[bytes | None]",
        locale: str = "ru",
        voice_id: str | None = None,
        sample_rate: int = 16_000,
    ) -> None:
        voice_id = voice_id or self._default_voice_id()

        utterance_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def consume_stt() -> None:
            try:
                async for event in self._stt.stream_transcribe(
                    audio_in, locale=locale, sample_rate=sample_rate
                ):
                    if event.is_final and event.text.strip():
                        text = event.text.strip()
                        if not _is_meaningful_utterance(text):
                            log.info("stt_ignored_noise", text=text)
                            continue
                        now = time.monotonic()
                        if (
                            text == self._last_accepted_utterance
                            and now - self._last_accepted_at < 1.5
                        ):
                            log.info("stt_ignored_duplicate", text=text)
                            continue
                        self._last_accepted_utterance = text
                        self._last_accepted_at = now
                        log.info("stt_final", text=text, conf=event.confidence)
                        await utterance_queue.put(text)
            finally:
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

        stt_task = asyncio.create_task(consume_stt(), name="ve-stt")
        resp_task = asyncio.create_task(respond_loop(), name="ve-respond")
        try:
            await asyncio.gather(stt_task, resp_task)
        except asyncio.CancelledError:
            stt_task.cancel()
            resp_task.cancel()
            await self.interrupt()
            raise

    async def _handle_turn(
        self,
        *,
        text: str,
        locale: str,
        voice_id: str,
        audio_out: "asyncio.Queue[bytes | None]",
    ) -> None:
        try:
            intent, sources = await self._orchestrator.preview_route(text, locale=locale)
            if self._recorder:
                await self._recorder.add_turn("customer", text)

            if intent.requires_human:
                await self._emit_escalated_reply(
                    text=text,
                    locale=locale,
                    voice_id=voice_id,
                    audio_out=audio_out,
                    assistant_text=Orchestrator.human_handoff_message(locale),
                    reason=f"intent:{intent.intent}",
                    extra_meta={"intent": intent.intent, "escalation": True},
                )
                return

            if not sources:
                now = time.monotonic()
                if now - self._last_fallback_at < 2.0:
                    log.info("fallback_skipped_cooldown", text=text)
                    return
                self._last_fallback_at = now
                fb = fallback_message(locale)
                if self._recorder:
                    await self._recorder.add_turn("assistant", fb, extra={"fallback": True})
                packet = await self._collect_tts_packet(
                    _once(fb),
                    voice_id=voice_id,
                    locale=locale,
                )
                if packet:
                    await audio_out.put(packet)
                self._history.append(ChatMessage(role="user", content=text))
                self._history.append(ChatMessage(role="assistant", content=fb))
                self._trim_history()
                return

            with timed_stage("turn", text_len=len(text)) as ctx:
                text_q: asyncio.Queue[str | None] = asyncio.Queue()

                async def llm_producer() -> None:
                    try:
                        with timed_stage("llm"):
                            async for delta in self._orchestrator.stream_answer_after_route(
                                text,
                                intent,
                                sources,
                                locale=locale,
                                history=self._history,
                            ):
                                await text_q.put(delta)
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
                    with timed_stage("tts"):
                        packet = await self._collect_tts_packet(
                            text_iter(),
                            voice_id=voice_id,
                            locale=locale,
                        )
                        if packet:
                            await audio_out.put(packet)

                await asyncio.gather(llm_producer(), tts_consumer())

                answer = (ctx.get("answer") or "").strip()
                if self._recorder and answer:
                    await self._recorder.add_turn("assistant", answer)
                self._history.append(ChatMessage(role="user", content=text))
                if answer:
                    self._history.append(ChatMessage(role="assistant", content=answer))
                self._trim_history()
        except asyncio.CancelledError:
            raise

    async def _emit_escalated_reply(
        self,
        *,
        text: str,
        locale: str,
        voice_id: str,
        audio_out: "asyncio.Queue[bytes | None]",
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
        packet = await self._collect_tts_packet(
            _once(assistant_text),
            voice_id=voice_id,
            locale=locale,
        )
        if packet:
            await audio_out.put(packet)
        self._history.append(ChatMessage(role="user", content=text))
        self._history.append(ChatMessage(role="assistant", content=assistant_text))
        self._trim_history()

    def _trim_history(self) -> None:
        rt = get_bot_runtime_payload_sync()
        n = max(2, int(rt.get("history_max_messages", 24)))
        self._history = self._history[-n:]

    async def _collect_tts_packet(
        self,
        text_chunks: AsyncIterator[str],
        *,
        voice_id: str,
        locale: str,
    ) -> bytes:
        """Aggregate TTS bytes into one websocket payload.

        Frontend currently treats each websocket binary message as a standalone
        audio blob, so sending partial codec chunks causes choppy playback.
        """
        out = bytearray()
        async for chunk in self._tts.stream_synthesize(
            text_chunks,
            voice_id=voice_id,
            locale=locale,
            voice_tts_params=self._voice_tts_params,
        ):
            if chunk:
                out.extend(chunk)
        return bytes(out)


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
