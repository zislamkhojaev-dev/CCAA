"""Case manager for operator-like diagnostics and handoff decisions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from apps.backend.core.bot_runtime import get_bot_runtime_payload_sync
from apps.backend.models.schemas import CaseContext, ChatMessage, IntentResult
from apps.backend.services import get_llm
from apps.backend.utils.logging import get_logger

log = get_logger(__name__)

_SLOT_RULES: dict[str, list[str]] = {
    "block_card": ["card_last4", "incident_time", "channel"],
    "transfer_money": ["source_account", "target_account", "amount", "incident_time"],
    "personal_data": ["customer_id", "field_to_change", "reason"],
    "complaint": ["topic", "incident_time", "details"],
    "human_agent": [],
    "tariffs": ["product_name"],
    "products": ["product_name"],
}

_HANDOFF_TRIGGER_WORDS = (
    "оператор",
    "человек",
    "менеджер",
    "позови",
    "переключи",
    "жалоба",
    "срочно",
)

_OPERATIONAL_INTENTS = {"block_card", "transfer_money", "personal_data", "complaint", "human_agent"}

_CONSULTATIVE_INTENTS = {"tariffs", "products", "office_hours", "branches"}

_SMALLTALK_INTENTS = {"greeting", "thanks", "goodbye"}


def _default_requires_human(intent_name: str) -> bool:
    return intent_name in {"block_card", "transfer_money", "personal_data", "complaint"}


@dataclass(slots=True)
class CaseDecision:
    context: CaseContext
    intent_confirmation: str | None = None
    clarification_question: str | None = None
    should_handoff: bool = False
    handoff_reason: str | None = None


class CaseManager:
    def __init__(self) -> None:
        self._llm = get_llm()

    async def evaluate_turn(
        self,
        *,
        text: str,
        locale: str,
        intent: IntentResult,
        prior: dict | None,
        fallback_count: int,
    ) -> CaseDecision:
        runtime = get_bot_runtime_payload_sync()
        current, intent = self._normalize_prior_and_intent(
            prior, text=text, intent=intent, runtime=runtime
        )
        clarification_limit = max(1, int(runtime.get("case_clarification_max_attempts", 2)))
        updated = self._update_intent_confirmation(
            context=current,
            intent=intent,
            locale=locale,
            confidence_min=float(runtime.get("case_confirm_intent_min_confidence", 0.72)),
        )
        if updated.intent_confirmed is False:
            updated.resolution_status = "in_progress"
            updated.attempted_steps.append("intent_confirmation")
            phrase = await self._intent_confirmation_phrase(
                intent=intent,
                locale=locale,
                runtime=runtime,
            )
            return CaseDecision(
                context=updated,
                intent_confirmation=phrase,
            )

        updated = await self._extract_slots(text=text, locale=locale, context=updated)
        updated.confidence = intent.confidence
        updated.missing_slots = [s for s in updated.required_slots if s not in updated.collected_slots]
        # For operational intents, prefer gathering minimum diagnostics before handoff.
        # This keeps transfer packets useful and avoids instant "соединяю" loops.
        if (
            intent.intent in _OPERATIONAL_INTENTS
            and updated.missing_slots
            and updated.clarification_attempts < clarification_limit
        ):
            updated.need_handoff = False
            updated.stage = "collecting"
            updated.resolution_status = "in_progress"
            updated.clarification_attempts += 1
            updated.attempted_steps.append(f"pre_handoff_clarify:{updated.missing_slots[0]}")
            q = await self._clarify_question(
                slot=updated.missing_slots[0],
                locale=locale,
                runtime=runtime,
                intent=intent,
            )
            return CaseDecision(context=updated, clarification_question=q)
        updated.need_handoff = self._should_handoff(
            text=text,
            intent=intent,
            context=updated,
            fallback_count=fallback_count,
            runtime=runtime,
        )
        if updated.need_handoff:
            updated.stage = "handoff"
            updated.resolution_status = "handoff"
            reason = self._handoff_reason(
                text=text,
                intent=intent,
                context=updated,
                fallback_count=fallback_count,
                runtime=runtime,
            )
            updated.handoff_reason = reason
            return CaseDecision(context=updated, should_handoff=True, handoff_reason=reason)
        if updated.missing_slots:
            updated.stage = "collecting"
            updated.resolution_status = "in_progress"
            updated.clarification_attempts += 1
            updated.attempted_steps.append(f"clarify:{updated.missing_slots[0]}")
            q = await self._clarify_question(
                slot=updated.missing_slots[0],
                locale=locale,
                runtime=runtime,
                intent=intent,
            )
            return CaseDecision(context=updated, clarification_question=q)
        updated.stage = "diagnosing"
        updated.resolution_status = "in_progress"
        updated.attempted_steps.append("diagnostics_ready")
        return CaseDecision(context=updated)

    def _normalize_prior_and_intent(
        self,
        prior: dict | None,
        *,
        text: str,
        intent: IntentResult,
        runtime: dict,
    ) -> tuple[CaseContext, IntentResult]:
        """Восстанавливает контекст кейса; при коротком ответе после операционного сценария не сбрасывает intent."""

        def _fresh(for_intent: IntentResult) -> CaseContext:
            required = list(_SLOT_RULES.get(for_intent.intent, []))
            return CaseContext(
                intent=for_intent.intent,
                stage="new",
                intent_confirmed=False,
                confidence=for_intent.confidence,
                required_slots=required,
                collected_slots={},
                missing_slots=required,
                attempted_steps=[],
                clarification_attempts=0,
                resolution_status="unknown",
                need_handoff=False,
                handoff_reason=None,
            )

        ctx: CaseContext | None = None
        if prior:
            try:
                ctx = CaseContext.model_validate(prior)
            except Exception:  # noqa: BLE001
                ctx = None

        if ctx is None:
            return _fresh(intent), intent

        if ctx.intent == intent.intent:
            return ctx, intent

        if self._should_stick_operational_prior(ctx, intent, text, runtime):
            log.info(
                "case_prior_intent_sticky",
                preserved_intent=ctx.intent,
                routed_intent=intent.intent,
                routed_confidence=intent.confidence,
            )
            sticky = IntentResult(
                intent=ctx.intent,
                confidence=max(float(ctx.confidence), float(intent.confidence), 0.85),
                requires_human=_default_requires_human(ctx.intent),
            )
            return ctx, sticky

        return _fresh(intent), intent

    @staticmethod
    def _should_stick_operational_prior(
        ctx: CaseContext,
        routed: IntentResult,
        text: str,
        runtime: dict,
    ) -> bool:
        if ctx.intent not in _OPERATIONAL_INTENTS or ctx.intent == "human_agent":
            return False
        if not ctx.missing_slots:
            return False
        if routed.intent in _SMALLTALK_INTENTS | {"human_agent"}:
            return False
        if routed.intent in _CONSULTATIVE_INTENTS and routed.confidence >= 0.72:
            return False
        max_chars = max(8, int(runtime.get("case_sticky_followup_max_chars", 40)))
        if len((text or "").strip()) > max_chars:
            return False
        sticky_max = float(runtime.get("case_sticky_routed_max_confidence", 0.62))
        if routed.intent == "other":
            return True
        return routed.confidence < sticky_max

    def _update_intent_confirmation(
        self,
        *,
        context: CaseContext,
        intent: IntentResult,
        locale: str,
        confidence_min: float,
    ) -> CaseContext:
        # Smalltalk or explicit operator request do not need extra confirmation.
        if intent.intent in {"greeting", "thanks", "goodbye", "human_agent"}:
            context.intent_confirmed = True
            return context
        if context.intent_confirmed:
            return context
        # High confidence intents can proceed without explicit confirmation.
        if intent.confidence >= confidence_min:
            context.intent_confirmed = True
            return context
        # Keep unconfirmed on first turn to ask a short operator-like confirmation.
        context.stage = "collecting"
        return context

    async def _extract_slots(self, *, text: str, locale: str, context: CaseContext) -> CaseContext:
        if not context.required_slots:
            return context
        prompt = (
            "Извлеки значения слотов из сообщения клиента. "
            'Верни только JSON: {"slots":{"slot_name":"value"}}. '
            "Если значения нет, слот не включай."
            if locale != "uz"
            else "Mijoz xabaridan slot qiymatlarini ajrating. "
            'Faqat JSON qaytaring: {"slots":{"slot_name":"value"}}.'
        )
        user = f"Intent: {context.intent}\nRequired slots: {context.required_slots}\nText: {text}"
        slots: dict = {}
        try:
            raw = await self._llm.complete(
                [ChatMessage(role="system", content=prompt), ChatMessage(role="user", content=user)],
                temperature=0.0,
                max_tokens=120,
            )
            data = json.loads(_first_json(raw) or "{}")
            parsed = data.get("slots") or {}
            if isinstance(parsed, dict):
                slots = {str(k): str(v).strip() for k, v in parsed.items() if str(v).strip()}
        except Exception as exc:  # noqa: BLE001
            log.warning("case_slot_parse_failed", error=str(exc))
        merged = dict(context.collected_slots)
        merged.update(slots)
        context.collected_slots = merged
        return context

    def _should_handoff(
        self,
        *,
        text: str,
        intent: IntentResult,
        context: CaseContext,
        fallback_count: int,
        runtime: dict,
    ) -> bool:
        t = (text or "").lower()
        fallback_limit = max(1, int(runtime.get("case_fallback_handoff_limit", 2)))
        clarification_limit = max(1, int(runtime.get("case_clarification_max_attempts", 2)))
        low_conf_threshold = float(runtime.get("case_low_confidence_threshold", 0.45))
        if intent.requires_human:
            return True
        if fallback_count >= fallback_limit:
            return True
        if intent.confidence < low_conf_threshold and context.stage != "new":
            return True
        if context.clarification_attempts >= clarification_limit and context.missing_slots:
            return True
        if any(k in t for k in _HANDOFF_TRIGGER_WORDS):
            return True
        return False

    def _handoff_reason(
        self,
        *,
        text: str,
        intent: IntentResult,
        context: CaseContext,
        fallback_count: int,
        runtime: dict,
    ) -> str:
        t = (text or "").lower()
        fallback_limit = max(1, int(runtime.get("case_fallback_handoff_limit", 2)))
        low_conf_threshold = float(runtime.get("case_low_confidence_threshold", 0.45))
        clarification_limit = max(1, int(runtime.get("case_clarification_max_attempts", 2)))
        if intent.requires_human:
            return f"intent:{intent.intent}"
        if any(k in t for k in _HANDOFF_TRIGGER_WORDS):
            return "customer_requested_human"
        if fallback_count >= fallback_limit:
            return "fallback_limit_reached"
        if intent.confidence < low_conf_threshold:
            return "low_intent_confidence"
        if context.clarification_attempts >= clarification_limit and context.missing_slots:
            return "clarification_limit_reached"
        if context.missing_slots:
            return "insufficient_diagnostics_data"
        return "manual"

    async def _intent_confirmation_phrase(self, *, intent: IntentResult, locale: str, runtime: dict) -> str:
        if bool(runtime.get("case_use_dynamic_phrasing", True)):
            strict_intents = runtime.get("case_strict_template_intents") or []
            if intent.intent not in strict_intents:
                generated = await self._generate_dynamic_intent_confirmation(
                    intent=intent.intent,
                    locale=locale,
                )
                if generated:
                    return generated
        templates = runtime.get("case_intent_confirmation_templates") or {}
        if isinstance(templates, dict):
            intent_tpl = templates.get(intent.intent)
            if isinstance(intent_tpl, dict):
                txt = intent_tpl.get(locale)
                if isinstance(txt, str) and txt.strip():
                    return txt.format(intent=intent.intent).strip()
            default_tpl = templates.get("default")
            if isinstance(default_tpl, dict):
                txt = default_tpl.get(locale)
                if isinstance(txt, str) and txt.strip():
                    return txt.format(intent=intent.intent).strip()
        ru = f"Правильно понимаю, что вопрос по теме «{intent.intent}»?"
        uz = f"To'g'ri tushundimmi, savol «{intent.intent}» mavzusi bo'yichami?"
        return uz if locale == "uz" else ru

    async def _clarify_question(
        self,
        *,
        slot: str,
        locale: str,
        runtime: dict,
        intent: IntentResult,
    ) -> str:
        if bool(runtime.get("case_use_dynamic_phrasing", True)):
            strict_slots = runtime.get("case_strict_template_slots") or []
            if slot not in strict_slots:
                generated = await self._generate_dynamic_slot_question(
                    slot=slot,
                    intent=intent.intent,
                    locale=locale,
                )
                if generated:
                    return generated
        templates = runtime.get("case_slot_question_templates") or {}
        if isinstance(templates, dict):
            slot_tpl = templates.get(slot)
            if isinstance(slot_tpl, dict):
                txt = slot_tpl.get(locale)
                if isinstance(txt, str) and txt.strip():
                    return txt.strip()
            default_tpl = templates.get("default")
            if isinstance(default_tpl, dict):
                txt = default_tpl.get(locale)
                if isinstance(txt, str) and txt.strip():
                    return txt.strip()
        ru_map = {
            "card_last4": "Подскажите, пожалуйста, последние 4 цифры карты.",
            "incident_time": "Когда примерно возникла проблема?",
            "channel": "Через какой канал была операция: приложение, сайт или банкомат?",
            "source_account": "С какого счета выполняли перевод?",
            "target_account": "На какой счет или карту отправляли перевод?",
            "amount": "Уточните сумму операции.",
            "customer_id": "Назовите, пожалуйста, номер договора или ID клиента.",
            "field_to_change": "Какие именно данные нужно изменить?",
            "reason": "Уточните, пожалуйста, причину обращения.",
            "topic": "По какому вопросу хотите оставить обращение?",
            "details": "Опишите, пожалуйста, проблему чуть подробнее.",
            "product_name": "Какой именно продукт или тариф вас интересует?",
        }
        uz_map = {
            "card_last4": "Iltimos, kartaning oxirgi 4 raqamini ayting.",
            "incident_time": "Muammo qachon yuz berdi?",
            "channel": "Operatsiya qaysi kanal orqali bo'lgan: ilova, sayt yoki bankomat?",
            "source_account": "Pul o'tkazma qaysi hisobdan yuborilgan?",
            "target_account": "Pul qaysi hisob yoki kartaga yuborilgan?",
            "amount": "Operatsiya summasini ayting.",
            "customer_id": "Shartnoma raqami yoki mijoz ID sini ayting.",
            "field_to_change": "Qaysi ma'lumotni o'zgartirmoqchisiz?",
            "reason": "Murojaat sababini aniqlashtiring.",
            "topic": "Murojaat mavzusini ayting.",
            "details": "Muammoni biroz batafsil tasvirlab bering.",
            "product_name": "Qaysi mahsulot yoki tarif sizni qiziqtiryapti?",
        }
        if locale == "uz":
            return uz_map.get(slot, "Iltimos, masalani aniqlashtirish uchun qo'shimcha ma'lumot bering.")
        return ru_map.get(slot, "Уточните, пожалуйста, дополнительную деталь по вашему запросу.")

    async def _generate_dynamic_intent_confirmation(self, *, intent: str, locale: str) -> str:
        sys = (
            "Сформулируй короткое подтверждение интента для клиента. "
            "Один вежливый вопрос, без markdown, без списков, без служебных слов."
            if locale != "uz"
            else "Mijoz uchun intentni tasdiqlovchi bitta qisqa savol yozing. "
            "Markdown va ro'yxatlarsiz, muloyim uslubda."
        )
        user = f"intent={intent}\nlocale={locale}\nstyle=contact_center"
        try:
            text = await self._llm.complete(
                [ChatMessage(role="system", content=sys), ChatMessage(role="user", content=user)],
                temperature=0.2,
                max_tokens=48,
            )
            out = " ".join((text or "").strip().split())
            return out[:220] if out else ""
        except Exception:  # noqa: BLE001
            return ""

    async def _generate_dynamic_slot_question(self, *, slot: str, intent: str, locale: str) -> str:
        sys = (
            "Сформулируй один уточняющий вопрос операторского стиля для сбора недостающего слота. "
            "Один вопрос, без markdown."
            if locale != "uz"
            else "Yetishmayotgan slotni yig'ish uchun operator uslubida bitta aniqlashtiruvchi savol yozing."
        )
        user = f"intent={intent}\nmissing_slot={slot}\nlocale={locale}"
        try:
            text = await self._llm.complete(
                [ChatMessage(role="system", content=sys), ChatMessage(role="user", content=user)],
                temperature=0.2,
                max_tokens=64,
            )
            out = " ".join((text or "").strip().split())
            return out[:240] if out else ""
        except Exception:  # noqa: BLE001
            return ""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _first_json(text: str) -> str | None:
    m = _JSON_RE.search(text or "")
    return m.group(0) if m else None
