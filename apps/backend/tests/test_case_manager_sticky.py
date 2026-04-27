"""Сохранение операционного контекста при коротком ответе с неверным интентом."""

from __future__ import annotations

from apps.backend.core.case_manager import CaseManager
from apps.backend.models.schemas import CaseContext, IntentResult


def _transfer_prior() -> dict:
    return CaseContext(
        intent="transfer_money",
        stage="collecting",
        intent_confirmed=True,
        confidence=0.9,
        required_slots=["source_account", "target_account", "amount", "incident_time"],
        collected_slots={},
        missing_slots=["source_account", "target_account", "amount", "incident_time"],
        attempted_steps=["pre_handoff_clarify:source_account"],
        clarification_attempts=1,
        resolution_status="in_progress",
        need_handoff=False,
        handoff_reason=None,
    ).model_dump(mode="json")


def test_sticky_prior_preserves_transfer_on_short_other() -> None:
    cm = CaseManager()
    routed = IntentResult(intent="other", confidence=0.2, requires_human=False)
    ctx, eff = cm._normalize_prior_and_intent(
        _transfer_prior(), text="своего.", intent=routed, runtime={}
    )
    assert eff.intent == "transfer_money"
    assert ctx.intent == "transfer_money"
    assert ctx.missing_slots


def test_no_stick_when_user_asks_operator() -> None:
    cm = CaseManager()
    routed = IntentResult(intent="human_agent", confidence=0.95, requires_human=True)
    ctx, eff = cm._normalize_prior_and_intent(
        _transfer_prior(), text="оператор", intent=routed, runtime={}
    )
    assert eff.intent == "human_agent"
    assert ctx.intent == "human_agent"


def test_no_stick_on_long_unrelated_message() -> None:
    cm = CaseManager()
    routed = IntentResult(intent="other", confidence=0.2, requires_human=False)
    long_text = "я вообще про другое хотел спросить про вклады и проценты по ним"
    ctx, eff = cm._normalize_prior_and_intent(
        _transfer_prior(), text=long_text, intent=routed, runtime={}
    )
    assert eff.intent == "other"
    assert ctx.intent == "other"
