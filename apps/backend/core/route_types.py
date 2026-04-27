"""Shared routing types for orchestrator + embedding-based semantic router."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RouteClass(str, Enum):
    NOISE = "noise"
    CACHED = "cached"
    SIMPLE = "simple"
    COMPLEX = "complex"
    ESCALATION = "escalation"


@dataclass(slots=True)
class RoutePolicy:
    run_intent: bool
    run_rag: bool
    allow_tooling: bool


@dataclass(slots=True)
class RouteDecision:
    route_class: RouteClass
    confidence: float
    policy: RoutePolicy
    intent_hint: str | None
    requires_human: bool
    reason: str
