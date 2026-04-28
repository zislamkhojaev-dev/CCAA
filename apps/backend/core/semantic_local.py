"""Local embeddings for semantic intent/router (default: multilingual-e5-small)."""

from __future__ import annotations

import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from apps.backend.config import get_settings
from apps.backend.core.route_types import RouteClass, RouteDecision, RoutePolicy
from apps.backend.models.schemas import IntentResult
from apps.backend.utils.logging import get_logger

log = get_logger(__name__)

# (intent_id, paraphrase) — RU/UZ mix for multilingual semantic matching.
INTENT_PROTOTYPES: list[tuple[str, str]] = [
    ("greeting", "здравствуйте"),
    ("greeting", "добрый день"),
    ("greeting", "привет"),
    ("greeting", "assalomu alaykum"),
    ("thanks", "спасибо"),
    ("thanks", "благодарю"),
    ("thanks", "rahmat"),
    ("goodbye", "до свидания"),
    ("goodbye", "пока"),
    ("goodbye", "xayr"),
    ("human_agent", "соедините с оператором"),
    ("human_agent", "хочу говорить с человеком"),
    ("human_agent", "переключите на оператора"),
    ("human_agent", "operator bilan gaplashmoqchiman"),
    ("block_card", "заблокировать карту"),
    ("block_card", "карту украли"),
    ("block_card", "хочу заблокировать карту"),
    ("block_card", "нужно срочно заблокировать карту"),
    ("block_card", "kartani bloklash"),
    ("transfer_money", "перевод денег"),
    ("transfer_money", "перевести деньги на другой счет"),
    ("transfer_money", "pul otkazma"),
    ("personal_data", "сменить паспортные данные"),
    ("personal_data", "изменить персональные данные"),
    ("complaint", "хочу оставить жалобу"),
    ("complaint", "жалоба в банк"),
    ("complaint", "shikoyat"),
    ("tariffs", "какие тарифы"),
    ("tariffs", "тарифы по вкладам"),
    ("tariffs", "стоимость услуг"),
    ("tariffs", "tariflar"),
    ("products", "какие продукты банка"),
    ("products", "какие услуги вы предлагаете"),
    ("office_hours", "режим работы отделений"),
    ("office_hours", "во сколько вы работаете"),
    ("branches", "где ближайший филиал"),
    ("branches", "адрес отделения"),
]

_ROUTES: dict[str, tuple[str, ...]] = {
    "escalation": (
        "соедините с оператором",
        "переключите на человека",
        "хочу живого оператора",
        "позовите менеджера",
        "human agent please",
    ),
    "greeting": ("здравствуйте", "добрый день", "привет", "слушаю вас"),
    "thanks": ("спасибо", "благодарю", "rahmat"),
    "goodbye": ("до свидания", "пока", "всего доброго"),
    "knowledge": (
        "какие тарифы",
        "сколько стоит обслуживание",
        "как оформить карту",
        "где отделение",
        "режим работы банка",
        "расскажите про услуги",
    ),
}

_OP_INTENTS = {"block_card", "transfer_money", "personal_data", "complaint"}

# Явный запрос человека — до эмбеддингового «привет vs оператор».
_EXPLICIT_HANDOFF_SUBSTR: tuple[str, ...] = (
    "оператор",
    "человек",
    "переключ",
    "соедин",
    "позови",
    "менеджер",
    "operator",
    "human agent",
    "живого оператора",
    "odam",
)
_SMALLTALK_GREETING_SUBSTR: tuple[str, ...] = (
    "привет",
    "здравств",
    "добрый",
    "hello",
    "hi",
    "salom",
    "assalomu",
)
_SMALLTALK_THANKS_SUBSTR: tuple[str, ...] = ("спасибо", "благодар", "rahmat", "thank")
_SMALLTALK_GOODBYE_SUBSTR: tuple[str, ...] = ("пока", "до свид", "goodbye", "bye", "xayr")

_lock = threading.Lock()
_embedder = None
_embed_mode: str | None = None
_embed_registration: str | None = None
_intent_mat: np.ndarray | None = None
_intent_ids: list[str] | None = None
_route_mats: dict[str, np.ndarray] | None = None
_embed_broken = False
_E5_MAX_LENGTH = 512


class OnnxE5Embedder:
    def __init__(self, model_id: str) -> None:
        onnx_path = hf_hub_download(repo_id=model_id, filename="onnx/model.onnx")
        tokenizer_path = hf_hub_download(repo_id=model_id, filename="tokenizer.json")
        self.model_path = str(Path(onnx_path).resolve())
        self.tokenizer_path = str(Path(tokenizer_path).resolve())
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {inp.name for inp in self.session.get_inputs()}

    def embed(self, texts: Sequence[str], *, batch_size: int = 32) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            if not batch:
                continue
            enc = self.tokenizer.encode_batch(batch)
            input_ids = [e.ids[:_E5_MAX_LENGTH] for e in enc]
            attention_mask = [e.attention_mask[:_E5_MAX_LENGTH] for e in enc]
            # Pad to the same sequence length for ONNX.
            max_len = int(max(len(row) for row in input_ids))
            padded_ids = np.zeros((len(batch), max_len), dtype=np.int64)
            padded_mask = np.zeros((len(batch), max_len), dtype=np.int64)
            for r, (ids, mask) in enumerate(zip(input_ids, attention_mask, strict=False)):
                ln = len(ids)
                padded_ids[r, :ln] = np.asarray(ids, dtype=np.int64)
                padded_mask[r, :ln] = np.asarray(mask, dtype=np.int64)

            feed: dict[str, np.ndarray] = {
                "input_ids": padded_ids,
                "attention_mask": padded_mask,
            }
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.zeros_like(padded_ids, dtype=np.int64)

            hidden = self.session.run(None, feed)[0]  # [B, T, H]
            mask = padded_mask.astype(np.float32)[..., None]  # [B, T, 1]
            summed = (hidden * mask).sum(axis=1)
            denom = np.clip(mask.sum(axis=1), 1e-9, None)
            pooled = summed / denom
            out.extend(np.asarray(row, dtype=np.float32) for row in pooled)
        return out


def _l2n(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v)) + 1e-9
    return v / n


def _get_embedder():
    global _embedder, _embed_broken, _embed_mode, _embed_registration
    if _embed_broken:
        return None
    model_id = (get_settings().semantic_embed_model or "").strip() or "intfloat/multilingual-e5-small"

    if _embedder is not None:
        return _embedder
    if not _lock.acquire(blocking=False):
        # Another request/background task is initializing model now.
        return None
    try:
        if _embedder is None:
            try:
                _embedder = OnnxE5Embedder(model_id)
                _embed_mode = "onnxruntime"
                _embed_registration = "hf_onnx"
            except Exception as exc:  # noqa: BLE001
                log.warning("semantic_embed_init_failed", model=model_id, error=str(exc))
                _embed_broken = True
                return None
            log.info(
                "semantic_embed_model_loaded",
                model=model_id,
                mode=_embed_mode,
                registration=_embed_registration or "unknown",
            )
    finally:
        _lock.release()
    return _embedder


def _embed_texts(texts: Sequence[str]) -> np.ndarray | None:
    m = _get_embedder()
    if m is None:
        return None
    try:
        rows = m.embed(list(texts), batch_size=32)  # type: ignore[attr-defined]
        return np.stack([np.asarray(r, dtype=np.float32) for r in rows])
    except Exception as exc:  # noqa: BLE001
        log.warning("semantic_embed_failed", mode=_embed_mode or "unknown", error=str(exc))
        return None


def _ensure_intent_index() -> bool:
    global _intent_mat, _intent_ids
    with _lock:
        if _intent_mat is not None and _intent_ids is not None:
            return True
    phrases = [p for _, p in INTENT_PROTOTYPES]
    labels = [lab for lab, _ in INTENT_PROTOTYPES]
    mat = _embed_texts(phrases)
    if mat is None:
        return False
    mat = np.stack([_l2n(row) for row in mat])
    with _lock:
        if _intent_mat is None:
            _intent_mat = mat
            _intent_ids = labels
    return _intent_mat is not None


def _ensure_route_mats() -> bool:
    global _route_mats
    with _lock:
        if _route_mats is not None:
            return True
    mats: dict[str, np.ndarray] = {}
    for key, phrases in _ROUTES.items():
        mat = _embed_texts(list(phrases))
        if mat is None:
            return False
        mats[key] = np.stack([_l2n(row) for row in mat])
    with _lock:
        if _route_mats is None:
            _route_mats = mats
    return _route_mats is not None


def _max_cos(q: np.ndarray, mat: np.ndarray) -> float:
    return float(np.max(mat @ q))


def _explicit_handoff(text: str) -> bool:
    t = (text or "").lower()
    return any(s in t for s in _EXPLICIT_HANDOFF_SUBSTR)


def is_explicit_handoff(text: str) -> bool:
    return _explicit_handoff(text)


def match_smalltalk_keyword(text: str) -> str | None:
    t = (text or "").lower()
    if any(k in t for k in _SMALLTALK_GREETING_SUBSTR):
        return "greeting"
    if any(k in t for k in _SMALLTALK_THANKS_SUBSTR):
        return "thanks"
    if any(k in t for k in _SMALLTALK_GOODBYE_SUBSTR):
        return "goodbye"
    return None


def _best_sim_per_intent(ids: list[str], sims: np.ndarray) -> dict[str, float]:
    agg: dict[str, float] = defaultdict(float)
    for i, lab in enumerate(ids):
        v = float(sims[i])
        if v > agg[lab]:
            agg[lab] = v
    return dict(agg)


def match_intent_semantic(text: str, runtime: dict) -> IntentResult | None:
    """Return intent from local embeddings, or None to fall back to LLM."""
    if not bool(runtime.get("semantic_intent_embed_enabled", True)):
        return None
    t = (text or "").strip()
    if len(t) < 2:
        return None
    if not _ensure_intent_index() or _intent_mat is None or _intent_ids is None:
        return None
    qv = _embed_texts([t])
    if qv is None:
        return None
    q = _l2n(qv[0])
    sims = _intent_mat @ q
    idx = int(np.argmax(sims))
    best = float(sims[idx])
    thr = float(runtime.get("semantic_intent_min_cos", 0.38))
    if best < thr:
        return None
    intent = _intent_ids[idx]
    per = _best_sim_per_intent(_intent_ids, sims)
    small_max = max(per.get("greeting", 0.0), per.get("thanks", 0.0), per.get("goodbye", 0.0))
    amb_gap = float(runtime.get("semantic_intent_human_greeting_ambiguity_max_gap", 0.15))
    if (
        intent == "human_agent"
        and not _explicit_handoff(t)
        and small_max >= thr
        and (best - small_max) < amb_gap
    ):
        # Argmax=human при сильном smalltalk и узком разрыве — сбой MiniLM на коротких RU («Привет!»).
        return None
    op_thr = float(runtime.get("semantic_intent_op_min_cos", 0.40))
    requires_human = False
    if intent == "human_agent" and best >= thr and _explicit_handoff(t):
        requires_human = True
    elif intent == "human_agent" and best >= float(
        runtime.get("semantic_intent_human_min_cos", 0.58)
    ):
        # Парафраз «хочу живого специалиста» без словарных токенов — высокий порог.
        requires_human = True
    elif intent in _OP_INTENTS and best >= op_thr:
        requires_human = True
    return IntentResult(intent=intent, confidence=best, requires_human=requires_human)


def route_by_embedding(text: str, *, locale: str, runtime: dict) -> RouteDecision | None:
    """Semantic route (MiniLM) — None → caller uses keyword/heuristic router."""
    del locale  # reserved for future locale-specific prototypes
    if not bool(runtime.get("semantic_router_embed_enabled", True)):
        return None
    t = " ".join((text or "").strip().lower().split())
    noise_max_words = max(1, int(runtime.get("router_noise_max_words", 2)))
    noise_max_chars = max(4, int(runtime.get("router_noise_max_chars", 14)))
    simple_max_words = max(2, int(runtime.get("router_simple_max_words", 5)))
    simple_conf = float(runtime.get("router_simple_min_confidence", 0.7))
    esc_conf = float(runtime.get("router_escalation_min_confidence", 0.95))
    noise_run_intent = bool(runtime.get("router_policy_noise_run_intent", False))
    noise_run_rag = bool(runtime.get("router_policy_noise_run_rag", False))
    simple_run_intent = bool(runtime.get("router_policy_simple_run_intent", True))
    simple_run_rag = bool(runtime.get("router_policy_simple_run_rag", True))
    complex_run_intent = bool(runtime.get("router_policy_complex_run_intent", True))
    complex_run_rag = bool(runtime.get("router_policy_complex_run_rag", True))

    thr_esc = float(runtime.get("router_embed_escalation_cos", 0.42))
    thr_sm = float(runtime.get("router_embed_smalltalk_cos", 0.40))
    thr_kn = float(runtime.get("router_embed_knowledge_cos", 0.36))
    thr_noise = float(runtime.get("router_embed_noise_max_cos", 0.34))

    if not t:
        return RouteDecision(
            route_class=RouteClass.NOISE,
            confidence=0.95,
            policy=RoutePolicy(
                run_intent=noise_run_intent, run_rag=noise_run_rag, allow_tooling=False
            ),
            intent_hint="other",
            requires_human=False,
            reason="empty_or_silence",
        )

    if not _ensure_route_mats() or _route_mats is None:
        return None

    qv = _embed_texts([t])
    if qv is None:
        return None
    q = _l2n(qv[0])
    s_esc = _max_cos(q, _route_mats["escalation"])
    s_gr = _max_cos(q, _route_mats["greeting"])
    s_th = _max_cos(q, _route_mats["thanks"])
    s_by = _max_cos(q, _route_mats["goodbye"])
    s_kn = _max_cos(q, _route_mats["knowledge"])
    substantive = max(s_esc, s_gr, s_th, s_by, s_kn)

    words = len(re.sub(r"[^\w\s]", " ", t).split())

    smalltalk_max_words = max(1, int(runtime.get("router_smalltalk_max_words", 3)))
    smalltalk_margin = float(runtime.get("router_smalltalk_vs_knowledge_margin", 0.03))

    # Сначала smalltalk: иначе ложный s_esc на «Привет!» даёт ESCALATION до приветствия.
    if (
        words <= smalltalk_max_words
        and s_gr >= thr_sm
        and s_gr >= s_th
        and s_gr >= s_by
        and s_gr >= (s_kn + smalltalk_margin)
    ):
        return RouteDecision(
            route_class=RouteClass.SIMPLE,
            confidence=s_gr,
            policy=RoutePolicy(run_intent=False, run_rag=False, allow_tooling=False),
            intent_hint="greeting",
            requires_human=False,
            reason="semantic_embed_greeting",
        )
    if (
        words <= smalltalk_max_words
        and s_th >= thr_sm
        and s_th >= s_gr
        and s_th >= s_by
        and s_th >= (s_kn + smalltalk_margin)
    ):
        return RouteDecision(
            route_class=RouteClass.SIMPLE,
            confidence=s_th,
            policy=RoutePolicy(run_intent=False, run_rag=False, allow_tooling=False),
            intent_hint="thanks",
            requires_human=False,
            reason="semantic_embed_thanks",
        )
    if words <= smalltalk_max_words and s_by >= thr_sm and s_by >= (s_kn + smalltalk_margin):
        return RouteDecision(
            route_class=RouteClass.SIMPLE,
            confidence=s_by,
            policy=RoutePolicy(run_intent=False, run_rag=False, allow_tooling=False),
            intent_hint="goodbye",
            requires_human=False,
            reason="semantic_embed_goodbye",
        )

    esc_margin = float(runtime.get("router_embed_escalation_vs_smalltalk_margin", 0.04))
    if s_esc >= thr_esc and (
        _explicit_handoff(t)
        or s_esc >= max(s_gr, s_th, s_by) + esc_margin
    ):
        return RouteDecision(
            route_class=RouteClass.ESCALATION,
            confidence=max(s_esc, esc_conf),
            policy=RoutePolicy(run_intent=False, run_rag=False, allow_tooling=True),
            intent_hint="human_agent",
            requires_human=True,
            reason="semantic_embed_escalation",
        )

    if (
        words <= noise_max_words
        and len(t) <= noise_max_chars
        and substantive < thr_noise
    ):
        return RouteDecision(
            route_class=RouteClass.NOISE,
            confidence=0.75,
            policy=RoutePolicy(
                run_intent=noise_run_intent, run_rag=noise_run_rag, allow_tooling=False
            ),
            intent_hint="other",
            requires_human=False,
            reason="semantic_embed_low_signal",
        )

    if words <= simple_max_words and s_kn >= thr_kn:
        return RouteDecision(
            route_class=RouteClass.SIMPLE,
            confidence=max(s_kn, simple_conf),
            policy=RoutePolicy(
                run_intent=simple_run_intent, run_rag=simple_run_rag, allow_tooling=True
            ),
            intent_hint=None,
            requires_human=False,
            reason="semantic_embed_short_kb",
        )

    if words <= simple_max_words:
        return RouteDecision(
            route_class=RouteClass.SIMPLE,
            confidence=simple_conf,
            policy=RoutePolicy(
                run_intent=simple_run_intent, run_rag=simple_run_rag, allow_tooling=True
            ),
            intent_hint=None,
            requires_human=False,
            reason="semantic_embed_short_default",
        )

    return RouteDecision(
        route_class=RouteClass.COMPLEX,
        confidence=0.65,
        policy=RoutePolicy(
            run_intent=complex_run_intent, run_rag=complex_run_rag, allow_tooling=True
        ),
        intent_hint=None,
        requires_human=False,
        reason="semantic_embed_complex",
    )


def prewarm_semantic_embeddings() -> bool:
    """Best-effort warmup: initialize embedder and build route/intent indexes."""
    try:
        if _get_embedder() is None:
            return False
        ok_intent = _ensure_intent_index()
        ok_route = _ensure_route_mats()
        log.info(
            "semantic_embed_prewarm_done",
            intent_index_ready=ok_intent,
            route_index_ready=ok_route,
        )
        return ok_intent and ok_route
    except Exception as exc:  # noqa: BLE001
        log.warning("semantic_embed_prewarm_failed", error=str(exc))
        return False
