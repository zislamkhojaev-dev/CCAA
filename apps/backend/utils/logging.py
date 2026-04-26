"""Structured logging configured once at startup.

We use `structlog` so that every record is JSON in prod and pretty in dev.
Latency events are emitted as structured fields (`stage`, `duration_ms`)
so a downstream collector can alert when the 1.5s budget is exceeded.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Iterator

import structlog

from apps.backend.config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.environment == "dev":
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(level=level, format="%(message)s", stream=sys.stdout)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


@contextmanager
def timed_stage(stage: str, **bindings) -> Iterator[dict]:
    """Measure wall time of a pipeline stage and emit a structured log line.

    Usage:
        with timed_stage("stt") as ctx:
            text = await stt.transcribe(audio)
            ctx["chars"] = len(text)
    """
    log = get_logger("voice_engine")
    start = time.perf_counter()
    payload: dict = {}
    try:
        yield payload
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0
        log.info(
            "stage_complete",
            stage=stage,
            duration_ms=round(duration_ms, 2),
            **bindings,
            **payload,
        )
