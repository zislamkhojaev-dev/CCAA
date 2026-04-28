"""FastAPI entry point.

Run locally:    uvicorn apps.backend.main:app --reload
Run in Docker:  see docker/docker-compose.yml
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.backend.api import api_router
from apps.backend.config import get_settings
from apps.backend.core.bot_runtime import get_bot_runtime_payload
from apps.backend.core.semantic_local import prewarm_semantic_embeddings
from apps.backend.models.db import init_db
from apps.backend.rag import get_rag_agent_assist, get_rag_voice
from apps.backend.utils.logging import configure_logging, get_logger


configure_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log.info("startup", env=settings.environment, providers={
        "stt": settings.stt_provider,
        "llm": settings.llm_provider,
        "tts": settings.tts_provider,
    })
    try:
        await init_db()
    except Exception as exc:  # noqa: BLE001
        log.warning("db_init_failed", error=str(exc))
    try:
        await get_rag_voice().ensure_collection()
        await get_rag_agent_assist().ensure_collection()
    except Exception as exc:  # noqa: BLE001
        log.warning("qdrant_init_failed", error=str(exc))
    try:
        await get_bot_runtime_payload()
    except Exception as exc:  # noqa: BLE001
        log.warning("bot_runtime_load_failed", error=str(exc))
    async def _run_semantic_prewarm() -> None:
        try:
            ok = await asyncio.to_thread(prewarm_semantic_embeddings)
            log.info("semantic_embed_prewarm", ready=ok)
        except Exception as exc:  # noqa: BLE001
            log.warning("semantic_embed_prewarm_failed", error=str(exc))

    # Do not block API startup on first-time model download.
    asyncio.create_task(_run_semantic_prewarm())
    log.info("semantic_embed_prewarm_started")
    yield
    log.info("shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()
