"""HTTP + WebSocket routers."""

from fastapi import APIRouter

from apps.backend.api.agent_assist_ws import router as agent_assist_router
from apps.backend.api.analytics import router as analytics_router
from apps.backend.api.bot_settings import router as bot_settings_router
from apps.backend.api.conversations import router as conversations_router
from apps.backend.api.documents import router as documents_router
from apps.backend.api.health import router as health_router
from apps.backend.api.playground import router as playground_router
from apps.backend.api.prompts import router as prompts_router
from apps.backend.api.slo import router as slo_router
from apps.backend.api.voice_ws import router as voice_router
from apps.backend.api.voices import router as voices_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(playground_router, prefix="/playground", tags=["playground"])
api_router.include_router(voice_router, tags=["voice"])
api_router.include_router(agent_assist_router, tags=["agent-assist"])
api_router.include_router(documents_router, prefix="/documents", tags=["documents"])
api_router.include_router(prompts_router, prefix="/prompts", tags=["prompts"])
api_router.include_router(voices_router, prefix="/voices", tags=["voices"])
api_router.include_router(analytics_router, prefix="/analytics", tags=["analytics"])
api_router.include_router(conversations_router, prefix="/conversations", tags=["conversations"])
api_router.include_router(bot_settings_router, tags=["bot-settings"])
api_router.include_router(slo_router, tags=["slo"])

__all__ = ["api_router"]
