"""Health & component-status endpoints."""

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.db import ping_mongo
from app.core.vectorstore import ping_chroma

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    """Liveness — always 200 if the app is up."""
    return {"status": "ok", "app": get_settings().app_name}


@router.get("/health/components")
async def component_health():
    """Readiness — per-component status (Mongo optional in dev)."""
    settings = get_settings()
    return {
        "mongo": "up" if await ping_mongo() else "down",
        "chroma": "up" if ping_chroma() else "down",
        "llm_configured": bool(settings.nvidia_api_key),
        "llm_model": settings.llm_model,
    }
