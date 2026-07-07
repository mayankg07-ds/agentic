"""Oracle Finance — FastAPI entrypoint.

Run (from backend/):  uvicorn app.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.routers import health

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description=(
        "Multi-Investor RAG Agent — grounded financial advice with citations."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers — one per checkpoint as they land:
app.include_router(health.router)
# app.include_router(chat.router, prefix="/api/v1")      # Checkpoint 3
# app.include_router(personas.router, prefix="/api/v1")  # Checkpoint 4
# app.include_router(market.router, prefix="/api/v1")    # Checkpoint 5
# app.include_router(ingest.router, prefix="/api/v1")    # Checkpoint 7


@app.get("/")
async def root():
    return {"app": settings.app_name, "docs": "/docs"}
