"""MongoDB (Motor) connection — lazy, optional in dev.

The app boots even if Mongo is unreachable; /health reports component status.
"""

from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import get_settings

_client: AsyncIOMotorClient | None = None


def get_mongo_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncIOMotorClient(
            settings.mongodb_uri, serverSelectionTimeoutMS=3000
        )
    return _client


def get_db() -> AsyncIOMotorDatabase:
    return get_mongo_client()[get_settings().mongodb_db]


async def ping_mongo() -> bool:
    try:
        await get_mongo_client().admin.command("ping")
        return True
    except Exception:
        return False
