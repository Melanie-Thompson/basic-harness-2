"""Redis connection management and low-level key/value operations."""

import os

import redis.asyncio as aioredis

SESSION_TTL = 3600  # 1 hour
KEY_PREFIX = "chat:session"
CLUSTER_PREFIX = "context:clusters"

_redis: aioredis.Redis | None = None


async def connect(url: str | None = None) -> None:
    global _redis
    _redis = aioredis.from_url(
        url or os.getenv("REDIS_URL", "redis://localhost:6379"),
        decode_responses=True,
    )
    await _redis.ping()


async def close() -> None:
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None


def _key(session_id: str) -> str:
    return f"{KEY_PREFIX}:{session_id}"


async def get(session_id: str) -> str | None:
    assert _redis is not None, "Redis not connected"
    return await _redis.get(_key(session_id))


async def set(session_id: str, data: str) -> None:
    assert _redis is not None, "Redis not connected"
    await _redis.set(_key(session_id), data, ex=SESSION_TTL)


def _cluster_key(session_id: str) -> str:
    return f"{CLUSTER_PREFIX}:{session_id}"


async def get_clusters(session_id: str) -> str | None:
    assert _redis is not None, "Redis not connected"
    return await _redis.get(_cluster_key(session_id))


async def set_clusters(session_id: str, data: str) -> None:
    assert _redis is not None, "Redis not connected"
    await _redis.set(_cluster_key(session_id), data, ex=SESSION_TTL)
