"""Redis connection management and low-level key/value operations."""

import json
import os

import redis.asyncio as aioredis

SESSION_TTL = 3600  # 1 hour
KEY_PREFIX = "chat:session"
CLUSTER_PREFIX = "context:clusters"
TOOL_CTX_PREFIX = "tool_ctx"

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


# ── Tool context store ────────────────────────────────────────────────────

def _tool_ctx_key(session_id: str, call_id: str) -> str:
    return f"{TOOL_CTX_PREFIX}:{session_id}:{call_id}"


def _tool_ctx_index_key(session_id: str) -> str:
    return f"{TOOL_CTX_PREFIX}:{session_id}:_index"


async def store_tool_result(session_id: str, call_id: str, tool_name: str, result: str) -> None:
    """Store a tool result in the context store and add to the session index."""
    assert _redis is not None, "Redis not connected"
    key = _tool_ctx_key(session_id, call_id)
    value = json.dumps({"tool": tool_name, "call_id": call_id, "result": result})
    await _redis.set(key, value, ex=SESSION_TTL)
    await _redis.rpush(_tool_ctx_index_key(session_id), call_id)
    await _redis.expire(_tool_ctx_index_key(session_id), SESSION_TTL)


async def get_tool_result(session_id: str, call_id: str) -> dict | None:
    """Load a single tool result from the context store."""
    assert _redis is not None, "Redis not connected"
    raw = await _redis.get(_tool_ctx_key(session_id, call_id))
    if raw is None:
        return None
    return json.loads(raw)


async def list_tool_results(session_id: str) -> list[dict]:
    """List all tool results for a session in execution order."""
    assert _redis is not None, "Redis not connected"
    index_key = _tool_ctx_index_key(session_id)
    call_ids = await _redis.lrange(index_key, 0, -1)
    results = []
    for cid in call_ids:
        entry = await get_tool_result(session_id, cid)
        if entry:
            results.append(entry)
    return results


async def clear_tool_context(session_id: str) -> int:
    """Clear all tool results for a session. Returns count removed."""
    assert _redis is not None, "Redis not connected"
    index_key = _tool_ctx_index_key(session_id)
    call_ids = await _redis.lrange(index_key, 0, -1)
    count = 0
    for cid in call_ids:
        count += await _redis.delete(_tool_ctx_key(session_id, cid))
    await _redis.delete(index_key)
    return count
