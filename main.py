"""FastAPI transport layer.  uvicorn main:app --reload --port 8001"""

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from AI import redis_store, sessions
from AI.agent import stream_reply
from AI.sandbox import close_sandbox

log = logging.getLogger("agent")

_session_id: str = ""


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _session_id
    await redis_store.connect()
    _session_id = sessions.new_session_id()
    log.info("session %s initialised", _session_id)
    yield
    await redis_store.close()
    await close_sandbox()


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    system: str | None = None


def sse(payload: dict | str) -> str:
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {body}\n\n"


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    history = await sessions.load(_session_id)
    turn_messages: list[BaseMessage] = []
    deltas: list[str] = []

    kwargs: dict = {"history": history, "turn_messages": turn_messages, "session_id": _session_id}
    if req.system is not None:
        kwargs["system"] = req.system

    # Consume the full stream first, then persist and send
    error: Exception | None = None
    try:
        async for delta in stream_reply(req.message, **kwargs):
            deltas.append(delta)
    except Exception as exc:                          # noqa: BLE001
        log.exception("stream failed")
        error = exc

    # Save to Redis before sending the response
    if turn_messages:
        await sessions.save(_session_id, history + turn_messages)

    async def events():
        yield sse({"session_id": _session_id})
        if error:
            yield sse({"error": f"{type(error).__name__}: {error}"})
        else:
            for d in deltas:
                yield sse({"delta": d})
        yield sse("[DONE]")

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
