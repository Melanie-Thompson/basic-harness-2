"""FastAPI transport layer.  uvicorn main:app --reload --port 8001"""

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from AI import redis_store, sessions
from AI.agent import stream_reply
from AI.package_lifecycle import PackageManifest, PackageRegistry
from AI.package_signing import PackageSignature
from AI.sandbox import close_sandbox, inspect_tools
from AI.tools import list_tools

log = logging.getLogger("agent")

_session_id: str = ""
_registry: PackageRegistry | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _session_id, _registry
    await redis_store.connect()
    inspect_tools()
    _registry = PackageRegistry()
    _session_id = sessions.new_session_id()
    log.info("session %s initialised (%d tools)", _session_id, _registry.installed_count)
    yield
    if _registry:
        discarded = _registry.discard_all_external()
        if discarded:
            log.info("session end: discarded %d external packages: %s",
                     len(discarded), discarded)
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


@app.get("/tools")
async def tools():
    """Return tool definitions in Code-to-Agent format (gates excluded)."""
    return {"tools": list_tools()}


class InstallRequest(BaseModel):
    definition: dict
    source_code: str
    publisher: str = ""
    signature: dict | None = None


@app.post("/packages/install")
async def install_package(req: InstallRequest):
    """Install a tool package into the live registry."""
    if _registry is None:
        return {"error": "registry not initialised"}

    sig = None
    if req.signature:
        sig = PackageSignature(**req.signature)

    manifest = PackageManifest(
        definition=req.definition,
        source_code=req.source_code,
        signature=sig,
        publisher=req.publisher,
    )

    try:
        record = _registry.install(manifest)
    except ValueError as e:
        return {"error": str(e)}

    return {
        "installed": record.name,
        "publisher": record.publisher,
        "kind": record.kind,
        "signature_verified": record.signature_verified,
        "warnings": record.warnings,
        "total_tools": _registry.installed_count,
    }


@app.get("/packages")
async def list_packages():
    """Inspect all installed packages."""
    if _registry is None:
        return {"error": "registry not initialised"}
    return {"packages": _registry.inspect()}


@app.get("/packages/{name}")
async def inspect_package(name: str):
    """Inspect a single installed package."""
    if _registry is None:
        return {"error": "registry not initialised"}
    result = _registry.inspect(name)
    if not result:
        return {"error": f"package '{name}' not found"}
    return {"package": result[0]}


@app.delete("/packages/{name}")
async def discard_package(name: str):
    """Discard an installed package."""
    if _registry is None:
        return {"error": "registry not initialised"}
    try:
        removed = _registry.discard(name)
    except ValueError as e:
        return {"error": str(e)}
    if not removed:
        return {"error": f"package '{name}' not found"}
    return {"discarded": name, "total_tools": _registry.installed_count}


@app.get("/test")
async def run_tests() -> StreamingResponse:
    async def stream_pytest():
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pytest", "tests/", "-v", "-s", "--tb=short", "--no-header",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            yield sse({"line": line.decode(errors="replace").rstrip()})

        code = await proc.wait()
        yield sse({"exit_code": code})
        yield sse("[DONE]")

    return StreamingResponse(
        stream_pytest(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
