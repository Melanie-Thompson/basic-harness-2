"""Pydantic Monty sandbox — pool management, per-reply sessions, and LangChain
tool wrappers that execute host functions inside the sandbox."""

import json
from collections.abc import Callable
from contextlib import AsyncExitStack

import pydantic_monty as pm
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, create_model, Field

from AI.models.message import ToolExecutionRecord
from AI.signing import canonicalize_args, sign_execution
from AI.tools import ALL_TOOLS, SANDBOX_FUNCTIONS

MAX_TOOL_OUTPUT = 4_000

SANDBOX_LIMITS = pm.ResourceLimits(
    max_feed_duration_secs=2.0,
    max_memory=64 * 1024 * 1024,
)

# ── Pool ─────────────────────────────────────────────────────────────────────

_pool: pm.AsyncMonty | None = None


async def _get_pool() -> pm.AsyncMonty:
    global _pool
    if _pool is None:
        pool = pm.AsyncMonty(min_processes=1, max_processes=4)
        await pool.__aenter__()
        _pool = pool
    return _pool


async def close_sandbox() -> None:
    global _pool
    if _pool is not None:
        await _pool.__aexit__(None, None, None)
        _pool = None


# ── Session ──────────────────────────────────────────────────────────────────

class SandboxSession:
    def __init__(self, stack: AsyncExitStack):
        self._stack = stack
        self._session: pm.AsyncMontySession | None = None

    async def _fresh(self) -> pm.AsyncMontySession:
        pool = await _get_pool()
        self._session = await self._stack.enter_async_context(
            pool.checkout(limits=SANDBOX_LIMITS)
        )
        return self._session

    async def call(self, func_name: str, **kwargs) -> str:
        """Call a host function by name inside the sandbox."""
        session = self._session or await self._fresh()
        out = pm.CollectString(max_bytes=64 * 1024)

        args_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in kwargs.items())
        code = f"{func_name}({args_str})"

        try:
            result = await session.feed_run(
                code, external_lookup=SANDBOX_FUNCTIONS, print_callback=out
            )
        except pm.MontySyntaxError as e:
            return f"SyntaxError: {e}"
        except pm.MontyRuntimeError as e:
            if "TimeoutError" in str(e):
                self._session = None
                return f"{e}\n(sandbox reset)"
            return _with_stdout(out.output, str(e))
        except pm.MontyError as e:
            self._session = None
            return f"sandbox error: {type(e).__name__}: {e}"

        shown = "" if result is None else _to_text(result)
        return _with_stdout(out.output, shown) or "(no output)"

    async def call_signed(self, func_name: str, **kwargs) -> tuple[str, ToolExecutionRecord]:
        result = await self.call(func_name, **kwargs)
        canonical = canonicalize_args(kwargs)
        signature = sign_execution(func_name, canonical, result)
        record = ToolExecutionRecord(
            tool_name=func_name,
            tool_input=kwargs,
            tool_output=result,
            signature=signature,
        )
        return result, record


# ── Helpers ──────────────────────────────────────────────────────────────────

def _to_text(value) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=repr)
    except (TypeError, ValueError):
        return repr(value)


def _with_stdout(stdout: str, tail: str) -> str:
    text = "\n".join(p for p in (stdout.rstrip(), tail) if p)
    if len(text) > MAX_TOOL_OUTPUT:
        text = text[:MAX_TOOL_OUTPUT] + f"\n…[truncated {len(text) - MAX_TOOL_OUTPUT} chars]"
    return text


def _args_model(json_schema: dict, name: str) -> type[BaseModel]:
    """Build a Pydantic model from a JSON Schema for LangChain's args_schema."""
    props = json_schema.get("properties", {})
    required = set(json_schema.get("required", []))
    fields = {}
    type_map = {"string": str, "integer": int, "number": float, "boolean": bool}
    for prop_name, prop in props.items():
        py_type = type_map.get(prop.get("type", "string"), str)
        desc = prop.get("description", "")
        if prop_name in required and "default" not in prop:
            fields[prop_name] = (py_type, Field(..., description=desc))
        else:
            fields[prop_name] = (py_type, Field(default=prop.get("default"), description=desc))
    return create_model(f"{name}_args", **fields)


# ── LangChain tools ─────────────────────────────────────────────────────────

def make_tools(sandbox: SandboxSession) -> list[BaseTool]:
    """Create one LangChain tool per ToolDefinition, each executing inside the sandbox."""
    tools: list[BaseTool] = []
    for defn in ALL_TOOLS:

        async def _run(sandbox=sandbox, func_name=defn.name, **kwargs) -> tuple[str, ToolExecutionRecord]:
            return await sandbox.call_signed(func_name, **kwargs)

        tools.append(StructuredTool.from_function(
            coroutine=_run,
            name=defn.name,
            description=defn.description,
            args_schema=_args_model(defn.inputSchema, defn.name),
            response_format="content_and_artifact",
        ))
    return tools
