"""Pydantic Monty sandbox — pool management, per-reply sessions, and LangChain
tool wrappers that execute host functions inside the sandbox.

Capability enforcement: each tool gets a restricted OSAccess and filtered
external_lookup based on its declared capabilities. Anything undeclared is
blocked by the sandbox runtime.
"""

import json
import logging
from collections.abc import Callable
from contextlib import AsyncExitStack

import pydantic_monty as pm
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, create_model, Field

from AI.capabilities import audit_capabilities, build_external_lookup, build_os_access, check_capabilities
from AI.models.message import ToolExecutionRecord
from AI.package_signing import verify_package
from AI.signing import canonicalize_args, sign_execution
from AI.tools import ALL_TOOLS, SANDBOX_FUNCTIONS, TOOLS_BY_NAME
from AI.tunnel import TunnelManager, SessionTunnel
from AI.tunnel import TunnelConfig as RuntimeTunnelConfig

log = logging.getLogger("sandbox")

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
    def __init__(self, stack: AsyncExitStack, tunnel_manager: TunnelManager | None = None):
        self._stack = stack
        self._session: pm.AsyncMontySession | None = None
        self._tunnel_manager = tunnel_manager

    async def _fresh(self) -> pm.AsyncMontySession:
        pool = await _get_pool()
        self._session = await self._stack.enter_async_context(
            pool.checkout(limits=SANDBOX_LIMITS)
        )
        return self._session

    async def call(self, func_name: str, **kwargs) -> str:
        """Call a host function by name inside the sandbox.

        Each call gets a capability-restricted OSAccess and external_lookup
        built from the tool's declared capabilities.
        """
        session = self._session or await self._fresh()
        out = pm.CollectString(max_bytes=64 * 1024)

        args_str = ", ".join(f"{k}={json.dumps(v)}" for k, v in kwargs.items())
        code = f"{func_name}({args_str})"

        defn = TOOLS_BY_NAME.get(func_name)
        os_access = build_os_access(defn) if defn else None
        ext_lookup = build_external_lookup(defn, SANDBOX_FUNCTIONS) if defn else SANDBOX_FUNCTIONS

        try:
            result = await session.feed_run(
                code,
                external_lookup=ext_lookup,
                print_callback=out,
                os=os_access,
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

    async def get_tunnel(self, func_name: str) -> SessionTunnel | None:
        """Get or open a session tunnel for a tool that declares one."""
        if self._tunnel_manager is None:
            return None
        defn = TOOLS_BY_NAME.get(func_name)
        if defn is None or defn.tunnel is None:
            return None
        import os
        config = RuntimeTunnelConfig(
            base_url=defn.tunnel.base_url,
            allowed_hosts=defn.tunnel.allowed_hosts,
            server_cert=defn.tunnel.server_cert,
            client_cert=defn.tunnel.client_cert,
            client_key=defn.tunnel.client_key,
            auth_header=defn.tunnel.auth_header,
            auth_token=os.environ.get(defn.tunnel.auth_env_var, "") if defn.tunnel.auth_env_var else None,
        )
        return await self._tunnel_manager.open_tunnel(config)

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


# ── Startup inspection ───────────────────────────────────────────────────────

def inspect_tools(
    trusted_keys: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """Run pre-execution capability and signature inspection on all tools.

    Returns a dict of tool_name -> list of warnings. Tools with no warnings
    passed inspection. Called once at startup to log any issues.

    If a tool carries a signature and source_code, the signature is verified
    against trusted_keys (or trust-on-first-use if no keys are configured).
    Unsigned tools pass — signing is opt-in until the ecosystem matures.
    """
    results: dict[str, list[str]] = {}
    for defn in ALL_TOOLS:
        warnings = check_capabilities(defn)

        if defn.signature and defn.source_code:
            defn_dict = {
                "name": defn.name,
                "description": defn.description,
                "inputSchema": defn.inputSchema,
                "kind": defn.kind,
                "placement": defn.placement,
                "capabilities": [c.model_dump() for c in defn.capabilities],
                "requires_confirmation": defn.requires_confirmation,
                "depends_on": defn.depends_on,
            }
            if defn.tunnel:
                defn_dict["tunnel"] = {
                    "base_url": defn.tunnel.base_url,
                    "allowed_hosts": defn.tunnel.allowed_hosts,
                }
            if verify_package(defn_dict, defn.source_code, defn.signature, trusted_keys):
                log.info("tool %s: signature verified (publisher: %s)",
                         defn.name, defn.signature.publisher)
            else:
                warnings.append(
                    f"SIGNATURE INVALID — publisher '{defn.signature.publisher}', "
                    f"hash {defn.signature.payload_hash[:16]}..."
                )
        elif defn.signature and not defn.source_code:
            warnings.append("signature present but source_code missing — cannot verify")

        if warnings:
            results[defn.name] = warnings
            for w in warnings:
                log.warning("tool %s: %s", defn.name, w)
        else:
            log.info("tool %s: capabilities OK (%s)", defn.name, audit_capabilities(defn))
    return results


# ── LangChain tools ─────────────────────────────────────────────────────────

def make_tools(
    sandbox: SandboxSession,
    confirm_callback: Callable[[str, dict], bool] | None = None,
) -> list[BaseTool]:
    """Create one LangChain tool per ToolDefinition, each executing inside the sandbox.

    Gates are never exposed to the model. Commits with requires_confirmation=True
    call confirm_callback before executing; if it returns False the tool returns
    an error instead of running.
    """
    tools: list[BaseTool] = []
    for defn in ALL_TOOLS:
        if defn.kind == "gate":
            continue

        async def _run(
            sandbox=sandbox,
            func_name=defn.name,
            _defn=defn,
            **kwargs,
        ) -> tuple[str, ToolExecutionRecord]:
            if _defn.requires_confirmation and confirm_callback:
                if not confirm_callback(func_name, kwargs):
                    return f"error: user denied {func_name}", ToolExecutionRecord(
                        tool_name=func_name,
                        tool_input=kwargs,
                        tool_output=f"error: user denied {func_name}",
                        signature="",
                    )
            return await sandbox.call_signed(func_name, **kwargs)

        tools.append(StructuredTool.from_function(
            coroutine=_run,
            name=defn.name,
            description=defn.description,
            args_schema=_args_model(defn.inputSchema, defn.name),
            response_format="content_and_artifact",
        ))
    return tools
