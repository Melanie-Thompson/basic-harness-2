"""Tool registry — empty by default, populated from the API server at runtime."""

from typing import Any

from .definition import Assertion, Capability, Extractor, FunctionKind, Placement, ToolDefinition, TunnelConfig

ALL_TOOLS: list[ToolDefinition] = []

TOOLS_BY_NAME: dict[str, ToolDefinition] = {}


def list_tools() -> list[dict[str, Any]]:
    """Return tool definitions in Code-to-Agent format (gates excluded)."""
    tools = []
    for t in ALL_TOOLS:
        if t.kind == "gate":
            continue
        entry = {
            "name": t.name,
            "description": t.description,
            "inputSchema": t.inputSchema,
            "kind": t.kind,
            "placement": t.placement,
            "capabilities": [c.model_dump() for c in t.capabilities],
            "requires_confirmation": t.requires_confirmation,
        }
        if t.tunnel:
            entry["tunnel"] = {
                "base_url": t.tunnel.base_url,
                "allowed_hosts": t.tunnel.allowed_hosts,
            }
        tools.append(entry)
    return tools


SANDBOX_FUNCTIONS: dict[str, callable] = {}
TOOL_SCHEMAS: dict[str, dict] = {}
