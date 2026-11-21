"""Tool registry — collects Code-to-Agent ToolDefinitions from each module."""

from typing import Any

from .definition import Assertion, Capability, Extractor, FunctionKind, Placement, ToolDefinition, TunnelConfig
from .calculator import TOOL as _calculator
from .dice_roller import TOOL as _dice_roller
from .get_current_time import TOOL as _get_current_time
from .get_timezone_offset import TOOL as _get_timezone_offset
from .string_tools import HASH_TOOL as _hash_string, BASE64_TOOL as _base64_encode
from .unit_converter import TOOL as _unit_converter
from .word_counter import TOOL as _word_counter

ALL_TOOLS: list[ToolDefinition] = [
    _get_timezone_offset,
    _get_current_time,
    _calculator,
    _unit_converter,
    _word_counter,
    _dice_roller,
    _hash_string,
    _base64_encode,
]

TOOLS_BY_NAME: dict[str, ToolDefinition] = {t.name: t for t in ALL_TOOLS}


def list_tools() -> list[dict[str, Any]]:
    """Return tool definitions in Code-to-Agent format.

    Extends MCP's tools/list with kind, placement, capabilities, and
    requires_confirmation so the harness can inspect and enforce.
    """
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


# Backwards-compat aliases used by sandbox.py, causality.py, faithfulness.py
SANDBOX_FUNCTIONS: dict[str, callable] = {t.name: t.fn for t in ALL_TOOLS}
TOOL_SCHEMAS: dict[str, dict] = {
    t.name: {
        "inputSchema": t.inputSchema,
        "extractor": t.extractor,
        "depends_on": t.depends_on,
        "kind": t.kind,
    }
    for t in ALL_TOOLS
}
