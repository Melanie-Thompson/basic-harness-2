from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field


Assertion = tuple[str, Literal["exact", "numeric", "temporal"]]
Extractor = Callable[[str], list[Assertion]]

FunctionKind = Literal["gate", "fetch", "processor", "commit"]
Placement = Literal["local", "remote"]


class Capability(BaseModel):
    """A single declared capability a tool needs at runtime."""
    kind: Literal["network", "filesystem", "env", "subprocess"]
    target: str = Field(
        description="What the capability grants access to, e.g. a host pattern, path glob, or env var name"
    )


class ToolDefinition(BaseModel):
    """Code-to-Agent tool definition.

    Extends MCP's tool shape with function kind, placement, and declared
    capabilities so the harness can inspect, sandbox, and enforce.

    - gate:      auth/permissions/consent — not exposed to the model
    - fetch:     reads data from the API — client call from sandbox
    - processor: transforms/filters/calculates — ships as code, runs locally
    - commit:    writes/side effects — client call, may require confirmation
    """
    name: str
    description: str
    inputSchema: dict[str, Any] = Field(
        description="JSON Schema object describing the tool's parameters"
    )
    kind: FunctionKind = "processor"
    placement: Placement = "local"
    capabilities: list[Capability] = Field(default_factory=list)
    requires_confirmation: bool = Field(
        default=False,
        description="If true the harness must get user confirmation before executing"
    )
    fn: Callable = Field(exclude=True)
    extractor: Extractor | None = Field(default=None, exclude=True)
    depends_on: list[str] = Field(default_factory=list)
