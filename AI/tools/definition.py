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


class TunnelConfig(BaseModel):
    """Connection details for the session tunnel to an external API.

    Shipped with the tool package. The harness uses these to open a
    private, mutually authenticated channel scoped to one session.
    """
    base_url: str = Field(description="API base URL")
    allowed_hosts: list[str] = Field(
        default_factory=list,
        description="Hostnames reachable through this tunnel (empty = base_url host only)"
    )
    server_cert: str | None = Field(
        default=None, description="Path to server CA cert for verification"
    )
    client_cert: str | None = Field(
        default=None, description="Path to client cert for mTLS"
    )
    client_key: str | None = Field(
        default=None, description="Path to client key for mTLS"
    )
    auth_header: str | None = Field(
        default=None, description="Header name for auth token (e.g. 'Authorization')"
    )
    auth_env_var: str | None = Field(
        default=None,
        description="Env var holding the auth token (resolved at tunnel open, never shipped)"
    )


class ToolDefinition(BaseModel):
    """Code-to-Agent tool definition.

    Extends MCP's tool shape with function kind, placement, declared
    capabilities, and tunnel config so the harness can inspect, sandbox,
    and enforce.

    - gate:      auth/permissions/consent — not exposed to the model
    - fetch:     reads data from the API — client call through session tunnel
    - processor: transforms/filters/calculates — ships as code, runs locally
    - commit:    writes/side effects — client call through session tunnel, may require confirmation
    """
    name: str
    description: str
    inputSchema: dict[str, Any] = Field(
        description="JSON Schema object describing the tool's parameters"
    )
    kind: FunctionKind = "processor"
    placement: Placement = "local"
    capabilities: list[Capability] = Field(default_factory=list)
    tunnel: TunnelConfig | None = Field(
        default=None,
        description="Connection details for the session tunnel (fetch/commit tools that call an external API)"
    )
    requires_confirmation: bool = Field(
        default=False,
        description="If true the harness must get user confirmation before executing"
    )
    signature: Any = Field(
        default=None,
        exclude=True,
        description="PackageSignature from the publisher (verified at load time)"
    )
    source_code: str | None = Field(
        default=None,
        exclude=True,
        description="Source code of the tool (used for signature verification)"
    )
    fn: Callable = Field(exclude=True)
    extractor: Extractor | None = Field(default=None, exclude=True)
    depends_on: list[str] = Field(default_factory=list)
