"""Package lifecycle — install, inspect, and discard tool packages at runtime.

The lifecycle manages a dynamic tool registry that starts with the built-in
tools and can grow or shrink as packages are installed or discarded during
a session.

Install flow:
    1. Receive a package manifest (definition JSON + source code + signature)
    2. Verify the signature against trusted keys
    3. Compile the source into a callable function
    4. Register the ToolDefinition into the live registry
    5. The tool becomes available to the model on the next turn

Discard flow:
    1. Remove the tool from the live registry
    2. Close any tunnels associated with the tool
    3. The tool disappears from the model's available tools on the next turn

All installed packages are session-scoped — they're discarded when the
session ends. The built-in tools are always present and cannot be discarded.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from AI.capabilities import check_capabilities
from AI.package_signing import PackageSignature, verify_package
from AI.tools.definition import Capability, ToolDefinition, TunnelConfig

log = logging.getLogger("package_lifecycle")


@dataclass
class PackageManifest:
    """Everything needed to install a tool package."""
    definition: dict[str, Any]
    source_code: str
    signature: PackageSignature | None = None
    publisher: str = ""


@dataclass
class InstalledPackage:
    """Record of an installed package."""
    name: str
    publisher: str
    kind: str
    placement: str
    signature_verified: bool
    installed_at: float = field(default_factory=time.time)
    builtin: bool = False
    warnings: list[str] = field(default_factory=list)


class PackageRegistry:
    """Dynamic tool registry with install/inspect/discard lifecycle.

    Wraps the static ALL_TOOLS/TOOLS_BY_NAME/SANDBOX_FUNCTIONS registries
    and adds runtime package management.
    """

    def __init__(self, trusted_keys: dict[str, str] | None = None):
        self._trusted_keys = trusted_keys or {}
        self._installed: dict[str, InstalledPackage] = {}
        self._tools: dict[str, ToolDefinition] = {}
        self._functions: dict[str, callable] = {}
        self._builtin_names: set[str] = set()

    def install(self, manifest: PackageManifest) -> InstalledPackage:
        """Register a tool definition from the API server.

        No code is compiled or executed — the harness is a portal to the API.
        Tool calls are routed to the API's execute endpoint at runtime.
        """
        name = manifest.definition.get("name", "")
        if not name:
            raise ValueError("package definition must have a name")

        if name in self._tools:
            raise ValueError(f"tool '{name}' is already installed — discard it first")

        sig_verified = False
        if manifest.signature:
            if not verify_package(
                manifest.definition,
                manifest.source_code,
                manifest.signature,
                self._trusted_keys or None,
            ):
                raise ValueError(
                    f"signature verification failed for '{name}' "
                    f"(publisher: {manifest.signature.publisher})"
                )
            sig_verified = True
            log.info("package '%s': signature verified (publisher: %s)",
                     name, manifest.signature.publisher)

        defn = manifest.definition
        capabilities = [
            Capability(**c) if isinstance(c, dict) else c
            for c in defn.get("capabilities", [])
        ]

        tunnel_data = defn.get("tunnel")
        tunnel = TunnelConfig(**tunnel_data) if tunnel_data else None

        tool_defn = ToolDefinition(
            name=name,
            description=defn.get("description", ""),
            inputSchema=defn.get("inputSchema", {"type": "object", "properties": {}}),
            kind=defn.get("kind", "processor"),
            placement=defn.get("placement", "remote"),
            capabilities=capabilities,
            tunnel=tunnel,
            requires_confirmation=defn.get("requires_confirmation", False),
            signature=manifest.signature,
            source_code=manifest.source_code,
            fn=lambda **_: None,
            depends_on=defn.get("depends_on", []),
        )

        warnings = check_capabilities(tool_defn)
        if warnings:
            for w in warnings:
                log.warning("package '%s': %s", name, w)

        self._tools[name] = tool_defn
        self._functions[name] = tool_defn.fn
        record = InstalledPackage(
            name=name,
            publisher=manifest.signature.publisher if manifest.signature else manifest.publisher,
            kind=tool_defn.kind,
            placement=tool_defn.placement,
            signature_verified=sig_verified,
            warnings=warnings,
        )
        self._installed[name] = record
        log.info("package '%s' registered (kind=%s, placement=%s, verified=%s)",
                 name, tool_defn.kind, tool_defn.placement, sig_verified)
        return record

    def discard(self, name: str) -> bool:
        """Remove a tool package from the registry.

        Built-in tools cannot be discarded. Returns True if the tool was
        removed, False if it wasn't found.
        """
        if name in self._builtin_names:
            raise ValueError(f"cannot discard built-in tool '{name}'")

        if name not in self._tools:
            return False

        del self._tools[name]
        del self._functions[name]
        del self._installed[name]
        log.info("package '%s' discarded", name)
        return True

    def inspect(self, name: str | None = None) -> list[dict[str, Any]]:
        """Inspect installed packages.

        If name is given, returns info for that one package.
        Otherwise returns info for all installed packages.
        """
        targets = [name] if name else list(self._installed.keys())
        result = []
        for n in targets:
            pkg = self._installed.get(n)
            if pkg is None:
                continue
            defn = self._tools.get(n)
            entry: dict[str, Any] = {
                "name": pkg.name,
                "publisher": pkg.publisher,
                "kind": pkg.kind,
                "placement": pkg.placement,
                "signature_verified": pkg.signature_verified,
                "builtin": pkg.builtin,
                "installed_at": pkg.installed_at,
                "warnings": pkg.warnings,
            }
            if defn:
                entry["capabilities"] = [c.model_dump() for c in defn.capabilities]
                entry["depends_on"] = defn.depends_on
                if defn.tunnel:
                    entry["tunnel"] = {
                        "base_url": defn.tunnel.base_url,
                        "allowed_hosts": defn.tunnel.allowed_hosts,
                    }
            result.append(entry)
        return result

    def discard_all_external(self) -> list[str]:
        """Discard all non-builtin packages. Returns names of discarded tools."""
        external = [n for n in self._tools if n not in self._builtin_names]
        for name in external:
            self.discard(name)
        return external

    @property
    def all_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    @property
    def tools_by_name(self) -> dict[str, ToolDefinition]:
        return dict(self._tools)

    @property
    def sandbox_functions(self) -> dict[str, callable]:
        return dict(self._functions)

    @property
    def installed_count(self) -> int:
        return len(self._tools)

    @property
    def external_count(self) -> int:
        return len(self._tools) - len(self._builtin_names)
