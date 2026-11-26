"""Effect enforcement — compares observed effects against declared capabilities.

After the API server executes a tool, it reports what the tool actually did
(network calls, file access, env reads). This module checks those reported
effects against the tool's declared capabilities and flags violations.

A violation means the tool did something it didn't declare — the harness
logs the violation and can block the result from reaching the model.
"""

import logging

from AI.tools.definition import ToolDefinition

log = logging.getLogger("effects")


def check_effects(
    tool: ToolDefinition,
    effects: list[dict],
) -> tuple[bool, list[str]]:
    """Check observed effects against declared capabilities.

    Returns (allowed, violations) where allowed is False if any
    undeclared effect was observed.
    """
    if not effects:
        return True, []

    declared_network = set()
    declared_filesystem = set()
    declared_env = set()
    has_subprocess = False

    for cap in tool.capabilities:
        if cap.kind == "network":
            declared_network.add(cap.target)
        elif cap.kind == "filesystem":
            declared_filesystem.add(cap.target)
        elif cap.kind == "env":
            declared_env.add(cap.target)
        elif cap.kind == "subprocess":
            has_subprocess = True

    violations: list[str] = []

    for effect in effects:
        kind = effect.get("kind", "")
        target = effect.get("target", "")
        action = effect.get("action", "")

        if kind == "network":
            if action == "import":
                if not declared_network:
                    violations.append(
                        f"imported network module '{target}' but no network capability declared"
                    )
            elif not _network_match(target, declared_network):
                violations.append(
                    f"network access to '{target}' not in declared hosts: {sorted(declared_network)}"
                )

        elif kind == "filesystem":
            if not _filesystem_match(target, declared_filesystem):
                violations.append(
                    f"filesystem {action} '{target}' not covered by declared paths: {sorted(declared_filesystem)}"
                )

        elif kind == "env":
            if target not in declared_env:
                violations.append(
                    f"env read '{target}' not declared in capabilities"
                )

        elif kind == "subprocess":
            if not has_subprocess:
                violations.append(
                    f"subprocess '{target}' not declared in capabilities"
                )

    allowed = len(violations) == 0
    if not allowed:
        for v in violations:
            log.warning("effect violation [%s]: %s", tool.name, v)
    else:
        log.info("effects OK [%s]: %d effects, all declared", tool.name, len(effects))

    return allowed, violations


def _network_match(target: str, declared: set[str]) -> bool:
    """Check if a network target matches any declared host."""
    if not declared:
        return False
    for host in declared:
        if target == host or target.endswith("." + host):
            return True
    return False


def _filesystem_match(target: str, declared: set[str]) -> bool:
    """Check if a filesystem path matches any declared path pattern."""
    import fnmatch
    if not declared:
        return False
    for pattern in declared:
        if fnmatch.fnmatch(target, pattern):
            return True
    return False
