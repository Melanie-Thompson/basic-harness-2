"""Causal evaluator — verifies tool calls were made in the correct dependency order."""

from AI.models.message import CausalResult, ToolExecutionRecord
from AI.tools import TOOL_SCHEMAS


def check_causality(records: list[ToolExecutionRecord]) -> CausalResult:
    tool_sequence = [r.tool_name for r in records]
    called_before: set[str] = set()
    violations: list[str] = []

    for record in records:
        schema = TOOL_SCHEMAS.get(record.tool_name)
        if schema is None:
            called_before.add(record.tool_name)
            continue

        deps = schema.get("depends_on", [])
        for dep in deps:
            if dep not in called_before:
                violations.append(
                    f"'{record.tool_name}' called before its dependency '{dep}'"
                )

        called_before.add(record.tool_name)

    return CausalResult(
        valid=len(violations) == 0,
        tool_sequence=tool_sequence,
        violations=violations,
    )
