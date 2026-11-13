"""Causal evaluator — static dependency DAG + runtime data-flow validation.

Static layer: tools declare hard `depends_on` constraints (e.g. get_current_time
always requires get_timezone_offset). Violations are flagged when a tool runs
before its declared prerequisites.

Data-flow layer: for ad-hoc chains (e.g. base64_encode after get_current_time),
traces whether each tool's input values came from a preceding tool's output or
from the user's original message. A tool whose inputs can't be traced to any
source is flagged as an ungrounded data-flow gap.
"""

import re

from AI.models.message import CausalResult, DataFlowEdge, ToolExecutionRecord
from AI.tools import TOOL_SCHEMAS


def _extract_values(d: dict) -> list[str]:
    values: list[str] = []
    for v in d.values():
        if isinstance(v, dict):
            values.extend(_extract_values(v))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    values.extend(_extract_values(item))
                else:
                    values.append(str(item))
        else:
            s = str(v)
            if not s:
                continue
            if isinstance(v, (int, float)) or len(s) >= 2:
                values.append(s)
    return values


_WORD_BOUNDARY_CACHE: dict[str, re.Pattern] = {}


def _word_boundary_match(needle: str, text: str) -> bool:
    if needle not in _WORD_BOUNDARY_CACHE:
        _WORD_BOUNDARY_CACHE[needle] = re.compile(
            r"(?<![.\w])" + re.escape(needle) + r"(?![.\w])"
        )
    return _WORD_BOUNDARY_CACHE[needle].search(text) is not None


def _find_value_in_text(value: str, text: str) -> bool:
    try:
        f = float(value)
        for fmt in (f"{f:+.1f}", f"{f:.1f}", f"{f:.0f}", f"{f:g}", value):
            if _word_boundary_match(fmt, text):
                return True
        return False
    except (ValueError, TypeError):
        pass
    if len(value) <= 3:
        return _word_boundary_match(value, text)
    return value in text


def _trace_data_flow(
    records: list[ToolExecutionRecord],
    user_message: str,
) -> tuple[list[DataFlowEdge], list[str]]:
    edges: list[DataFlowEdge] = []
    warnings: list[str] = []

    for i, curr in enumerate(records):
        input_values = _extract_values(curr.tool_input)
        if not input_values:
            continue

        matched_values: dict[str, list[str]] = {}

        for val in input_values:
            if _find_value_in_text(val, user_message):
                matched_values.setdefault("user", []).append(val)
                continue

            found = False
            for prev in records[:i]:
                if _find_value_in_text(val, prev.tool_output):
                    matched_values.setdefault(prev.tool_name, []).append(val)
                    found = True
                    break
            if not found:
                matched_values.setdefault("_untraced", []).append(val)

        for source, vals in matched_values.items():
            if source in ("user", "_untraced"):
                continue
            edges.append(DataFlowEdge(
                from_tool=source,
                to_tool=curr.tool_name,
                values=vals,
            ))

        untraced = matched_values.get("_untraced", [])
        from_user = matched_values.get("user", [])
        from_tools = {k for k in matched_values if k not in ("user", "_untraced")}

        if untraced and not from_user and not from_tools:
            warnings.append(
                f"'{curr.tool_name}' inputs {untraced} not traceable to "
                f"any preceding tool output or user message"
            )

    return edges, warnings


def check_causality(
    records: list[ToolExecutionRecord],
    user_message: str = "",
) -> CausalResult:
    violations: list[str] = []

    # --- static DAG layer ---
    called_before: set[str] = set()
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

    # --- data-flow layer ---
    data_flow, flow_warnings = _trace_data_flow(records, user_message)
    violations.extend(flow_warnings)

    return CausalResult(
        valid=len(violations) == 0,
        violations=violations,
        data_flow=data_flow,
    )
