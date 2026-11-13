"""Step faithfulness — checks that each tool call's arguments are consistent
with the preceding tool output the model was reacting to."""

import json

from AI.embeddings import semantic_similarity
from AI.models.message import StepResult, ToolExecutionRecord

STEP_THRESHOLD = 0.25


def check_step_faithfulness(
    records: list[ToolExecutionRecord],
) -> list[StepResult]:
    results: list[StepResult] = []

    for i in range(1, len(records)):
        prev = records[i - 1]
        curr = records[i]

        args_text = json.dumps(curr.tool_input, sort_keys=True)
        prev_output = prev.tool_output

        score = semantic_similarity(prev_output, args_text)

        values_carried = _check_value_carryforward(prev_output, curr.tool_input)

        faithful = values_carried or score >= STEP_THRESHOLD

        detail = ""
        if not faithful:
            detail = (
                f"args {args_text} do not reflect "
                f"preceding output '{prev_output[:100]}'"
            )

        results.append(StepResult(
            step=i,
            tool_called=curr.tool_name,
            preceding_tool=prev.tool_name,
            semantic_score=score,
            faithful=faithful,
            detail=detail,
        ))

    return results


def _check_value_carryforward(output: str, tool_input: dict) -> bool:
    """Check whether any numeric or string value from the output appears in the tool input."""
    input_values = _extract_values(tool_input)
    if not input_values:
        return True

    for val in input_values:
        val_str = str(val)
        if val_str in output:
            return True
        if isinstance(val, (int, float)):
            try:
                if f"{float(val):+.1f}" in output:
                    return True
                if f"{float(val):.1f}" in output:
                    return True
            except (ValueError, TypeError):
                pass

    return False


def _extract_values(d: dict) -> list:
    values = []
    for v in d.values():
        if isinstance(v, dict):
            values.extend(_extract_values(v))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    values.extend(_extract_values(item))
                else:
                    values.append(item)
        else:
            values.append(v)
    return values
