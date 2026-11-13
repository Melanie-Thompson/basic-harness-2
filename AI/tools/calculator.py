from pydantic import BaseModel, Field

from .definition import ToolDefinition


class CalculatorInput(BaseModel):
    expression: str = Field(description="Arithmetic expression, e.g. '(3 + 4) * 2'")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression. Supports +, -, *, /, %, parentheses."""
    allowed = set("0123456789+-*/(). %")
    if not set(expression) <= allowed:
        return "error: only digits and + - * / ( ) % . are allowed"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def _extractor(output: str):
    if output.startswith("error:"):
        return [(output, "exact")]
    return [(output, "numeric")]


TOOL = ToolDefinition(
    name="calculator",
    description="Evaluate a basic arithmetic expression. Supports +, -, *, /, %, parentheses.",
    inputSchema=CalculatorInput.model_json_schema(),
    fn=calculator,
    extractor=_extractor,
)
