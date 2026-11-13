from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    expression: str = Field(description="Arithmetic expression, e.g. '(3 + 4) * 2'")


class CalculatorOutput(BaseModel):
    result: str = Field(description="Evaluated result or error message")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '(3 + 4) * 2'."""
    allowed = set("0123456789+-*/(). %")
    if not set(expression) <= allowed:
        return "error: only digits and + - * / ( ) % . are allowed"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"
