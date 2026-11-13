from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel

from .calculator import CalculatorInput, CalculatorOutput, calculator
from .get_current_time import GetCurrentTimeInput, GetCurrentTimeOutput, get_current_time

Assertion = tuple[str, Literal["exact", "numeric", "temporal"]]
Extractor = Callable[[str], list[Assertion]]

SANDBOX_FUNCTIONS: dict[str, Callable] = {
    "get_current_time": get_current_time,
    "calculator": calculator,
}


def _calculator_extractor(output: str) -> list[Assertion]:
    if output.startswith("error:"):
        return [(output, "exact")]
    return [(output, "numeric")]


def _time_extractor(output: str) -> list[Assertion]:
    return [(output, "temporal")]


TOOL_SCHEMAS: dict[str, dict] = {
    "get_current_time": {
        "input": GetCurrentTimeInput,
        "output": GetCurrentTimeOutput,
        "extractor": _time_extractor,
    },
    "calculator": {
        "input": CalculatorInput,
        "output": CalculatorOutput,
        "extractor": _calculator_extractor,
    },
}
