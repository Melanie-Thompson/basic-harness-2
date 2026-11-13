from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field


Assertion = tuple[str, Literal["exact", "numeric", "temporal"]]
Extractor = Callable[[str], list[Assertion]]


class ToolDefinition(BaseModel):
    """MCP-conformant tool definition.

    See https://spec.modelcontextprotocol.io/latest/server/tools/
    """
    name: str
    description: str
    inputSchema: dict[str, Any] = Field(
        description="JSON Schema object describing the tool's parameters"
    )
    fn: Callable = Field(exclude=True)
    extractor: Extractor | None = Field(default=None, exclude=True)
    depends_on: list[str] = Field(default_factory=list)
