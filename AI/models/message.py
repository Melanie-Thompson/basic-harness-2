from typing import Annotated, Literal

from pydantic import BaseModel, Discriminator, Field, Tag

from .meta import Meta


class ToolCall(BaseModel):
    name: str
    args: dict = Field(default_factory=dict)
    id: str | None = None
    type: str = "tool_call"


class ToolExecutionRecord(BaseModel):
    tool_name: str
    tool_input: dict
    tool_output: str
    signature: str


class FaithfulnessResult(BaseModel):
    tool_name: str
    semantic_score: float
    contained: bool
    faithful: bool
    mismatches: list[str] = Field(default_factory=list)


class Message(Meta):
    role: str
    content: str


class HumanMessage(Message):
    role: Literal["human"] = "human"


class AIMessage(Message):
    role: Literal["ai"] = "ai"
    tool_calls: list[ToolCall] | None = None
    faithfulness: list[FaithfulnessResult] | None = None


class ToolMessage(Message):
    role: Literal["tool"] = "tool"
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_input: dict | None = None
    tool_output: dict | None = None
    execution_record: ToolExecutionRecord | None = None


class SystemMessage(Message):
    role: Literal["system"] = "system"


def _message_discriminator(v: dict | Message) -> str:
    if isinstance(v, dict):
        return v.get("role", "human")
    return v.role


AnyMessage = Annotated[
    Annotated[HumanMessage, Tag("human")]
    | Annotated[AIMessage, Tag("ai")]
    | Annotated[ToolMessage, Tag("tool")]
    | Annotated[SystemMessage, Tag("system")],
    Discriminator(_message_discriminator),
]
