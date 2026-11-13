from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, Field, Tag, model_serializer

from .meta import Meta


class ToolCall(BaseModel):
    name: str
    args: dict = Field(default_factory=dict)
    id: str | None = None
    type: str = "tool_call"


class ToolExecutionRecord(BaseModel):
    tool_name: str
    tool_call_id: str | None = None
    tool_input: dict
    tool_output: str
    signature: str


class FaithfulnessResult(BaseModel):
    semantic_score: float
    contained: bool
    faithful: bool
    mismatches: list[str] = Field(default_factory=list)


class ToolExecution(BaseModel):
    tool_name: str
    tool_call_id: str | None = None
    tool_input: dict = Field(default_factory=dict)
    tool_output: str = ""
    signature: str = ""
    faithfulness: FaithfulnessResult | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> dict[str, Any]:
        d = handler(self)
        if not self.faithfulness:
            d.pop("faithfulness", None)
        return d


class DataFlowEdge(BaseModel):
    from_tool: str
    to_tool: str
    values: list[str] = Field(default_factory=list)


class CausalResult(BaseModel):
    valid: bool
    violations: list[str] = Field(default_factory=list)
    data_flow: list[DataFlowEdge] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> dict[str, Any]:
        d = handler(self)
        if not self.violations:
            d.pop("violations", None)
        if not self.data_flow:
            d.pop("data_flow", None)
        return d


class InterstitialCheck(BaseModel):
    semantic_score: float
    values_carried: list[str] = Field(default_factory=list)
    faithful: bool
    detail: str = ""

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> dict[str, Any]:
        d = handler(self)
        if not self.values_carried:
            d.pop("values_carried", None)
        if not self.detail:
            d.pop("detail", None)
        return d


class Message(Meta):
    role: str
    content: str


class HumanMessage(Message):
    role: Literal["human"] = "human"


class AIMessage(Message):
    role: Literal["ai"] = "ai"
    tool_calls: list[ToolCall] | None = None
    tool_executions: list[ToolExecution] | None = None
    causality: CausalResult | None = None
    interstitial: InterstitialCheck | None = None
    faithfulness: FaithfulnessResult | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> dict[str, Any]:
        d = handler(self)
        for key in ("tool_calls", "tool_executions", "causality", "interstitial", "faithfulness"):
            if not getattr(self, key, None):
                d.pop(key, None)
        return d


class ToolMessage(Message):
    role: Literal["tool"] = "tool"
    tool_call_id: str | None = None
    tool_name: str | None = None
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
