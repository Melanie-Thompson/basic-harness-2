"""Session middleware — serialization between LangChain messages and Redis
via the Pydantic models in AI.models."""

import uuid
from datetime import datetime, timezone

from langchain_core.messages import BaseMessage
from langchain_core.messages import AIMessage as LCAIMessage
from langchain_core.messages import HumanMessage as LCHumanMessage
from langchain_core.messages import SystemMessage as LCSystemMessage
from langchain_core.messages import ToolMessage as LCToolMessage

from AI import redis_store
from AI.models import (
    AIMessage,
    AnyMessage,
    FaithfulnessResult,
    HumanMessage,
    Session,
    SystemMessage,
    ToolCall,
    ToolExecutionRecord,
    ToolMessage,
)
from AI.tools import TOOL_SCHEMAS

_TYPE_TO_ROLE: dict[str, str] = {
    "human": "human",
    "ai": "ai",
    "AIMessageChunk": "ai",
    "system": "system",
    "tool": "tool",
}

_ROLE_TO_LC: dict[str, type[BaseMessage]] = {
    "human": LCHumanMessage,
    "ai": LCAIMessage,
    "system": LCSystemMessage,
    "tool": LCToolMessage,
}


def _langchain_to_models(messages: list[BaseMessage]) -> list[AnyMessage]:
    """Convert a list of LangChain messages to typed Pydantic models.

    Processes them together so ToolMessages can look up their corresponding
    AIMessage tool call to populate tool_name, tool_input, and tool_output.
    """
    # Build a map of tool_call_id -> (name, args) from AI messages
    call_index: dict[str, tuple[str, dict]] = {}
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                call_index[tc["id"]] = (tc["name"], tc.get("args", {}))

    result: list[AnyMessage] = []
    for msg in messages:
        role = _TYPE_TO_ROLE.get(msg.type, "human")
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        if role == "ai":
            tool_calls = None
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls = [
                    ToolCall(
                        name=tc["name"],
                        args=tc.get("args", {}),
                        id=tc.get("id"),
                        type=tc.get("type", "tool_call"),
                    )
                    for tc in msg.tool_calls
                ]
            faithfulness = None
            raw_faith = getattr(msg, "additional_kwargs", {}).get("faithfulness")
            if raw_faith:
                try:
                    faithfulness = [FaithfulnessResult(**r) for r in raw_faith]
                except Exception:
                    pass
            result.append(AIMessage(content=content, tool_calls=tool_calls, faithfulness=faithfulness))

        elif role == "tool":
            tool_call_id = msg.tool_call_id if hasattr(msg, "tool_call_id") else None
            tool_name = None
            tool_input = None
            tool_output = None
            execution_record = None

            artifact = getattr(msg, "artifact", None)
            if isinstance(artifact, ToolExecutionRecord):
                execution_record = artifact
            elif isinstance(artifact, dict):
                try:
                    execution_record = ToolExecutionRecord(**artifact)
                except Exception:
                    pass

            if tool_call_id and tool_call_id in call_index:
                name, args = call_index[tool_call_id]
                tool_name = name
                schema = TOOL_SCHEMAS.get(name)
                if schema:
                    tool_input = schema["input"](**args).model_dump()
                    try:
                        tool_output = schema["output"](
                            **{list(schema["output"].model_fields.keys())[0]: content}
                        ).model_dump()
                    except Exception:
                        tool_output = {"raw": content}
                else:
                    tool_input = args
                    tool_output = {"raw": content}

            result.append(ToolMessage(
                content=content,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_output,
                execution_record=execution_record,
            ))

        elif role == "system":
            result.append(SystemMessage(content=content))
        else:
            result.append(HumanMessage(content=content))

    return result


def _model_to_langchain(msg: AnyMessage) -> BaseMessage:
    kwargs: dict = {"content": msg.content}
    if isinstance(msg, AIMessage) and msg.tool_calls:
        kwargs["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
    if isinstance(msg, ToolMessage):
        if msg.tool_call_id:
            kwargs["tool_call_id"] = msg.tool_call_id
        if msg.execution_record:
            kwargs["artifact"] = msg.execution_record

    cls = _ROLE_TO_LC.get(msg.role, LCHumanMessage)
    return cls(**kwargs)


def new_session_id() -> str:
    return uuid.uuid4().hex


async def load(session_id: str) -> list[BaseMessage]:
    raw = await redis_store.get(session_id)
    if raw is None:
        return []
    session = Session.model_validate_json(raw)
    return [_model_to_langchain(m) for m in session.messages]


async def save(session_id: str, messages: list[BaseMessage]) -> None:
    existing_raw = await redis_store.get(session_id)
    if existing_raw:
        existing = Session.model_validate_json(existing_raw)
        created_at = existing.created_at
    else:
        created_at = datetime.now(timezone.utc)

    session = Session(
        session_id=session_id,
        created_at=created_at,
        updated_at=datetime.now(timezone.utc),
        messages=_langchain_to_models(messages),
    )
    await redis_store.set(session_id, session.model_dump_json())
