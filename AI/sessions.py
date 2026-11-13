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
    CausalResult,
    FaithfulnessResult,
    HumanMessage,
    InterstitialCheck,
    Session,
    SystemMessage,
    ToolCall,
    ToolExecution,
    ToolExecutionRecord,
    ToolMessage as ModelToolMessage,
)

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
    """Convert LangChain messages to typed Pydantic models.

    Tool messages are omitted — their data is on the preceding AI message's
    tool_executions. Every AI message is kept: intermediate ones carry
    tool_executions + causality, the final one carries faithfulness.
    """
    result: list[AnyMessage] = []
    for msg in messages:
        role = _TYPE_TO_ROLE.get(msg.type, "human")
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

        if role == "ai":
            kwargs = getattr(msg, "additional_kwargs", {})

            causality = None
            raw = kwargs.get("causality")
            if raw:
                try:
                    causality = CausalResult(**raw)
                except Exception:
                    pass

            faithfulness = None
            raw = kwargs.get("faithfulness")
            if raw:
                try:
                    faithfulness = FaithfulnessResult(**raw)
                except Exception:
                    pass

            interstitial = None
            raw = kwargs.get("interstitial")
            if raw:
                try:
                    interstitial = InterstitialCheck(**raw)
                except Exception:
                    pass

            tool_executions = None
            raw = kwargs.get("tool_executions")
            if raw:
                try:
                    tool_executions = [ToolExecution(**e) for e in raw]
                except Exception:
                    pass

            tool_calls = None
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls = [ToolCall(**tc) for tc in msg.tool_calls]

            result.append(AIMessage(
                content=content,
                tool_calls=tool_calls,
                tool_executions=tool_executions,
                causality=causality,
                interstitial=interstitial,
                faithfulness=faithfulness,
            ))

        elif role == "tool":
            continue

        elif role == "system":
            result.append(SystemMessage(content=content))
        else:
            result.append(HumanMessage(content=content))

    return result


def _model_to_langchain(msg: AnyMessage) -> BaseMessage:
    kwargs: dict = {"content": msg.content}
    if isinstance(msg, AIMessage) and msg.tool_calls:
        kwargs["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
    if isinstance(msg, ModelToolMessage):
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
