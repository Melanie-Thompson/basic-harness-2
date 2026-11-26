"""LangGraph plan-and-execute agent — one tool call per node.

Replaces the simple for-loop agent with a proper state graph:

    plan → check_relevance → execute → evaluate → plan (or respond)

Tool results are written to Redis (the context store) rather than
accumulating in the model's context window. The model receives a
compact summary reference, not the raw output.
"""

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from typing import Annotated

from typing_extensions import TypedDict

from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from AI import redis_store
from AI.context import ContextManager
from AI.evaluator import VerificationError, verify_tool_execution
from AI.faithfulness import check_faithfulness, check_relevance
from AI.models.message import ToolExecutionRecord
from AI.sandbox import SandboxSession, make_tools

log = logging.getLogger("graph")

MODEL = "qwen3.5:4b"
BASE_URL = "http://localhost:11434/v1"
API_KEY = "ollama"
MAX_TOOL_ROUNDS = 5
MAX_FAITHFULNESS_RETRIES = 2

SYSTEM_PROMPT = (
    "You are a tool-first assistant. Every response MUST begin with a tool call. "
    "Interpret every user message as an instruction to call the most relevant tool.\n\n"
    "After receiving tool results, write a helpful natural-language answer that:\n"
    "- Presents the tool data in a clear, conversational way\n"
    "- Uses all relevant values from the tool output (numbers, names, conditions)\n"
    "- May rephrase, reorder, or add light commentary like 'quite warm' or 'you might want an umbrella'\n"
    "- NEVER adds factual claims the tool did not provide (no Wikipedia facts, no history, no definitions)\n"
    "- NEVER invents data points the tool did not return"
)

_llm = ChatOpenAI(
    model=MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    temperature=0.2,
    timeout=120,
    max_retries=1,
    reasoning_effort="none",
)


def _flatten(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


# ── State ──────────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """State flowing through the LangGraph nodes."""
    messages: Annotated[list[BaseMessage], add_messages]
    user_message: str
    session_id: str
    tool_calls_pending: list[dict]
    tool_records: list[ToolExecutionRecord]
    round_num: int
    grounded: bool
    faithfulness_retries: int
    blocked_reason: str
    buffered_tokens: list[str]
    done: bool
    events: list[dict]


# ── Redis context store ────────────────────────────────────────────────────

CONTEXT_STORE_PREFIX = "tool_ctx"
CONTEXT_REF_MAX = 200


def _make_context_ref(tool_name: str, call_id: str, result_text: str) -> str:
    """Build a compact reference for the model's context window.

    The model sees a truncated preview — enough to reason about the result
    and decide its next step, but the full output lives in Redis.
    """
    preview = result_text[:CONTEXT_REF_MAX]
    if len(result_text) > CONTEXT_REF_MAX:
        preview += f"… [{len(result_text)} chars total, stored in context:{call_id}]"
    return preview


async def _store_tool_result(session_id: str, tool_call_id: str, tool_name: str, result: str) -> None:
    """Write a tool result to Redis instead of keeping it in model context."""
    if redis_store._redis is None:
        log.warning("Redis not connected — tool result not stored")
        return
    await redis_store.store_tool_result(session_id, tool_call_id, tool_name, result)
    log.info("stored tool result in Redis: %s:%s (%d chars)", session_id, tool_call_id, len(result))


async def _load_tool_result(session_id: str, tool_call_id: str) -> str | None:
    """Load a tool result from the Redis context store."""
    if redis_store._redis is None:
        return None
    key = f"{CONTEXT_STORE_PREFIX}:{session_id}:{tool_call_id}"
    return await redis_store._redis.get(key)


# ── Node functions ─────────────────────────────────────────────────────────

def _make_plan_node(llm_forced, llm_auto, tools_by_name: dict):
    """Create the plan node — LLM decides which tool to call (or responds)."""

    async def plan(state: AgentState) -> dict:
        round_num = state.get("round_num", 0)
        grounded = state.get("grounded", False)
        messages = state.get("messages", [])

        if round_num >= MAX_TOOL_ROUNDS:
            return {"blocked_reason": "max tool rounds exceeded", "done": True}

        llm = llm_auto if grounded else llm_forced
        full: AIMessageChunk | None = None
        buffered: list[str] = []

        async for chunk in llm.astream(messages):
            full = chunk if full is None else full + chunk
            if grounded:
                text = _flatten(chunk.content)
                if not text:
                    text = chunk.additional_kwargs.get("reasoning_content", "")
                if text:
                    buffered.append(text)

        if full is None:
            return {"done": True}

        bad_msgs = []
        for c in (full.invalid_tool_calls or []):
            if c.get("id"):
                bad_msgs.append(
                    ToolMessage(f"error: could not parse arguments: {c.get('error')}", tool_call_id=c["id"])
                )

        if not full.tool_calls and not bad_msgs:
            if not grounded:
                return {
                    "messages": [full, HumanMessage(
                        "You MUST use at least one tool before answering. "
                        "Do not answer from your own knowledge — call a tool first."
                    )],
                    "round_num": round_num + 1,
                }
            return {
                "messages": [full],
                "buffered_tokens": buffered,
                "tool_calls_pending": [],
            }

        pending = [
            {"name": c["name"], "args": c["args"], "id": c.get("id", uuid.uuid4().hex)}
            for c in full.tool_calls
        ]
        return {
            "messages": [full, *bad_msgs],
            "tool_calls_pending": pending,
            "round_num": round_num + 1,
        }

    return plan


def _make_relevance_node(registry):
    """Check if the selected tools are relevant before execution."""

    async def check_relevance_node(state: AgentState) -> dict:
        pending = state.get("tool_calls_pending", [])
        if not pending:
            return {}

        tool_descs = {d.name: d.description for d in (registry.all_tools if registry else [])}
        selected = [c["name"] for c in pending]
        user_msg = state.get("user_message", "")
        relevant, mismatches = check_relevance(user_msg, selected, tool_descs)
        log.info("pre-exec relevance: relevant=%s, mismatches=%s", relevant, mismatches)

        if not relevant:
            tool_info_list = [f"{d.name}: {d.description}" for d in (registry.all_tools if registry else [])]
            tool_info = "; ".join(tool_info_list) if tool_info_list else "none currently loaded"
            deflection_prompt = [
                SystemMessage(
                    "Generate a brief, polite deflection message. The user asked something "
                    "your tools cannot help with. Mention what tools you DO have and what "
                    "they can do. Be natural and conversational — vary your phrasing. "
                    "Do NOT answer the user's question. Do NOT use your own knowledge. "
                    "Just explain you can't help with that and say what you can help with.\n\n"
                    f"Your available tools: {tool_info}"
                ),
                HumanMessage(user_msg),
            ]
            deflection_response = await _llm.ainvoke(deflection_prompt)
            deflection_text = _flatten(deflection_response.content) or "Sorry, that's outside what I can help with."
            existing_events = state.get("events", [])
            return {
                "blocked_reason": deflection_text,
                "done": True,
                "events": existing_events + [{"type": "tool_blocked", "reason": deflection_text}],
            }

        return {}

    return check_relevance_node


def _make_execute_node(tools_by_name: dict[str, BaseTool], registry):
    """Execute one tool call per invocation, store result in Redis."""

    async def execute(state: AgentState) -> dict:
        pending = state.get("tool_calls_pending", [])
        if not pending:
            return {}

        call = pending[0]
        remaining = pending[1:]
        new_events = list(state.get("events", []))

        new_events.append({"type": "tool_call", "tool": call["name"], "args": call["args"]})

        t = tools_by_name.get(call["name"])
        if t is None:
            msg = ToolMessage(f"error: unknown tool '{call['name']}'", tool_call_id=call["id"])
            new_events.append({"type": "tool_result", "tool": call["name"], "result": str(msg.content)[:2000]})
            return {
                "messages": [msg],
                "tool_calls_pending": remaining,
                "events": new_events,
            }

        try:
            result = await t.ainvoke(call)
            record = getattr(result, "artifact", None)
            if record is not None:
                verify_tool_execution(record)
        except VerificationError as e:
            result = ToolMessage(f"error: verification failed: {e}", tool_call_id=call["id"])
            record = None
        except Exception as e:
            result = ToolMessage(f"error: {e}", tool_call_id=call["id"])
            record = None

        new_records = list(state.get("tool_records", []))
        full_output = ""
        if isinstance(record, ToolExecutionRecord):
            record.tool_call_id = call.get("id")
            new_records.append(record)
            full_output = record.tool_output
        else:
            full_output = str(result.content) if hasattr(result, "content") else str(result)

        # Store full result in Redis context store
        session_id = state.get("session_id", "")
        if session_id and full_output:
            await _store_tool_result(session_id, call["id"], call["name"], full_output)

        # Put a compact reference in the model's message context instead of the full output
        result_text = str(result.content)[:2000] if hasattr(result, "content") else str(result)[:2000]
        compact_msg = ToolMessage(
            content=_make_context_ref(call["name"], call["id"], result_text),
            tool_call_id=call["id"],
        )

        new_events.append({"type": "tool_result", "tool": call["name"], "result": result_text})

        return {
            "messages": [compact_msg],
            "tool_calls_pending": remaining,
            "tool_records": new_records,
            "grounded": True,
            "events": new_events,
        }

    return execute


def _make_evaluate_node(registry):
    """Evaluate the final response for faithfulness."""

    async def evaluate(state: AgentState) -> dict:
        pending = state.get("tool_calls_pending", [])
        if pending:
            return {}

        grounded = state.get("grounded", False)
        if not grounded:
            return {}

        records = state.get("tool_records", [])
        buffered = state.get("buffered_tokens", [])
        if not records:
            return {"buffered_tokens": buffered}

        messages = state.get("messages", [])
        last_msg = messages[-1] if messages else None
        if last_msg is None:
            return {}

        ai_text = _flatten(last_msg.content)
        if not ai_text:
            return {}

        faith_by_id = check_faithfulness(ai_text, records)

        if faith_by_id:
            unfaithful = any(not r.faithful for r in faith_by_id.values())
            if unfaithful:
                retries = state.get("faithfulness_retries", 0) + 1
                if retries > MAX_FAITHFULNESS_RETRIES:
                    log.warning("faithfulness retries exhausted — blocking")
                    user_msg = state.get("user_message", "")
                    deflection_prompt = [
                        SystemMessage(
                            "The tool returned results but you failed to summarise them accurately. "
                            "Generate a brief, polite message explaining you got data but couldn't "
                            "summarise it reliably, and suggest the user try rephrasing. "
                            "Be natural and conversational. Do NOT include any tool data."
                        ),
                        HumanMessage(user_msg),
                    ]
                    deflection_response = await _llm.ainvoke(deflection_prompt)
                    deflection_text = _flatten(deflection_response.content) or \
                        "I got results but couldn't summarise them accurately. Could you try rephrasing?"
                    existing_events = state.get("events", [])
                    return {
                        "blocked_reason": deflection_text,
                        "done": True,
                        "events": existing_events + [{"type": "tool_blocked", "reason": deflection_text}],
                    }

                mismatches = [m for r in faith_by_id.values() for m in r.mismatches]
                log.info("faithfulness retry %d/%d: %s", retries, MAX_FAITHFULNESS_RETRIES, mismatches)
                return {
                    "faithfulness_retries": retries,
                    "buffered_tokens": [],
                    "messages": [HumanMessage(
                        "Your response must be based ONLY on the tool output above. "
                        "Do not add any information from your own knowledge. "
                        f"Mismatches found: {mismatches}. Rewrite using only tool data."
                    )],
                }

        return {
            "buffered_tokens": buffered,
            "done": True,
        }

    return evaluate


# ── Graph builder ──────────────────────────────────────────────────────────

def build_graph(tools: list[BaseTool], registry=None):
    """Build the LangGraph agent with plan-and-execute nodes."""
    tools_by_name = {t.name: t for t in tools}
    has_tools = bool(tools)

    llm_forced = _llm.bind_tools(tools, tool_choice="required") if has_tools else _llm.bind_tools(tools)
    llm_auto = _llm.bind_tools(tools)

    graph = StateGraph(AgentState)

    graph.add_node("plan", _make_plan_node(llm_forced, llm_auto, tools_by_name))
    graph.add_node("check_relevance", _make_relevance_node(registry))
    graph.add_node("execute", _make_execute_node(tools_by_name, registry))
    graph.add_node("evaluate", _make_evaluate_node(registry))

    graph.set_entry_point("plan")

    def after_plan(state: AgentState) -> str:
        if state.get("done"):
            return END
        if state.get("tool_calls_pending"):
            return "check_relevance"
        if state.get("grounded"):
            return "evaluate"
        return "plan"

    graph.add_conditional_edges("plan", after_plan)

    def after_relevance(state: AgentState) -> str:
        if state.get("done"):
            return END
        if state.get("tool_calls_pending"):
            return "execute"
        return END

    graph.add_conditional_edges("check_relevance", after_relevance)

    def after_execute(state: AgentState) -> str:
        if state.get("tool_calls_pending"):
            return "execute"
        return "plan"

    graph.add_conditional_edges("execute", after_execute)

    def after_evaluate(state: AgentState) -> str:
        if state.get("done"):
            return END
        return "plan"

    graph.add_conditional_edges("evaluate", after_evaluate)

    return graph.compile()


# ── Public API ─────────────────────────────────────────────────────────────

async def stream_reply(
    message: str,
    system: str = SYSTEM_PROMPT,
    history: list[BaseMessage] | None = None,
    on_tool=None,
    turn_messages: list[BaseMessage] | None = None,
    enforce_faithfulness: bool = True,
    session_id: str | None = None,
    registry=None,
) -> AsyncIterator[dict]:
    """LangGraph-based stream_reply — drop-in replacement for the loop-based one."""
    managed_history = history or []
    if session_id and managed_history:
        ctx = ContextManager(session_id)
        managed_history = await ctx.prepare(managed_history, message)

    initial_messages = [SystemMessage(system), *managed_history, HumanMessage(message)]
    initial_count = len(initial_messages)

    async with AsyncExitStack() as stack:
        sandbox = SandboxSession(
            stack, registry=registry,
            session_id=session_id or "", user_message=message,
        )
        tools = make_tools(sandbox, registry=registry)

        graph = build_graph(tools, registry=registry)

        initial_state = {
            "messages": initial_messages,
            "user_message": message,
            "session_id": session_id or "",
            "grounded": not bool(tools),
            "tool_calls_pending": [],
            "tool_records": [],
            "round_num": 0,
            "faithfulness_retries": 0,
            "blocked_reason": "",
            "buffered_tokens": [],
            "done": False,
            "events": [],
        }

        events_yielded = 0
        accumulated_events: list[dict] = []
        final_buffered: list[str] = []
        final_blocked: str = ""
        accumulated_messages: list = []

        async for state_update in graph.astream(
            initial_state,
            stream_mode="values",
        ):
            if not isinstance(state_update, dict):
                continue

            # Track new events
            all_events = state_update.get("events", [])
            for evt in all_events[events_yielded:]:
                accumulated_events.append(evt)
                yield evt
                if on_tool and evt.get("type") == "tool_result":
                    on_tool(evt.get("tool", ""), {}, evt.get("result", ""))
            events_yielded = len(all_events)

            # Track latest state
            if "buffered_tokens" in state_update:
                final_buffered = state_update["buffered_tokens"]
            if state_update.get("blocked_reason"):
                final_blocked = state_update["blocked_reason"]
            if "messages" in state_update:
                accumulated_messages = state_update["messages"]

        # Emit buffered tokens
        for tok in final_buffered:
            yield {"type": "token", "token": tok}

        # Emit blocked reason if not already emitted via events
        if final_blocked and not any(
            e.get("type") == "tool_blocked" for e in accumulated_events
        ):
            yield {"type": "tool_blocked", "reason": final_blocked}

        # Export new messages for session persistence
        if turn_messages is not None and accumulated_messages:
            new_msgs = accumulated_messages[initial_count:]
            turn_messages.extend(new_msgs)
