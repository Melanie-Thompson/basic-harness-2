"""LangChain side: builds the chain, runs a tool-calling loop, and yields text deltas.

Tools execute inside Pydantic Monty — a sandboxed Python interpreter written in
Rust, running in a separate worker process. Each tool defined in AI/tools/ is
exposed to the model as a LangChain tool, but executes inside the sandbox.

pip install langchain-openai pydantic-monty
"""

import json
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack

from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from AI.context import ContextManager
from AI.causality import check_causality
from AI.embeddings import semantic_similarity
from AI.evaluator import VerificationError, verify_tool_execution
from AI.faithfulness import check_faithfulness
from AI.models.message import InterstitialCheck, ToolExecution, ToolExecutionRecord
from AI.sandbox import SandboxSession, close_sandbox, make_tools
from AI.tunnel import TunnelManager

MODEL = "qwen3.5:4b"                          # or "qwen3.5:9b" until 4b is pulled
BASE_URL = "http://localhost:11434/v1"        # LM Studio: http://localhost:1234/v1
API_KEY = "ollama"                            # ignored locally, must be non-empty

MAX_TOOL_ROUNDS = 5                           # stops runaway call loops on small models

SYSTEM_PROMPT = (
    "You are a concise assistant. Always use the provided tools when they can help. "
    "Never answer a question a tool could handle without calling that tool first.\n\n"
    "IMPORTANT: When asked about the time in a specific location, you MUST first call "
    "get_timezone_offset to look up the UTC offset, then call get_current_time with "
    "that offset. Never guess timezone offsets — always look them up first."
)


# ── Model ────────────────────────────────────────────────────────────────────

_llm = ChatOpenAI(
    model=MODEL,
    base_url=BASE_URL,
    api_key=API_KEY,
    temperature=0.2,
    timeout=120,
    max_retries=1,
    reasoning_effort="none",                     # ← kills the chain of thought
)


def _flatten(content) -> str:
    """Normalise AIMessageChunk.content, which is str *or* a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _check_interstitial(
    prev_executions: list[ToolExecution], reasoning: str,
) -> InterstitialCheck:
    outputs = [t.tool_output for t in prev_executions if t.tool_output]
    if not outputs:
        return InterstitialCheck(semantic_score=1.0, faithful=True)
    combined_output = " | ".join(outputs)
    score = semantic_similarity(combined_output, reasoning)
    carried: list[str] = []
    for t in prev_executions:
        for token in t.tool_output.split():
            if len(token) >= 3 and token in reasoning:
                carried.append(token)
                break
    faithful = score >= 0.25 or len(carried) > 0
    detail = ""
    if not faithful:
        detail = "reasoning does not reflect preceding tool outputs"
    return InterstitialCheck(
        semantic_score=score, values_carried=carried, faithful=faithful, detail=detail,
    )


async def _run_tool_call(call: dict, tools_by_name: dict[str, BaseTool]) -> ToolMessage:
    """Execute one parsed tool call, verify HMAC, return ToolMessage."""
    t = tools_by_name.get(call["name"])
    if t is None:
        return ToolMessage(f"error: unknown tool '{call['name']}'", tool_call_id=call["id"])
    try:
        result = await t.ainvoke(call)
        record = getattr(result, "artifact", None)
        if record is not None:
            verify_tool_execution(record)
        return result
    except VerificationError as e:
        return ToolMessage(f"error: verification failed: {e}", tool_call_id=call["id"])
    except Exception as e:  # noqa: BLE001
        return ToolMessage(f"error: {e}", tool_call_id=call["id"])


async def stream_reply(
    message: str,
    system: str = SYSTEM_PROMPT,
    history: list[BaseMessage] | None = None,
    on_tool: Callable[[str, dict, str], None] | None = None,
    turn_messages: list[BaseMessage] | None = None,
    enforce_faithfulness: bool = False,
    session_id: str | None = None,
) -> AsyncIterator[str]:
    """Yield plain text deltas, running sandboxed tool calls between model turns.

    on_tool(name, args, result) is called after each tool runs — handy for
    logging or showing a "🔧 running code…" line in the UI.

    If turn_messages is provided, this turn's new messages (human + AI + tool)
    are appended to it so the caller can persist them.
    """
    managed_history = history or []
    if session_id and managed_history:
        ctx = ContextManager(session_id)
        managed_history = await ctx.prepare(managed_history, message)

    messages: list[BaseMessage] = [SystemMessage(system), *managed_history, HumanMessage(message)]
    new_start = 1 + len(managed_history)
    all_records: list[ToolExecutionRecord] = []
    prev_executions: list[ToolExecution] = []

    def _export():
        if turn_messages is not None:
            turn_messages.extend(messages[new_start:])

    async with AsyncExitStack() as stack:
        tunnel_mgr = TunnelManager(session_id or "anonymous")
        stack.push_async_callback(tunnel_mgr.close_all)
        sandbox = SandboxSession(stack, tunnel_manager=tunnel_mgr)
        tools = make_tools(sandbox)
        tools_by_name = {t.name: t for t in tools}
        llm = _llm.bind_tools(tools)

        for _round_num in range(1, MAX_TOOL_ROUNDS + 1):
            full: AIMessageChunk | None = None

            async for chunk in llm.astream(messages):
                full = chunk if full is None else full + chunk

                text = _flatten(chunk.content)
                if not text:
                    text = chunk.additional_kwargs.get("reasoning_content", "")
                if text:
                    yield text

            if full is None:
                _export()
                return

            bad = [
                ToolMessage(f"error: could not parse arguments: {c.get('error')}", tool_call_id=c["id"])
                for c in full.invalid_tool_calls if c.get("id")
            ]

            # ── Final response (no tool calls) ──────────────────────────
            if not full.tool_calls and not bad:
                if all_records:
                    ai_text = _flatten(full.content)
                    causal = check_causality(all_records, user_message=message)
                    full.additional_kwargs["causality"] = causal.model_dump()

                    faith_by_id = check_faithfulness(ai_text, all_records)
                    if faith_by_id:
                        combined = next(iter(faith_by_id.values()))
                        if len(faith_by_id) > 1:
                            scores = [r.semantic_score for r in faith_by_id.values()]
                            all_contained = all(r.contained for r in faith_by_id.values())
                            all_mismatches = [m for r in faith_by_id.values() for m in r.mismatches]
                            from AI.models.message import FaithfulnessResult
                            combined = FaithfulnessResult(
                                semantic_score=sum(scores) / len(scores),
                                contained=all_contained,
                                faithful=all_contained,
                                mismatches=all_mismatches,
                            )
                        full.additional_kwargs["faithfulness"] = combined.model_dump()

                    unfaithful = any(not r.faithful for r in faith_by_id.values())
                    if unfaithful and enforce_faithfulness:
                        mismatches = [m for r in faith_by_id.values() for m in r.mismatches]
                        messages.append(full)
                        messages.append(HumanMessage(
                            f"Your response does not accurately reflect the tool output. "
                            f"Mismatches: {mismatches}. Please correct your answer."
                        ))
                        all_records.clear()
                        prev_executions.clear()
                        continue

                messages.append(full)
                _export()
                return

            # ── Intermediate round (has tool calls) ─────────────────────
            reasoning = _flatten(full.content)

            messages.append(full)
            messages.extend(bad)

            round_records: list[ToolExecutionRecord] = []
            round_executions: list[ToolExecution] = []
            for call in full.tool_calls:
                result = await _run_tool_call(call, tools_by_name)
                record = getattr(result, "artifact", None)
                if isinstance(record, ToolExecutionRecord):
                    record.tool_call_id = call.get("id")
                    all_records.append(record)
                    round_records.append(record)
                    round_executions.append(ToolExecution(
                        tool_name=record.tool_name,
                        tool_call_id=record.tool_call_id,
                        tool_input=record.tool_input,
                        tool_output=record.tool_output,
                        signature=record.signature,
                    ))
                if on_tool:
                    on_tool(call["name"], call["args"], str(result.content))
                messages.append(result)

            # Causality for this round's tools (cumulative records up to now)
            causal = check_causality(all_records, user_message=message)
            full.additional_kwargs["causality"] = causal.model_dump()
            full.additional_kwargs["tool_executions"] = [
                e.model_dump() for e in round_executions
            ]

            # Interstitial: does this reasoning reflect the previous round's outputs?
            if prev_executions and reasoning:
                interstitial = _check_interstitial(prev_executions, reasoning)
                full.additional_kwargs["interstitial"] = interstitial.model_dump()

            prev_executions = round_executions

        yield "\n[stopped: too many tool calls]"
        _export()


# ── Quick manual test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio

    async def _main():
        def log(name, args, result):
            print(f"\n  🔧 {name}({json.dumps(args)}) → {result}\n")

        try:
            async for delta in stream_reply(
                "What's 17.5% of 2,340, and what time is it in UTC?", on_tool=log
            ):
                print(delta, end="", flush=True)
            print()
        finally:
            await close_sandbox()

    asyncio.run(_main())
