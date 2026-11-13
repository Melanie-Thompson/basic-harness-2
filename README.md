# Basic Harness 2

AI agent harness with sandboxed tool execution, cryptographic verification, and semantic faithfulness checking.

## Architecture

```
Client  ->  FastAPI (/chat SSE)  ->  LangChain tool-calling loop  ->  Ollama (qwen3.5:4b)
                                           |
                                     Pydantic Monty sandbox
                                     (Rust, isolated process)
                                           |
                                     Tool functions
                                     (calculator, get_current_time)
```

### Core loop

1. User sends a message to `/chat`
2. The LLM streams a response, optionally requesting tool calls
3. Each tool call executes inside a Pydantic Monty sandbox (Rust-based, memory-limited, time-limited)
4. The sandbox HMAC-signs the execution (tool name + input + output) using `itsdangerous`
5. An inline evaluator verifies the signature before the result enters the conversation
6. After the final AI response, a BERT-based faithfulness checker compares tool outputs against the AI's text
7. The full conversation is persisted to Redis

### Verification pipeline

Two independent checks protect the tool-calling chain:

**HMAC execution verification** (`AI/evaluator.py`, `AI/signing.py`)
- At execution time, the sandbox signs every tool call with a server-side secret
- Before the LLM sees a tool result, the evaluator verifies the signature
- Catches: fabricated tool results, tampered outputs, expired executions (5 min TTL)

**Semantic faithfulness checking** (`AI/faithfulness.py`)
- After the final AI response, BERT sentence embeddings (`all-MiniLM-L6-v2`) compute cosine similarity between tool outputs and the AI's text
- Per-tool extractors provide precision fallback (numeric matching for calculator, temporal matching for time)
- Catches: AI ignoring tool results, contradicting values, hallucinating over outputs
- Configurable: annotate-only (default) or enforce with re-prompting

## Project structure

```
AI/
  agent.py           LLM config, tool-calling loop, stream_reply()
  sandbox.py         Pydantic Monty pool, session management, LangChain tool wrappers
  signing.py         HMAC signing/verification via itsdangerous
  evaluator.py       Inline HMAC verification (deterministic, no LLM)
  faithfulness.py    BERT semantic similarity + per-tool extractor checks
  redis_store.py     Redis connection lifecycle and key/value operations
  sessions.py        LangChain <-> Pydantic model conversion, Redis load/save
  models/
    meta.py          Base model with created_at/updated_at timestamps
    message.py       Typed message hierarchy (Human/AI/Tool/System), ToolExecutionRecord, FaithfulnessResult
    session.py       Session model wrapping a message list
  tools/
    __init__.py      Tool registries (SANDBOX_FUNCTIONS, TOOL_SCHEMAS, extractors)
    calculator.py    Arithmetic expression evaluator
    get_current_time.py  Current time with timezone offset
main.py              FastAPI transport layer with SSE streaming
```

## Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai) running locally with `qwen3.5:4b` pulled
- Redis running on localhost:6379

### Install and run

```bash
# Install dependencies
uv sync

# Pull the model
ollama pull qwen3.5:4b

# Start Redis (Docker or Homebrew)
docker run -d -p 6379:6379 --name redis redis
# or: brew services start redis

# Start the server
uv run uvicorn main:app --reload --port 8001
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `TOOL_HMAC_SECRET` | Random (per-process) | HMAC signing key. Set in production for consistency across restarts |

## API

### POST /chat

```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 17.5% of 2340?"}'
```

Response is an SSE stream:

```
data: {"session_id": "a1b2c3..."}

data: {"delta": "17.5"}
data: {"delta": "% of"}
data: {"delta": " 2,340 is "}
data: {"delta": "409.5"}

data: "[DONE]"
```

### Request body

| Field | Type | Description |
|-------|------|-------------|
| `message` | string | User message |
| `system` | string (optional) | Override the system prompt |

## Tools

### Adding a new tool

1. Create `AI/tools/your_tool.py` with:
   - Input/Output Pydantic models
   - A plain function that takes primitive args and returns a string
   - An extractor function for faithfulness checking

2. Register in `AI/tools/__init__.py`:
   ```python
   SANDBOX_FUNCTIONS["your_tool"] = your_tool
   TOOL_SCHEMAS["your_tool"] = {
       "input": YourToolInput,
       "output": YourToolOutput,
       "extractor": your_extractor,
   }
   ```

The tool automatically gets:
- LangChain integration (exposed to the model)
- Sandbox execution (isolated in Monty)
- HMAC signing (cryptographic proof of execution)
- Faithfulness checking (BERT + extractor verification)

## Session persistence

Sessions are stored in Redis with a 1-hour TTL. Each session contains the full typed message history including:
- Tool call records with HMAC signatures (`ToolExecutionRecord`)
- Faithfulness results on AI messages (`FaithfulnessResult`)
- Structured tool input/output validated through Pydantic schemas

Session key format: `chat:session:{uuid}`

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | HTTP transport |
| `langchain-core`, `langchain-openai` | LLM interaction and tool-calling |
| `pydantic-monty` | Rust-based sandboxed Python interpreter |
| `redis[hiredis]` | Session persistence |
| `itsdangerous` | HMAC-signed timed tokens for tool verification |
| `sentence-transformers` | BERT embeddings for semantic faithfulness |
| `torch` | Tensor operations for embeddings |
| `python-dateutil` | Temporal parsing for time tool extractor |
