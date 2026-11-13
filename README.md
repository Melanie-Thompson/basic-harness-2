# Basic Harness 2

AI agent harness with sandboxed tool execution, cryptographic verification, and semantic faithfulness checking.

## Architecture

```
Client  ->  FastAPI (/chat SSE)  ->  LangChain tool-calling loop  ->  Ollama (qwen3.5:4b)
                                           |                                  ^
                                     Pydantic Monty sandbox                   |
                                     (Rust, isolated process)           Context Manager
                                           |                          (TDA clustering)
                                     Tool functions                         |
                                     (calculator, get_current_time)   Redis (archived clusters)
```

### Core loop

1. User sends a message to `/chat`
2. The context manager checks whether conversation history exceeds the token budget
3. If over budget: TDA persistent homology clusters messages by topic, archives least-relevant clusters to Redis, rehydrates archived clusters that match the current query
4. The LLM streams a response, optionally requesting tool calls
5. Each tool call executes inside a Pydantic Monty sandbox (Rust-based, memory-limited, time-limited)
6. The sandbox HMAC-signs the execution (tool name + input + output) using `itsdangerous`
7. An inline evaluator verifies the signature before the result enters the conversation
8. After the final AI response, a BERT-based faithfulness checker compares tool outputs against the AI's text
9. The full conversation is persisted to Redis

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

### TDA context management (`AI/context.py`, `AI/embeddings.py`, `AI/tokenizer.py`)

Manages the LLM's context window using Topological Data Analysis. Messages are embedded with BERT and clustered via Vietoris-Rips persistent homology (gudhi). The persistence diagram's H0 features (connected components) represent topic clusters — the epsilon threshold is chosen adaptively from the largest gap in H0 death times.

**Clustering**: Each turn, all active messages are embedded and fed into a Vietoris-Rips filtration. Connected components at the chosen epsilon form topic clusters. Cluster membership is recomputed every turn — it's a property of the current point cloud, not stored on messages.

**Scoring**: Each cluster's centroid is compared to the current query via cosine similarity. Higher-scoring clusters are more relevant to the user's current question.

**Budget packing**: A greedy knapsack packs the highest-scoring clusters into the token budget (default 32K, configurable via `CONTEXT_MAX_TOKENS` env var). Token counting uses the Qwen tokenizer for accuracy.

**Archival**: Clusters that don't fit are archived to Redis as frozen snapshots containing the centroid embedding, serialized messages, and pre-computed token count. Key format: `context:clusters:{session_id}`.

**Rehydration**: Before clustering, archived cluster centroids are compared to the current query. If cosine similarity exceeds the rehydration threshold (0.5), those clusters' messages rejoin the active set and participate in the new clustering pass. Old topics resurface naturally when the user brings them up again.

## Project structure

```
AI/
  agent.py           LLM config, tool-calling loop, stream_reply()
  context.py         TDA context manager — clustering, scoring, archival, rehydration
  embeddings.py      Shared BERT embedding singleton (all-MiniLM-L6-v2)
  tokenizer.py       Qwen tokenizer singleton for accurate token counting
  sandbox.py         Pydantic Monty pool, session management, LangChain tool wrappers
  signing.py         HMAC signing/verification via itsdangerous
  evaluator.py       Inline HMAC verification (deterministic, no LLM)
  faithfulness.py    BERT semantic similarity + per-tool extractor checks
  redis_store.py     Redis connection lifecycle and key/value operations
  sessions.py        LangChain <-> Pydantic model conversion, Redis load/save
  models/
    meta.py          Base model with created_at/updated_at timestamps
    message.py       Typed message hierarchy (Human/AI/Tool/System), ToolExecutionRecord, FaithfulnessResult
    context.py       ArchivedCluster model for persisted cluster snapshots
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
| `CONTEXT_MAX_TOKENS` | `32768` | Token budget for context window management |
| `TOKENIZER_MODEL` | `Qwen/Qwen3-4B` | HuggingFace model ID for the tokenizer |

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

Archived context clusters are stored separately under `context:clusters:{uuid}` with the same TTL. Each archived cluster contains its centroid embedding, serialized messages, and token count — enough to rehydrate without recomputing embeddings.

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | HTTP transport |
| `langchain-core`, `langchain-openai` | LLM interaction and tool-calling |
| `pydantic-monty` | Rust-based sandboxed Python interpreter |
| `redis[hiredis]` | Session persistence |
| `itsdangerous` | HMAC-signed timed tokens for tool verification |
| `sentence-transformers` | BERT embeddings for faithfulness and context clustering |
| `torch` | Tensor operations for embeddings |
| `gudhi` | Vietoris-Rips persistent homology for TDA context clustering |
| `transformers` | Qwen tokenizer for accurate token counting |
| `scipy` | Connected components for cluster extraction |
| `python-dateutil` | Temporal parsing for time tool extractor |
