# Streaming Memory

An AI chat agent with long-term memory powered by semantic embeddings and multi-factor retrieval.

## Architecture

```
User Input
    ↓
Query Long-term Memory (semantic + recency + emotion + frequency)
    ↓
Inject relevant memories into context (with timestamps)
    ↓
Stream response with extended thinking (Claude Opus 4.5)
    ↓
Ingest all experiences into cache
    ↓
Background: LLM creates memory blocks from cache
```

## Features

- **Extended Thinking**: Claude Opus 4.5 with visible thinking stream
- **Semantic Search**: OpenAI embeddings (`text-embedding-3-small`)
- **Multi-factor Retrieval**: Memories ranked by composite score
- **First-person Memories**: Agent remembers from its own perspective
- **Emotional Intensity**: Memories tagged with surprise, arousal, control
- **Temporal Awareness**: Agent sees current time and memory ages
- **Save/Load**: Persist and restore memories to/from JSON
- **Benchmarking**: Process existing conversation logs (JSONL format)

## Setup

```bash
# Install dependencies
uv sync

# Set up API keys in .env
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key
```

## Running

```bash
# Start fresh
uv run python chat.py

# Load existing memories
uv run python chat.py --load memories.json

# Auto-save on exit
uv run python chat.py -l memories.json -s memories.json
```

### Commands

| Command | Description |
|---------|-------------|
| `stats` | Show memory statistics |
| `working` | Show working memory (messages array) |
| `memories` | Show all memory blocks |
| `clear` | Clear both memories |
| `clear working` | Clear only working memory |
| `clear ltm` | Clear only long-term memory |
| `save <path>` | Save memories to file |
| `exit` | Quit |

## Benchmarking with Existing Data

Process existing conversation logs to build memories:

```python
from memory import LongTermMemory, load_raw_jsonl

# Load conversation data (JSONL format)
chunks = load_raw_jsonl("conversations.jsonl", limit=500)

# Process into memories (uses Haiku by default - cheap & fast)
ltm = LongTermMemory()
memories = ltm.process_chunk_stream(chunks, batch_size=100)

# Save for later use
ltm.save("memories.json")

# Load and chat
ltm = LongTermMemory.from_file("memories.json")
results = ltm.query("What did the user struggle with?")
```

### JSONL Format

The loader expects JSONL with `message_data` objects:

```json
{"timestamp": "...", "message_data": {"type": "message", "role": "user", "content": "Hello!"}}
{"timestamp": "...", "message_data": {"type": "message", "role": "assistant", "content": "Hi there!"}}
{"timestamp": "...", "message_data": {"type": "function_call", "name": "search", "arguments": "{...}"}}
{"timestamp": "...", "message_data": {"type": "function_call_output", "output": "{...}"}}
```

### Model Configuration

Memory extraction uses Haiku 4.5 by default ($1/M in, $5/M out):

```python
# Default (Haiku - cheap & fast)
ltm = LongTermMemory()

# Or specify model
ltm = LongTermMemory(memory_model="claude-sonnet-4-20250514")  # Balanced
ltm = LongTermMemory(memory_model="claude-opus-4-5-20251101")  # Best quality
```

## Memory System

### What Each Memory Stores

```python
MemoryBlock(
    summary="I learned the user loves hiking in Utah",
    embedding=[...],           # 1536-dim vector
    frequency=3,               # Times surfaced
    created_at=datetime(...),  # When created
    emotions=EmotionIntensity(
        surprise=0.2,          # How unexpected (0-1)
        arousal=0.5,           # How stimulating (0-1)
        control=0.6            # Sense of agency (0-1)
    )
)
```

### Multi-factor Scoring

Memories are ranked by composite score combining:

| Factor | Weight | Function |
|--------|--------|----------|
| **Similarity** | 35% | Cosine similarity of embeddings |
| **Emotion** | 30% | Power law: `x^5` (high emotions weighted exponentially) |
| **Recency** | 25% | Exponential decay: `e^(-t/τ)` where τ=1 hour |
| **Frequency** | 10% | Logarithmic: `log(1+f)/log(1+max)` |

### Recency Decay

| Time Ago | Score |
|----------|-------|
| Just now | 1.00 |
| 30 min | 0.61 |
| 1 hour | 0.37 |
| 2 hours | 0.14 |

### Emotion Power Law

| Avg Emotion | Score |
|-------------|-------|
| 0.5 | 0.03 |
| 0.7 | 0.17 |
| 0.9 | 0.59 |
| 1.0 | 1.00 |

## Memory Creation

After each response, a background process:

1. Takes the cache (user input, thinking, response)
2. Sends to LLM with Pydantic schema (Haiku by default)
3. Extracts first-person memories with emotional ratings
4. Embeds summaries with OpenAI
5. Stores as memory blocks

### Example Memory

```
🧠 NEW MEMORY CREATED:
   Summary: I learned that the user loves the mountains near their house in Utah
   Emotions: surprise=0.20, arousal=0.50, control=0.60
   Created: 11:32:12
```

## Context Injection

Before responding, memories are queried and injected:

```
Current time: 2024-12-09 11:45:32

You have the following relevant memories from past experiences:
- [30s ago] I learned that the user loves the mountains in Utah
- [2m ago] I learned that the user owns a house in Utah
- [5m ago] I was greeted by a user who said 'hi'
```

## API Reference

### LongTermMemory

```python
# Initialize
ltm = LongTermMemory(default_top_k=10, memory_model="claude-haiku-4-5-20251001")

# Ingest content
ltm.ingest(content, entry_type)  # "user_input" | "thinking" | "response_text"
ltm.ingest_batch([(content, type), ...])

# Process cache into memories
ltm.process_cache_async()  # Background (non-blocking)
ltm.process_cache_sync()   # Foreground (blocking, returns memories)

# Query memories
results = ltm.query("search text", top_k=5)  # Returns List[QueryResult]

# Convenience methods for benchmarking
ltm.ingest_interaction(user_input, assistant_response, thinking=None)
ltm.ingest_interactions([{"user": "...", "assistant": "..."}])
ltm.process_chunk_stream(chunks, batch_size=10)

# Persistence
ltm.save("memories.json")
ltm.load("memories.json")
ltm = LongTermMemory.from_file("memories.json")

# Inspection
ltm.get_stats()
ltm.get_all_memories()
ltm.clear()
```

### Loaders

```python
from memory import load_raw_jsonl, load_chunks_from_jsonl, load_interactions_from_jsonl

# Load all entries (production behavior)
chunks = load_raw_jsonl("data.jsonl", limit=500)

# Load filtered by role
chunks = load_chunks_from_jsonl("data.jsonl", include_roles=["user", "assistant"])

# Load as interaction pairs
interactions = load_interactions_from_jsonl("data.jsonl", limit=100)
```
