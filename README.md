# Streaming Memory - Dual Memory Architecture

A chat agent with two memory systems working together:

1. **Working Memory** - Traditional messages array for API calls (short-term)
2. **Long-term Memory** - Background process that ingests all experiences and surfaces relevant memories

## Core Concept

When thinking chunks stream in, they trigger **both** ingestion AND query on long-term memory. If there's a memory hit (relevant past experience), it gets surfaced to the agent.

```
User Input → Working Memory (messages array)
           → Long-term Memory (ingest)

Thinking Stream → Long-term Memory (ingest + query)
               → If hit: Surface memory to agent

Response Stream → Working Memory (append)
               → Long-term Memory (ingest)
```

## Setup

```bash
# Install dependencies
uv sync

# Set up API key in .env
ANTHROPIC_API_KEY=your-key-here
```

## Running

```bash
# Interactive chat with dual memory
uv run python chat.py
```

Commands:

- `stats` - Show both memory statistics
- `working` - Show working memory (messages array)
- `longterm` - Show recent long-term memory entries
- `clear` - Clear both memories
- `exit` - Quit

## Memory Architecture

### Working Memory (`messages: list[dict]`)

Traditional messages array for API calls:

```python
messages = [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
]
```

### Long-term Memory (`LongTermMemory`)

Indexes all experiences for retrieval:

```python
from memory import LongTermMemory

ltm = LongTermMemory(relevance_threshold=0.3)

# Ingest an experience
ltm.ingest("user said something", entry_type="user_input")

# Query for relevant memories
results = ltm.query("related content")

# Ingest AND query in one operation (for thinking chunks)
results = ltm.ingest_and_query(thinking_chunk, entry_type="thinking")
```

### Entry Types

- `user_input` - User messages
- `thinking` - Agent's thinking/reasoning (triggers query)
- `response_text` - Agent's responses

### Query Results

When a query hits, you get:

```python
QueryResult(
    entry=MemoryEntry(...),
    relevance_score=0.65,
    matched_keywords=["python", "learning", "programming"]
)
```

## How Memory Hits Work

1. Thinking chunk streams in: `"I should consider Python programming options..."`
2. Long-term memory indexes it AND queries for related content
3. Query finds relevant past entry: `"user said they love Python"`
4. Memory hit is surfaced: `💡 MEMORY HIT (relevance: 0.65)`
5. Agent can use this context (future: inject into thinking stream)

## Files

| File        | Purpose                                            |
| ----------- | -------------------------------------------------- |
| `memory.py` | `LongTermMemory` class with indexing and retrieval |
| `chat.py`   | Interactive chat with dual memory system           |

## Example Output

```
You: I love playing basketball every weekend.
Assistant: That's great! Basketball is excellent exercise...

You: What sports are good for cardio?
🤔 [thinking] The user mentioned basketball earlier...

💡 MEMORY HIT (relevance: 0.45):
   Type: user_input
   Keywords: basketball, weekend, playing
   Content: I love playing basketball every weekend.
```
