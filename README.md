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
uv run python chat.py
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
| `exit` | Quit |

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
2. Sends to Claude Opus 4.5 with Pydantic schema
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

## Example Session

```
You: hi

🔍 Querying memories... No memories yet.