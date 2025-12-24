# Streaming Memory

A Hebbian-inspired memory system for LLMs that dynamically retrieves and updates memories during generation. Memories are re-evaluated every token, allowing the model's context to evolve as it thinks.

![Demo](https://img.shields.io/badge/demo-live-brightgreen)
![Modal](https://img.shields.io/badge/backend-Modal-purple)
![Vercel](https://img.shields.io/badge/frontend-Vercel-black)

## Live Demo

**Frontend:** [streaming-memory.vercel.app](https://streaming-memory.vercel.app)

## How It Works

```
User Input
    ↓
Embed query → Retrieve top-k memories
    ↓
Inject memories into LLM context
    ↓
Generate tokens...
    ↓
Every N tokens: re-embed recent output → re-retrieve memories
    ↓
If memories changed: rebuild context, continue generation
    ↓
Hebbian update: strengthen associations between co-retrieved memories
```

### Key Features

- **Dynamic Memory Injection**: Memories are re-queried during generation, not just at the start
- **Hebbian Learning**: Memories that are retrieved together strengthen their association
- **Soft Scoring**: No hard thresholds - semantic similarity gates all other factors
- **Query Diversity**: Prevents "generalist" memories from dominating by normalizing by query diversity
- **Real-time Visualization**: See which memories are active at each token

## Architecture

### Memory Scoring

Each memory is scored by combining:

| Factor | Description |
|--------|-------------|
| **Semantic Similarity** | Cosine similarity² (squared to make it multiplicative) |
| **Recency** | Exponential decay from last retrieval |
| **Frequency** | How often retrieved (normalized by average) |
| **Emotional Intensity** | Surprise/importance factor |
| **Hebbian Associations** | Strength of connections to other retrieved memories |
| **Query Specificity** | Penalizes memories retrieved by many different queries |

```python
# Core scoring formula
emb_sim = cosine_similarity(query, memory) ** 2  # Squared = multiplicative gate
hebbian_boost = recency + frequency + emotion + associations
multiplier = 1 + tanh(hebbian_boost)  # Soft saturation
score = emb_sim * multiplier
```

### Selection

1. Compute scores for all memories
2. Apply softmax with temperature
3. Sample memories weighted by probability
4. Apply diversity penalty (MMR-style) to avoid redundancy
5. Repeat until token budget filled

## Project Structure

```
streaming-memory/
├── streaming_memory/        # Core library
│   ├── memory.py           # MemoryPool, Memory, Connection classes
│   ├── config.py           # Configuration dataclasses
│   ├── service.py          # StreamingMemoryService (platform-agnostic)
│   ├── api.py              # FastAPI app factory
│   ├── longmemeval.py      # LongMemEval benchmark integration
│   └── sample_data.py      # Sample memories for testing
├── scripts/                # Executable scripts
│   ├── benchmarks/         # LongMemEval evaluation scripts
│   └── experiments/        # Experimental scripts
├── deployments/            # Platform-specific deployments
│   ├── modal_base.py       # Shared Modal infrastructure
│   └── family_assistant.py # Family assistant on Modal
├── examples/
│   └── dad_memories.json   # Family assistant memories
├── data/                   # Benchmark data and caches
│   ├── longmemeval_oracle.json
│   └── memory_cache*.json  # Embedding caches
├── frontend/               # React + Vite + Tailwind
│   └── src/Demo.jsx        # Main chat UI
└── pyproject.toml
```

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for Python dependencies
- [Modal](https://modal.com) account for GPU inference
- OpenAI API key (for embeddings)

### Installation

```bash
# Clone
git clone https://github.com/acadia-learning/streaming-memory
cd streaming-memory

# Install Python deps
uv sync
```

### Deploy Backend to Modal

```bash
# Install Modal CLI
pip install modal

# Authenticate
modal setup

# Create secret with OpenAI key
modal secret create openai-secret OPENAI_API_KEY=sk-...

# Deploy family assistant
modal deploy deployments/family_assistant.py
```

### Frontend Development

```bash
cd frontend
yarn install

# Create .env.local with your Modal API URL
echo "VITE_API_URL=https://YOUR_USERNAME--streaming-memory-familyassistant-serve.modal.run" > .env.local

yarn dev
```

### Deploy Frontend to Vercel

```bash
cd frontend
npx vercel --prod
```

**Important:** In Vercel, set the environment variable:
- Go to your Vercel project settings
- Navigate to "Environment Variables"
- Add: `VITE_API_URL` = `https://YOUR_USERNAME--streaming-memory-familyassistant-serve.modal.run`
- Redeploy

## Troubleshooting

### "Failed to fetch" error in production

If you see "Failed to fetch" in production:

1. **Check Modal deployment is running:**
   ```bash
   modal app list
   modal app logs streaming-memory
   ```

2. **Verify API URL is correct:**
   - The URL should match your Modal deployment
   - Check Vercel environment variables are set correctly

3. **Test the API directly:**
   ```bash
   curl https://YOUR_USERNAME--streaming-memory-familyassistant-serve.modal.run/health
   ```

4. **Check browser console:**
   - Open Developer Tools → Console
   - Look for detailed error messages about the connection

5. **Modal cold start:**
   - First request after inactivity takes ~10-30 seconds
   - The health check on page load should wake it up
   - If it times out, try again after a moment

### CORS issues

CORS is configured to allow all origins in `streaming_memory/api.py`. If you still see CORS errors:
- Verify your Modal deployment is using the latest code
- Redeploy with: `modal deploy deployments/family_assistant.py`

## Configuration

### Memory Settings (via UI)

| Setting | Range | Description |
|---------|-------|-------------|
| **Update Frequency** | 1-20 tokens | How often to re-query memories |
| **Max Memories** | 1-15 | Maximum memories in context |

### Pool Parameters

```python
MemoryPool(
    embed_fn=embed,
    softmax_temperature=0.15,    # Lower = more deterministic
    diversity_weight=0.5,        # Penalty for similar memories
    association_weight=0.5,      # Hebbian connection strength
)
```

## API

### POST `/chat/stream`

```json
{
  "message": "Can you help me with fractions?",
  "history": [],
  "update_every_n": 1,
  "max_memories": 5
}
```

Returns Server-Sent Events:

```
data: {"type": "memories", "memories": ["...", "..."]}
data: {"type": "thinking", "t": "Let"}
data: {"type": "thinking", "t": " me"}
data: {"type": "memory_update", "memories": [...], "added": [...], "removed": [...]}
data: {"type": "token", "t": "I"}
data: {"type": "token", "t": "'d"}
data: {"type": "done"}
```

## Memory Format

```json
{
  "content": "I've noticed Alex benefits from step-by-step scaffolding...",
  "type": "what_works",
  "subject": "math",
  "emotional_intensity": 0.5,
  "created_at": "2025-11-11T01:02:35.309703+00:00"
}
```

Types: `what_works`, `pattern`, `current_state`, `insight`, `relationship`, `parent`, `behavior`

## Tech Stack

- **Backend**: Modal (GPU inference), Qwen3-8B, FastAPI
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Frontend**: React 19, Vite, Tailwind CSS, Framer Motion
- **Hosting**: Modal (backend), Vercel (frontend)

## License

MIT
