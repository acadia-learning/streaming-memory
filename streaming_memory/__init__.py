"""
Streaming Memory: Hebbian-inspired dynamic memory system for LLMs.

Provides context injection that adapts based on:
- Semantic relevance
- Learned associations (Hebbian learning)
- Recency of retrieval
- Emotional intensity
- Memory-memory connections

## Quick Start

```python
from streaming_memory import MemoryPool, StreamingMemoryService
from streaming_memory.config import AssistantConfig, MemoryConfig

# Create a config
config = AssistantConfig(
    name="my-assistant",
    system_prompt="You are a helpful assistant...",
    memory=MemoryConfig(memory_file="memories.json"),
)

# Create a pool
pool = MemoryPool(embed_fn=my_embed_fn)

# Create the service
service = StreamingMemoryService(
    config=config,
    pool=pool,
    tokenizer=tokenizer,
    model=model,
    pool_total_tokens=1000,
)

# Stream a response
for event in service.generate_stream("Hello!"):
    print(event.to_sse())
```
"""

from .config import AssistantConfig, MemoryConfig, ModelConfig
from .longmemeval import (
    EvalResult,
    ExtractedMemory,
    LoadedInstance,
    SessionMemories,
    compute_retrieval_metrics,
    load_longmemeval_instance,
    run_streaming_eval,
    summarize_session_to_memories,
)
from .memory import Connection, Memory, MemoryPool, QueryAssociation
from .sample_data import SAMPLE_MEMORIES, load_sample_memories
from .service import StreamEvent, StreamingMemoryService
from .voice_agent import (
    AgentState,
    DeepgramFluxClient,
    Transcript,
    VoiceAgentService,
    VoiceEvent,
    create_voice_agent,
)


def create_app(*args, **kwargs):
    """Lazy import create_app to avoid fastapi dependency when not needed."""
    from .api import create_app as _create_app
    return _create_app(*args, **kwargs)

__version__ = "0.1.0"
__all__ = [
    # Core memory system
    "Memory",
    "MemoryPool",
    "Connection",
    "QueryAssociation",
    # Sample data
    "SAMPLE_MEMORIES",
    "load_sample_memories",
    # Configuration
    "AssistantConfig",
    "MemoryConfig",
    "ModelConfig",
    # Service
    "StreamingMemoryService",
    "StreamEvent",
    # Voice agent
    "VoiceAgentService",
    "VoiceEvent",
    "DeepgramFluxClient",
    "AgentState",
    "Transcript",
    "create_voice_agent",
    # API
    "create_app",
    # LongMemEval integration
    "ExtractedMemory",
    "SessionMemories",
    "summarize_session_to_memories",
    "load_longmemeval_instance",
    "run_streaming_eval",
    "compute_retrieval_metrics",
    "LoadedInstance",
    "EvalResult",
]
