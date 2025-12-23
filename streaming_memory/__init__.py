"""
Streaming Memory: Hebbian-inspired dynamic memory system for LLMs.

Provides context injection that adapts based on:
- Semantic relevance
- Learned associations (Hebbian learning)
- Recency of retrieval
- Emotional intensity
- Memory-memory connections
"""

from .memory import Memory, MemoryPool, Connection, QueryAssociation
from .sample_data import SAMPLE_MEMORIES, load_sample_memories

__version__ = "0.1.0"
__all__ = [
    "Memory",
    "MemoryPool", 
    "Connection",
    "QueryAssociation",
    "SAMPLE_MEMORIES",
    "load_sample_memories",
]


