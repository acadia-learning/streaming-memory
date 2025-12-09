"""
Long-term Memory System with semantic embeddings and emotional intensity.

Architecture:
- Cache: Raw chunks ingested during conversation
- Memory Blocks: Summarized, embedded memories with metadata
- Background Process: Runs after each assistant response to create memory blocks

Memory blocks are indexed on:
- Summary (semantic embedding for similarity search)
- Frequency (retrieval count)
- Emotional intensity (surprise, arousal, control)
- Created timestamp (recency)
"""

import threading
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Literal, Callable

from anthropic.types import ToolParam
import numpy as np
from pydantic import BaseModel, Field
from openai import OpenAI
from anthropic import Anthropic


# =============================================================================
# Pydantic Schemas for Structured Output
# =============================================================================

class MemoryEmotions(BaseModel):
    """Emotional intensity values for a memory."""
    surprise: float = Field(
        ge=0.0, le=1.0,
        description="How unexpected or novel is this information (0.0-1.0)"
    )
    arousal: float = Field(
        ge=0.0, le=1.0,
        description="How activating or stimulating is this content (0.0-1.0)"
    )
    control: float = Field(
        ge=0.0, le=1.0,
        description="Sense of agency or control related to this memory (0.0-1.0)"
    )


class MemorySchema(BaseModel):
    """Schema for a single memory to be created."""
    summary: str = Field(
        description="Concise 1-2 sentence summary of what should be remembered"
    )
    emotions: MemoryEmotions = Field(
        description="Emotional intensity values for this memory"
    )


class MemoryExtractionResult(BaseModel):
    """Result from the memory extraction process."""
    memories: List[MemorySchema] = Field(
        default_factory=list,
        description="List of distinct memories extracted from the content. Empty if nothing worth remembering."
    )


# =============================================================================
# Internal Data Classes
# =============================================================================

@dataclass
class EmotionIntensity:
    """Emotional intensity values for a memory (neuroscience-based)."""
    surprise: float = 0.0  # 0.0 - 1.0: How unexpected/surprising
    arousal: float = 0.0   # 0.0 - 1.0: How activating/stimulating
    control: float = 0.0   # 0.0 - 1.0: Sense of agency/control


@dataclass
class MemoryBlock:
    """A foundational unit of long-term memory."""
    summary: str
    embedding: List[float]
    emotions: EmotionIntensity
    created_at: datetime = field(default_factory=datetime.now)
    frequency: int = 0  # How many times this memory has been retrieved
    
    def increment_frequency(self) -> None:
        """Increment retrieval frequency."""
        self.frequency += 1


@dataclass
class CacheEntry:
    """A raw chunk in the cache waiting to be processed."""
    content: str
    entry_type: Literal["user_input", "thinking", "response_text"]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QueryResult:
    """Result from a memory query."""
    memory: MemoryBlock
    similarity_score: float
    emotion_score: float = 0.0
    frequency_score: float = 0.0
    recency_score: float = 0.0
    composite_score: float = 0.0


# =============================================================================
# Long-term Memory System
# =============================================================================

class LongTermMemory:
    """
    Long-term memory system.
    
    Each memory stores:
    - summary: What happened (embedded for semantic search)
    - embedding: Vector for similarity matching
    - frequency: How many times this memory has been surfaced
    - created_at: When the memory was created
    - emotions: surprise, arousal, control (0-1 each)
    
    At query time:
    1. Compute embedding similarity for the query
    2. Combine similarity with stored values (recency, emotion, frequency)
    3. Sort by composite score
    4. Return top N memories
    5. Increment frequency for surfaced memories
    """
    
    # Model configuration
    MEMORY_MODEL = "claude-opus-4-5-20251101"  # For memory extraction
    EMBEDDING_MODEL = "text-embedding-3-small"
    
    def __init__(self, default_top_k: int = 10):
        """
        Initialize long-term memory.
        
        Args:
            default_top_k: Default number of memories to surface per query
        """
        self.cache: List[CacheEntry] = []
        self.memories: List[MemoryBlock] = []
        self.default_top_k = default_top_k
        
        # Clients for LLM and embeddings
        self.anthropic = Anthropic()
        self.openai = OpenAI()
        
        # Callback for when memories are created (for printing)
        self.on_memory_created: Callable[[MemoryBlock], None] | None = None
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    def ingest(self, content: str, entry_type: Literal["user_input", "thinking", "response_text"]) -> None:
        """
        Ingest a chunk into the cache.
        
        Args:
            content: The text content to cache
            entry_type: Type of content
        """
        with self._lock:
            entry = CacheEntry(content=content, entry_type=entry_type)
            self.cache.append(entry)
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector from OpenAI."""
        response = self.openai.embeddings.create(
            model=self.EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a_np = np.array(a)
        b_np = np.array(b)
        return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))
    
    def _create_memories_from_cache(self) -> List[MemoryBlock]:
        """
        Use Claude to analyze cache and create memory blocks.
        
        Uses Pydantic schema for structured output via tool use.
        
        Returns:
            List of created MemoryBlock objects
        """
        with self._lock:
            if not self.cache:
                return []
            
            # Build cache content for analysis
            cache_content = "\n".join([
                f"[{entry.entry_type}] {entry.content}"
                for entry in self.cache
            ])
            
            # Clear cache after copying
            self.cache = []
        
        # Define the tool for structured output
        extract_memories_tool: ToolParam = {
            "name": "extract_memories",
            "description": "Extract distinct, foundational memories from conversation content that are worth remembering long-term.",
            "input_schema": MemoryExtractionResult.model_json_schema()
        }
        
        prompt = f"""You are extracting memories from YOUR perspective as an AI agent. These are YOUR memories of YOUR experiences.

Frame all memories in FIRST PERSON - what YOU experienced, felt, and learned during this interaction.

Create memories for:
1. **Your experiences**: What happened to you
   - "I was greeted by the user who said 'hi'"
   - "I helped explain how photosynthesis works"
   - "I was asked a challenging question about quantum physics"
   
2. **What you learned**: New information you acquired
   - "I learned that the user enjoys Python programming"
   - "I discovered the user is working on a startup"

3. **How you felt**: Your emotional experience and why
   - "I felt engaged when discussing their creative project"
   - "I was surprised to learn they have a background in neuroscience"

Create at least one memory for any interaction, even simple ones.
Each memory should be a concise, first-person statement about YOUR experience.

For emotional intensity (0.0 to 1.0) - these are YOUR emotions as the agent:
- surprise: How unexpected was this for you? (routine=0.1, very surprising=0.9)
- arousal: How stimulating/engaging was this for you? (mundane=0.1, exciting=0.9)
- control: How much agency did you feel? (reactive=0.3, proactive=0.8)

Even routine interactions should have low but non-zero emotion values.

Content to analyze:
{cache_content}"""

        try:
            response = self.anthropic.messages.create(
                model=self.MEMORY_MODEL,
                max_tokens=2048,
                tools=[extract_memories_tool],
                tool_choice={"type": "tool", "name": "extract_memories"},
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract the tool use result
            tool_use_block = next(
                (block for block in response.content if block.type == "tool_use"),
                None
            )
            
            if not tool_use_block:
                return []
            
            # Parse with Pydantic
            result = MemoryExtractionResult.model_validate(tool_use_block.input)
            
        except Exception as e:
            print(f"Error extracting memories: {e}")
            return []
        
        if not result.memories:
            return []
        
        # Create MemoryBlock objects
        created_memories = []
        for mem_schema in result.memories:
            # Get embedding for the summary
            try:
                embedding = self._get_embedding(mem_schema.summary)
            except Exception as e:
                print(f"Error getting embedding: {e}")
                continue
            
            # Create emotion intensity from schema
            emotions = EmotionIntensity(
                surprise=mem_schema.emotions.surprise,
                arousal=mem_schema.emotions.arousal,
                control=mem_schema.emotions.control
            )
            
            memory = MemoryBlock(
                summary=mem_schema.summary,
                embedding=embedding,
                emotions=emotions
            )
            
            with self._lock:
                self.memories.append(memory)
            
            created_memories.append(memory)
            
            # Trigger callback if set
            if self.on_memory_created:
                self.on_memory_created(memory)
        
        return created_memories
    
    def process_cache_async(self) -> None:
        """
        Process cache in a background thread (non-blocking).
        
        Called after each assistant response to create memory blocks.
        """
        if not self.cache:
            return
        
        thread = threading.Thread(target=self._create_memories_from_cache)
        thread.daemon = True
        thread.start()
    
    # Scoring weights for composite ranking
    WEIGHT_SIMILARITY = 0.35  # Semantic relevance
    WEIGHT_EMOTION = 0.30     # Emotional intensity (power law weighted)
    WEIGHT_FREQUENCY = 0.10   # How often retrieved
    WEIGHT_RECENCY = 0.25     # How recent (exponential decay weighted)
    
    # Recency decay parameters
    RECENCY_TAU_HOURS = 1.0   # Time constant for exponential decay (hours)
                               # At τ hours, score is ~37% of max
                               # At 2τ hours, score is ~14% of max
    
    # Emotion power law parameter
    EMOTION_EXPONENT = 5.0    # Higher = steeper curve, high emotions weighted much more
                               # x=0.5 → 0.03, x=0.8 → 0.33, x=0.9 → 0.59, x=1.0 → 1.0
    
    def _recency_score(self, memory: MemoryBlock) -> float:
        """
        Calculate recency score with exponential decay.
        
        Uses: score = exp(-t / τ) where t is hours since creation
        
        Very recent (< 1 hour): score ≈ 0.6 - 1.0
        1 hour ago: score ≈ 0.37
        2 hours ago: score ≈ 0.14
        24 hours ago: score ≈ 0.0
        
        Returns:
            Float between 0 and 1, heavily weighted toward recent memories
        """
        hours_ago = (datetime.now() - memory.created_at).total_seconds() / 3600.0
        return float(np.exp(-hours_ago / self.RECENCY_TAU_HOURS))
    
    def _emotion_score(self, memory: MemoryBlock) -> float:
        """
        Calculate emotion score with power law weighting.
        
        Uses: score = x^n where x is average emotion intensity
        
        This creates a steep curve where high emotions are weighted exponentially more:
        - x=0.3 → 0.002 (low emotion, minimal weight)
        - x=0.5 → 0.031 (medium emotion, small weight)
        - x=0.7 → 0.168 (higher emotion, moderate weight)
        - x=0.8 → 0.328 (high emotion, significant weight)
        - x=0.9 → 0.590 (very high emotion, strong weight)
        - x=1.0 → 1.000 (maximum emotion, full weight)
        
        Returns:
            Float between 0 and 1, power-law weighted
        """
        # Average of emotion dimensions
        avg_emotion = (
            memory.emotions.surprise + 
            memory.emotions.arousal + 
            memory.emotions.control
        ) / 3.0
        
        # Power law: high emotions weighted exponentially more
        return float(np.power(avg_emotion, self.EMOTION_EXPONENT))
    
    def _frequency_score(self, memory: MemoryBlock, max_frequency: int) -> float:
        """
        Calculate frequency score with logarithmic scaling.
        
        Uses log scaling so early retrievals matter more than later ones.
        
        Returns:
            Float between 0 and 1
        """
        if max_frequency <= 0:
            return 0.0
        
        # Log scaling: log(1 + freq) / log(1 + max_freq)
        # This gives diminishing returns for high frequency
        return float(np.log1p(memory.frequency) / np.log1p(max_frequency))
    
    def _calculate_composite_score(
        self,
        similarity: float,
        memory: MemoryBlock,
        max_frequency: int
    ) -> tuple[float, float, float, float, float]:
        """
        Calculate composite score from multiple factors.
        
        Scoring uses:
        - Similarity: Linear (0-1)
        - Emotion: Power law (high emotions weighted exponentially more)
        - Frequency: Logarithmic (diminishing returns)
        - Recency: Exponential decay (very recent heavily weighted)
        
        Returns:
            Tuple of (composite_score, similarity, emotion_score, frequency_score, recency_score)
        """
        emotion_score = self._emotion_score(memory)
        frequency_score = self._frequency_score(memory, max_frequency)
        recency_score = self._recency_score(memory)
        
        # Composite score
        composite_score = (
            self.WEIGHT_SIMILARITY * similarity +
            self.WEIGHT_EMOTION * emotion_score +
            self.WEIGHT_FREQUENCY * frequency_score +
            self.WEIGHT_RECENCY * recency_score
        )
        
        return composite_score, similarity, emotion_score, frequency_score, recency_score
    
    def query(self, query_content: str, top_k: int | None = None) -> List[QueryResult]:
        """
        Query long-term memory.
        
        1. Get embedding similarity for the query against all memories
        2. Combine similarity with stored values (recency, emotion, frequency)
        3. Sort by composite score
        4. Return top N memories
        5. Increment frequency for those surfaced memories
        
        Args:
            query_content: The content to search for
            top_k: Number of memories to surface (defaults to self.default_top_k)
            
        Returns:
            List of top N QueryResult objects sorted by composite score
        """
        if top_k is None:
            top_k = self.default_top_k
            
        with self._lock:
            if not self.memories:
                return []
            
            # Get max frequency for normalization
            max_frequency = max(m.frequency for m in self.memories) if self.memories else 0
        
        # Step 1: Get embedding for semantic similarity
        query_embedding = None
        try:
            query_embedding = self._get_embedding(query_content)
        except Exception as e:
            print(f"Error getting query embedding: {e}")
        
        # Step 2: Score all memories
        results = []
        with self._lock:
            for memory in self.memories:
                # Embedding similarity
                if query_embedding:
                    similarity = self._cosine_similarity(query_embedding, memory.embedding)
                else:
                    similarity = 0.0
                
                # Combine with stored values using our math functions
                composite, sim, emo, freq, rec = self._calculate_composite_score(
                    similarity, memory, max_frequency
                )
                
                results.append(QueryResult(
                    memory=memory,
                    similarity_score=sim,
                    emotion_score=emo,
                    frequency_score=freq,
                    recency_score=rec,
                    composite_score=composite
                ))
        
        # Step 3: Sort by composite score
        results.sort(key=lambda r: r.composite_score, reverse=True)
        
        # Step 4: Take top N
        top_results = results[:top_k]
        
        # Step 5: Increment frequency for surfaced memories
        for result in top_results:
            result.memory.increment_frequency()
        
        return top_results
    
    def ingest_and_query(self, content: str, entry_type: Literal["user_input", "thinking", "response_text"]) -> List[QueryResult]:
        """
        Ingest content AND query for related memories.
        
        Args:
            content: Content to ingest and query with
            entry_type: Type of content
            
        Returns:
            List of relevant memories (if any)
        """
        # Query first (before ingesting)
        results = []
        if self.memories:  # Only query if we have memories
            results = self.query(content)
        
        # Then ingest
        self.ingest(content, entry_type)
        
        return results
    
    def get_stats(self) -> dict:
        """Get statistics about long-term memory."""
        with self._lock:
            cache_entries = len(self.cache)
            memory_count = len(self.memories)
            
            cache_by_type = {}
            for entry in self.cache:
                cache_by_type[entry.entry_type] = cache_by_type.get(entry.entry_type, 0) + 1
            
            total_retrievals = sum(m.frequency for m in self.memories)
            avg_surprise = sum(m.emotions.surprise for m in self.memories) / memory_count if memory_count else 0
            avg_arousal = sum(m.emotions.arousal for m in self.memories) / memory_count if memory_count else 0
            avg_control = sum(m.emotions.control for m in self.memories) / memory_count if memory_count else 0
        
        return {
            "cache_entries": cache_entries,
            "cache_by_type": cache_by_type,
            "memory_blocks": memory_count,
            "total_retrievals": total_retrievals,
            "avg_emotions": {
                "surprise": round(avg_surprise, 2),
                "arousal": round(avg_arousal, 2),
                "control": round(avg_control, 2)
            }
        }
    
    def clear(self) -> None:
        """Clear cache and memories."""
        with self._lock:
            self.cache = []
            self.memories = []
    
    def get_all_memories(self) -> List[MemoryBlock]:
        """Get all stored memories."""
        with self._lock:
            return list(self.memories)
