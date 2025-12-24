"""
Core Hebbian memory system for LLM context injection.

Key principles:
1. Associations are LEARNED through co-activation, not just embedding similarity
2. NO THRESHOLDS - everything is continuous and smooth
3. Multiple factors: relevance, recency, emotion, repetition, connections
4. Self-stabilizing: prevents runaway where few memories dominate
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable
import numpy as np


@dataclass
class Memory:
    """A single memory unit with Hebbian properties."""
    
    id: str
    content: str
    embedding: np.ndarray
    
    # Hebbian properties
    retrieval_count: int = 0
    last_retrieved: Optional[datetime] = None
    emotional_intensity: float = 0.5  # 0-1, represents surprise/salience
    created_at: datetime = field(default_factory=datetime.now)
    
    def token_estimate(self) -> int:
        """Rough token count (~4 chars per token)."""
        return len(self.content) // 4 + 1


@dataclass 
class Connection:
    """Link between two memories, strengthened by co-retrieval."""
    
    co_retrieval_count: int = 0
    last_co_retrieval: Optional[datetime] = None
    
    def strength(self, now: datetime, decay_hours: float = 168) -> float:
        """Connection strength with recency decay."""
        base = np.log1p(self.co_retrieval_count)
        if self.last_co_retrieval:
            hours = (now - self.last_co_retrieval).total_seconds() / 3600
            recency = np.exp(-hours / decay_hours)
        else:
            recency = 0.5
        return base * recency


@dataclass
class QueryAssociation:
    """
    Learned association between a query pattern and a memory.
    
    This is the key Hebbian component - we learn that certain queries
    lead to certain memories being relevant, regardless of embedding similarity.
    """
    retrieval_count: int = 0
    last_retrieved: Optional[datetime] = None
    query_embeddings: list = field(default_factory=list)
    max_stored_queries: int = 10
    
    def add_query(self, query_embedding: np.ndarray):
        """Record a query that led to this memory."""
        self.query_embeddings.append(query_embedding)
        if len(self.query_embeddings) > self.max_stored_queries:
            self.query_embeddings.pop(0)
        self.retrieval_count += 1
        self.last_retrieved = datetime.now()
    
    def association_strength(
        self, 
        query_embedding: np.ndarray, 
        decay_hours: float = 72
    ) -> float:
        """
        How strongly is this query associated with this memory?
        
        KEY: Normalized by query DIVERSITY. If a memory is retrieved by
        many different queries, it's a "generalist" and gets less boost.
        """
        if not self.query_embeddings:
            return 0.0
        
        # Find max similarity to any stored query pattern
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        max_sim = 0.0
        for stored_emb in self.query_embeddings:
            stored_norm = stored_emb / (np.linalg.norm(stored_emb) + 1e-8)
            sim = float(np.dot(query_norm, stored_norm))
            max_sim = max(max_sim, sim)
        
        # Query diversity: how varied are the queries that led here?
        if len(self.query_embeddings) > 1:
            total_sim = 0.0
            count = 0
            for i, emb_a in enumerate(self.query_embeddings):
                norm_a = emb_a / (np.linalg.norm(emb_a) + 1e-8)
                for emb_b in self.query_embeddings[i+1:]:
                    norm_b = emb_b / (np.linalg.norm(emb_b) + 1e-8)
                    total_sim += float(np.dot(norm_a, norm_b))
                    count += 1
            specificity = total_sim / count if count > 0 else 1.0
        else:
            specificity = 1.0
        
        # Scale by retrieval count with diminishing returns
        count_factor = np.log1p(self.retrieval_count) / 3
        
        # Recency decay
        if self.last_retrieved:
            hours = (datetime.now() - self.last_retrieved).total_seconds() / 3600
            recency = np.exp(-hours / decay_hours)
        else:
            recency = 0.5
        
        return max_sim * count_factor * recency * specificity


class MemoryPool:
    """
    Pool of memories with true Hebbian learning.
    
    Score components (all continuous, no thresholds):
    1. Embedding similarity (base semantic relevance)
    2. Learned query associations (Hebbian: "hobbies" → photography)
    3. Memory-memory connections (co-retrieval strengthens links)
    4. Recency (recently retrieved memories are more accessible)
    5. Emotional intensity (salient memories are stronger)
    6. Repetition (frequently retrieved memories are stronger)
    """
    
    def __init__(
        self,
        embed_fn: Callable[[str], np.ndarray],
        # Weights for each score component
        embedding_weight: float = 1.0,
        association_weight: float = 0.5,
        connection_weight: float = 0.3,
        recency_weight: float = 0.2,
        emotion_weight: float = 0.15,
        repetition_weight: float = 0.1,
        # Decay parameters
        recency_decay_hours: float = 24,
        # Selection parameters
        diversity_weight: float = 0.3,
        softmax_temperature: float = 0.5,
    ):
        self.embed_fn = embed_fn
        self.memories: dict[str, Memory] = {}
        self.connections: dict[tuple[str, str], Connection] = {}
        self.query_associations: dict[str, QueryAssociation] = {}
        
        # Weights
        self.embedding_weight = embedding_weight
        self.association_weight = association_weight
        self.connection_weight = connection_weight
        self.recency_weight = recency_weight
        self.emotion_weight = emotion_weight
        self.repetition_weight = repetition_weight
        self.recency_decay_hours = recency_decay_hours
        self.diversity_weight = diversity_weight
        self.softmax_temperature = softmax_temperature
        
        # Cache for fast similarity computation
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._id_list: list[str] = []
        self._cache_valid = False
    
    def add(
        self, 
        content: str, 
        emotional_intensity: float = 0.5,
        memory_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
    ) -> Memory:
        """Add a memory to the pool."""
        if memory_id is None:
            memory_id = f"mem_{len(self.memories)}"
        
        embedding = self.embed_fn(content)
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        memory = Memory(
            id=memory_id,
            content=content,
            embedding=embedding,
            emotional_intensity=emotional_intensity,
            created_at=created_at or datetime.now(),
        )
        self.memories[memory_id] = memory
        self.query_associations[memory_id] = QueryAssociation()
        self._cache_valid = False
        return memory
    
    def _rebuild_cache(self):
        """Rebuild the embeddings matrix for fast similarity."""
        self._id_list = list(self.memories.keys())
        if self._id_list:
            self._embeddings_matrix = np.stack([
                self.memories[mid].embedding for mid in self._id_list
            ])
        else:
            self._embeddings_matrix = None
        self._cache_valid = True
    
    def _compute_embedding_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """Compute cosine similarities between query and all memories."""
        if not self._cache_valid:
            self._rebuild_cache()
        
        if self._embeddings_matrix is None:
            return np.array([])
        
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        mem_norms = self._embeddings_matrix / (
            np.linalg.norm(self._embeddings_matrix, axis=1, keepdims=True) + 1e-8
        )
        
        return mem_norms @ query_norm
    
    def _recency_factor(self, memory: Memory, now: datetime) -> float:
        """
        Compute recency factor with exponential decay.
        
        Uses last_retrieved if available, otherwise uses created_at (event date).
        This ensures memories about recent events rank higher even if never retrieved.
        """
        # Use event date (created_at) if never retrieved, otherwise use last_retrieved
        reference_time = memory.last_retrieved or memory.created_at
        hours = (now - reference_time).total_seconds() / 3600
        return float(np.exp(-hours / self.recency_decay_hours))
    
    def _repetition_factor(self, memory: Memory) -> float:
        """Compute repetition factor with log scaling."""
        if not self.memories:
            return 0.0
        avg_count = sum(m.retrieval_count for m in self.memories.values()) / len(self.memories)
        if avg_count == 0:
            return 0.5 if memory.retrieval_count > 0 else 0.0
        return float(np.log1p(memory.retrieval_count) / np.log1p(avg_count + 1))
    
    def _compute_scores(
        self, 
        query_embedding: np.ndarray, 
        now: datetime,
        recently_selected: list[str] = None,
    ) -> dict[str, float]:
        """
        Compute scores for all memories.
        
        KEY: Embedding similarity is MULTIPLICATIVE - it gates everything else.
        """
        similarities = self._compute_embedding_similarities(query_embedding)
        recently_selected = recently_selected or []
        
        scores = {}
        for i, mem_id in enumerate(self._id_list):
            memory = self.memories[mem_id]
            
            # 1. Base embedding similarity (GATES everything else)
            raw_sim = similarities[i]
            emb_sim = raw_sim ** 2  # Square to amplify differences
            
            # 2. Learned query associations (THE HEBBIAN PART)
            assoc = self.query_associations[mem_id]
            assoc_boost = assoc.association_strength(query_embedding) * self.association_weight
            
            # 3. Connection boost from recently selected memories
            conn_boost = 0.0
            for selected_id in recently_selected:
                conn = self._get_connection(mem_id, selected_id, create=False)
                if conn:
                    conn_boost += conn.strength(now) * self.connection_weight
            
            # 4. Recency
            recency_boost = self._recency_factor(memory, now) * self.recency_weight
            
            # 5. Emotional intensity
            emotion_boost = memory.emotional_intensity * self.emotion_weight
            
            # 6. Repetition
            rep_boost = self._repetition_factor(memory) * self.repetition_weight
            
            # MULTIPLICATIVE: relevance gates all boosts
            total_boost = assoc_boost + conn_boost + recency_boost + emotion_boost + rep_boost
            hebbian_multiplier = 1 + np.tanh(total_boost)
            
            total = float(emb_sim * self.embedding_weight * hebbian_multiplier)
            scores[mem_id] = total
        
        return scores
    
    def _get_connection(self, id_a: str, id_b: str, create: bool = True) -> Optional[Connection]:
        """Get or create connection between two memories."""
        key = tuple(sorted([id_a, id_b]))
        if key not in self.connections:
            if create:
                self.connections[key] = Connection()
            else:
                return None
        return self.connections[key]
    
    def retrieve(
        self,
        query: str,
        token_budget: int = 3000,
        max_memories: int = 10,
        now: Optional[datetime] = None,
    ) -> list[Memory]:
        """
        Retrieve memories using continuous Hebbian scoring.
        
        Selectivity comes from:
        1. max_memories limit
        2. Softmax concentration
        3. Diversity penalty
        """
        if not self.memories:
            return []
        
        now = now or datetime.now()
        query_embedding = self.embed_fn(query)
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        
        # Compute initial scores
        scores = self._compute_scores(query_embedding, now)
        
        # Convert to probabilities via softmax
        score_array = np.array([scores[mid] for mid in self._id_list])
        probs = self._softmax(score_array / self.softmax_temperature)
        prob_dict = {mid: p for mid, p in zip(self._id_list, probs)}
        
        # Greedy selection with diversity penalty
        selected: list[Memory] = []
        selected_ids: list[str] = []
        tokens_used = 0
        remaining = set(self._id_list)
        
        while remaining and tokens_used < token_budget and len(selected) < max_memories:
            # Recompute with connection boost from selected
            if selected_ids:
                scores = self._compute_scores(query_embedding, now, selected_ids)
                remaining_list = list(remaining)
                score_array = np.array([scores[mid] for mid in remaining_list])
                probs = self._softmax(score_array / self.softmax_temperature)
                prob_dict = {mid: p for mid, p in zip(remaining_list, probs)}
            
            # Apply diversity penalty
            adjusted_probs = {}
            for mem_id in remaining:
                prob = prob_dict.get(mem_id, 0)
                if selected:
                    mem = self.memories[mem_id]
                    max_sim = max(
                        self._cosine_sim(mem.embedding, s.embedding)
                        for s in selected
                    )
                    diversity_factor = float(np.exp(-max_sim * self.diversity_weight * 3))
                    prob *= diversity_factor
                adjusted_probs[mem_id] = prob
            
            if not adjusted_probs:
                break
            
            # Select best
            best_id = max(adjusted_probs.keys(), key=lambda k: adjusted_probs[k])
            best_mem = self.memories[best_id]
            
            # Check token budget
            mem_tokens = best_mem.token_estimate()
            if tokens_used + mem_tokens > token_budget:
                remaining.remove(best_id)
                continue
            
            selected.append(best_mem)
            selected_ids.append(best_id)
            tokens_used += mem_tokens
            remaining.remove(best_id)
        
        # Update Hebbian associations
        self._update_on_retrieval(selected, query_embedding, now)
        
        return selected
    
    def _update_on_retrieval(
        self, 
        selected: list[Memory], 
        query_embedding: np.ndarray,
        now: datetime
    ):
        """Update Hebbian associations after retrieval."""
        for mem in selected:
            mem.retrieval_count += 1
            mem.last_retrieved = now
            self.query_associations[mem.id].add_query(query_embedding.copy())
        
        # Strengthen memory-memory connections
        for i, mem_a in enumerate(selected):
            for mem_b in selected[i+1:]:
                conn = self._get_connection(mem_a.id, mem_b.id)
                conn.co_retrieval_count += 1
                conn.last_co_retrieval = now
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    
    def get_stats(self) -> dict:
        """Get pool statistics."""
        if not self.memories:
            return {"memories": 0}
        
        retrieval_counts = [m.retrieval_count for m in self.memories.values()]
        
        return {
            "memories": len(self.memories),
            "connections": len(self.connections),
            "avg_retrieval_count": float(np.mean(retrieval_counts)),
            "max_retrieval_count": max(retrieval_counts),
            "total_retrievals": sum(retrieval_counts),
            "learned_associations": sum(
                1 for a in self.query_associations.values() 
                if a.retrieval_count > 0
            ),
        }


