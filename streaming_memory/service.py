"""
Core streaming memory service.

Platform-agnostic implementation of streaming generation with dynamic memory retrieval.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generator, Protocol

from .config import AssistantConfig
from .memory import MemoryPool


class Tokenizer(Protocol):
    """Protocol for tokenizer interface."""

    eos_token_id: int

    def encode(self, text: str, return_tensors: str | None = None) -> Any: ...
    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str: ...
    def apply_chat_template(
        self, messages: list[dict], tokenize: bool = False, add_generation_prompt: bool = False
    ) -> str: ...


class Model(Protocol):
    """Protocol for model interface."""

    device: Any

    def generate(self, input_ids: Any, **kwargs) -> Any: ...


@dataclass
class StreamEvent:
    """An event yielded during streaming generation."""

    type: str
    data: dict

    def to_sse(self) -> str:
        """Convert to Server-Sent Event format."""
        return f"data: {json.dumps({'type': self.type, **self.data})}\n\n"


class StreamingMemoryService:
    """
    Core service for streaming generation with dynamic memory retrieval.

    This class is platform-agnostic and can be used with any tokenizer/model
    that implements the required protocols.
    """

    def __init__(
        self,
        config: AssistantConfig,
        pool: MemoryPool,
        tokenizer: Tokenizer,
        model: Model,
        pool_total_tokens: int,
    ):
        self.config = config
        self.pool = pool
        self.tokenizer = tokenizer
        self.model = model
        self.pool_total_tokens = pool_total_tokens

    def _format_memories(self, memories: list) -> str:
        """Format memories for inclusion in prompt."""
        if not memories:
            return ""
        lines = [self.config.memory.memory_prefix]
        for mem in memories:
            lines.append(f"- {mem.content}")
        return "\n".join(lines)

    def generate_stream(
        self,
        message: str,
        history: list[dict] | None = None,
        update_every_n: int = 1,
        max_memories: int = 5,
        lookback_tokens: int = 60,
    ) -> Generator[StreamEvent, None, None]:
        """
        Generate a response with streaming memory updates.

        Yields StreamEvent objects that can be converted to SSE format.

        Args:
            message: User's input message
            history: Previous conversation history
            update_every_n: Re-retrieve memories every N tokens
            max_memories: Maximum memories to include in context
            lookback_tokens: Number of recent tokens to use for re-retrieval
        """
        import torch

        history = history or []

        time.time()
        yield StreamEvent('timing', {'stage': 'init', 'ms': 0})

        # Build query from history + message
        query = message
        if history:
            recent = " ".join([m["content"] for m in history[-4:]])
            query = recent + " " + message

        # Initial retrieval
        t1 = time.time()
        memories = self.pool.retrieve(query, max_memories=max_memories)
        yield StreamEvent('timing', {'stage': 'embed', 'ms': int((time.time() - t1) * 1000)})

        current_mem_contents = [m.content for m in memories]
        yield StreamEvent('memories', {'memories': current_mem_contents})

        # Track unique memories and token counts
        all_unique_memories = set(current_mem_contents)
        memory_token_cache: dict[str, int] = {}

        def get_memory_tokens(mem_content: str) -> int:
            if mem_content not in memory_token_cache:
                memory_token_cache[mem_content] = len(self.tokenizer.encode(mem_content))
            return memory_token_cache[mem_content]

        for mem in current_mem_contents:
            get_memory_tokens(mem)

        # Calculate base context size (without memories)
        base_messages = [{"role": "system", "content": self.config.system_prompt}]
        for h in history[-6:]:
            base_messages.append(h)
        base_messages.append({"role": "user", "content": message})
        base_text = self.tokenizer.apply_chat_template(
            base_messages, tokenize=False, add_generation_prompt=True
        )
        base_context_size = len(self.tokenizer.encode(base_text))

        # Build full prompt with memories
        memory_context = self._format_memories(memories)
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "system", "content": memory_context},
        ]
        for h in history[-6:]:
            messages.append(h)
        messages.append({"role": "user", "content": message})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.model.device)

        max_tokens = self.config.model.max_tokens
        all_tokens: list[int] = []
        in_thinking = False
        timeline: list[dict] = []
        token_idx = 0
        token_history: list[dict] = []

        # Track initial context size
        current_memory_tokens = sum(get_memory_tokens(m) for m in current_mem_contents)
        rag_memory_tokens = sum(get_memory_tokens(m) for m in all_unique_memories)

        token_history.append({
            'token_idx': 0,
            'streaming': base_context_size + current_memory_tokens,
            'rag': base_context_size + rag_memory_tokens,
            'all': base_context_size + self.pool_total_tokens,
            'generated': 0,
        })

        yield StreamEvent('context_size', {
            'base_context_size': base_context_size,
            'current_memory_tokens': current_memory_tokens,
            'rag_memory_tokens': rag_memory_tokens,
            'all_memories_tokens': self.pool_total_tokens,
            'unique_memories': len(all_unique_memories),
            'token_history': token_history,
        })

        with torch.no_grad():
            current_ids = input_ids

            while len(all_tokens) < max_tokens:
                chunk_size = min(update_every_n, max_tokens - len(all_tokens))

                t_gen = time.time()
                outputs = self.model.generate(
                    current_ids,
                    max_new_tokens=chunk_size,
                    do_sample=True,
                    temperature=self.config.model.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                gen_ms = int((time.time() - t_gen) * 1000)

                new_token_ids = outputs[0, current_ids.shape[1]:].tolist()

                for tid in new_token_ids:
                    all_tokens.append(tid)
                    token_text = self.tokenizer.decode([tid], skip_special_tokens=False)

                    if '<think>' in token_text:
                        in_thinking = True
                        continue
                    elif '</think>' in token_text:
                        in_thinking = False
                        continue

                    if in_thinking:
                        timeline.append({
                            'idx': token_idx,
                            'token': token_text,
                            'type': 'thinking',
                            'memories': current_mem_contents.copy()
                        })
                        token_idx += 1
                        yield StreamEvent('thinking', {'t': token_text})
                    else:
                        clean_token = self.tokenizer.decode([tid], skip_special_tokens=True)
                        if clean_token:
                            timeline.append({
                                'idx': token_idx,
                                'token': clean_token,
                                'type': 'response',
                                'memories': current_mem_contents.copy()
                            })
                            token_idx += 1
                            yield StreamEvent('token', {'t': clean_token})

                    if tid == self.tokenizer.eos_token_id:
                        break

                if self.tokenizer.eos_token_id in new_token_ids:
                    break

                # Update token history
                generated_token_count = len(all_tokens)
                current_memory_tokens = sum(get_memory_tokens(m) for m in current_mem_contents)
                rag_memory_tokens = sum(get_memory_tokens(m) for m in all_unique_memories)

                token_history.append({
                    'token_idx': generated_token_count,
                    'streaming': base_context_size + current_memory_tokens,
                    'rag': base_context_size + rag_memory_tokens,
                    'all': base_context_size + self.pool_total_tokens,
                })

                yield StreamEvent('context_update', {
                    'generated_tokens': generated_token_count,
                    'streaming': base_context_size + current_memory_tokens,
                    'rag': base_context_size + rag_memory_tokens,
                    'all': base_context_size + self.pool_total_tokens,
                    'gen_ms': gen_ms,
                })

                # Re-retrieve memories based on recent generation
                lookback_text = self.tokenizer.decode(
                    all_tokens[-lookback_tokens:], skip_special_tokens=True
                )

                t_retrieve = time.time()
                new_memories = self.pool.retrieve(lookback_text, max_memories=max_memories)
                retrieve_ms = int((time.time() - t_retrieve) * 1000)
                new_mem_contents = [m.content for m in new_memories]

                if set(new_mem_contents) != set(current_mem_contents):
                    added = [m for m in new_mem_contents if m not in current_mem_contents]
                    removed = [m for m in current_mem_contents if m not in new_mem_contents]

                    # Track unique memories
                    for mem in new_mem_contents:
                        all_unique_memories.add(mem)
                        get_memory_tokens(mem)

                    current_mem_contents = new_mem_contents
                    memory_context = self._format_memories(new_memories)
                    messages = [
                        {"role": "system", "content": self.config.system_prompt},
                        {"role": "system", "content": memory_context},
                    ]
                    for h in history[-6:]:
                        messages.append(h)
                    messages.append({"role": "user", "content": message})

                    text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    new_prefix_ids = self.tokenizer.encode(text, return_tensors="pt").to(
                        self.model.device
                    )

                    generated_tensor = torch.tensor([all_tokens], device=self.model.device)
                    current_ids = torch.cat([new_prefix_ids, generated_tensor], dim=1)

                    # Track context size after memory swap
                    current_memory_tokens = sum(get_memory_tokens(m) for m in current_mem_contents)
                    rag_memory_tokens = sum(get_memory_tokens(m) for m in all_unique_memories)

                    token_history.append({
                        'token_idx': len(all_tokens),
                        'streaming': base_context_size + current_memory_tokens,
                        'rag': base_context_size + rag_memory_tokens,
                        'all': base_context_size + self.pool_total_tokens,
                    })

                    yield StreamEvent('memory_update', {
                        'memories': new_mem_contents,
                        'added': added,
                        'removed': removed,
                        'base_context_size': base_context_size,
                        'current_memory_tokens': current_memory_tokens,
                        'rag_memory_tokens': rag_memory_tokens,
                        'all_memories_tokens': self.pool_total_tokens,
                        'unique_memories': len(all_unique_memories),
                        'token_history': token_history,
                        'retrieve_ms': retrieve_ms,
                    })
                else:
                    current_ids = outputs

        # Check if we hit max tokens without EOS
        hit_max = (
            len(all_tokens) >= max_tokens
            and self.tokenizer.eos_token_id not in all_tokens[-10:]
        )

        yield StreamEvent('timeline', {'data': timeline})
        if hit_max:
            yield StreamEvent('max_tokens', {'limit': max_tokens})
        yield StreamEvent('done', {})


def load_memories_to_pool(
    pool: MemoryPool,
    memory_file: str,
    embedder: Any = None,
) -> int:
    """
    Load memories from a JSON file into a MemoryPool.

    Args:
        pool: The MemoryPool to populate
        memory_file: Path to JSON file with memories
        embedder: Optional embedder instance with embed_batch method (uses pool's embed_fn if None)

    Returns:
        Total token count of all memories
    """
    with open(memory_file) as f:
        memories = json.load(f)

    # Batch embed all memories if embedder supports it
    if embedder and hasattr(embedder, 'embed_batch'):
        print(f"Batch embedding {len(memories)} memories...")
        memory_contents = [m["content"] for m in memories]
        embeddings = embedder.embed_batch(memory_contents)

        # Update the embedder's cache if it has one
        if hasattr(embedder, 'embed_cache'):
            for content, embedding in zip(memory_contents, embeddings):
                embedder.embed_cache[content] = embedding

        print(f"✓ Embedded {len(memories)} memories (cache size: {embedder.get_cache_size() if hasattr(embedder, 'get_cache_size') else 'N/A'})")

    # Add memories to pool
    for mem in memories:
        created_str = mem.get("created_at", "")
        try:
            dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            created_at = dt.replace(tzinfo=None)
        except Exception:
            created_at = datetime.now()

        pool.add(
            content=mem["content"],
            emotional_intensity=mem.get("emotional_intensity", 0.5),
            created_at=created_at,
        )

    return len(memories)

