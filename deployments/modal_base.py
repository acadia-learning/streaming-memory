"""
Base Modal infrastructure for streaming memory deployments.

Provides reusable image builders and service class factories.
"""

from pathlib import Path
from typing import Callable

import modal

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def get_package_path() -> Path:
    """Get the path to the streaming_memory package."""
    return Path(__file__).parent.parent / "streaming_memory"


def create_model_image(
    model_id: str,
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL,
    memory_files: list[tuple[Path, str]] | None = None,
) -> modal.Image:
    """
    Create a Modal image with the LLM, embedding model, and dependencies.
    
    Args:
        model_id: HuggingFace model ID for LLM
        embedding_model_id: HuggingFace model ID for embeddings (default: BGE-small)
        memory_files: List of (local_path, container_path) tuples for memory files
    
    Returns:
        Modal Image configured for streaming memory
    """
    def download_model():
        from huggingface_hub import snapshot_download
        snapshot_download(model_id, ignore_patterns=["*.gguf"])
        print(f"Downloaded {model_id}")
    
    def download_embedding_model():
        from huggingface_hub import snapshot_download
        snapshot_download(embedding_model_id)
        print(f"Downloaded {embedding_model_id}")
    
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch",
            "sentence-transformers>=2.2.0",
            "transformers>=4.40",
            "accelerate>=0.28",
            "numpy",
            "huggingface_hub",
            "openai",
            "fastapi",
            "uvicorn",
            "pydantic",
            "sentencepiece>=0.1.99",
        )
        .run_function(download_model)
        .run_function(download_embedding_model)
        .add_local_dir(get_package_path(), "/root/streaming_memory")
    )
    
    # Add memory files
    if memory_files:
        for local_path, container_path in memory_files:
            image = image.add_local_file(local_path, container_path)
    
    return image


class ModalServiceBase:
    """
    Base class for Modal streaming memory services.
    
    Subclass this and implement setup() to configure your specific assistant.
    Uses local BGE embeddings for fast, free embedding operations.
    """
    
    model_id: str = "Qwen/Qwen3-8B"
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL
    
    def load_model(self):
        """Load the LLM and tokenizer."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"🚀 Loading {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"✅ LLM loaded! Device: {next(self.model.parameters()).device}")
    
    def load_embedder(self, device: str = "cuda"):
        """Load local embedding model (BGE-small by default)."""
        import sys
        sys.path.insert(0, "/root")
        from streaming_memory.embeddings import create_embedder
        
        print(f"🔧 Loading embedding model {self.embedding_model_id}...")
        self.embedder = create_embedder(
            model_name=self.embedding_model_id,
            device=device,
            cache_embeddings=True,
        )
        print(f"✅ Embedding model loaded!")
        return self.embedder
    
    def load_memories(self, memory_file: str, pool) -> int:
        """
        Load memories into the pool with batch embedding using local model.
        
        Returns the total token count of all memories.
        """
        import json
        from datetime import datetime
        
        print(f"📚 Loading memories from {memory_file}...")
        
        with open(memory_file) as f:
            memories = json.load(f)
        
        # Batch embed all memories using local model
        memory_contents = [m["content"] for m in memories]
        print(f"  Batch embedding {len(memory_contents)} memories...")
        self.embedder.embed_batch(memory_contents)
        print(f"  ✅ Embedded (cache size: {self.embedder.get_cache_size()})")
        
        # Add memories to pool
        for mem in memories:
            created_str = mem.get("created_at", "")
            try:
                dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                created_at = dt.replace(tzinfo=None)
            except:
                created_at = datetime.now()
            
            pool.add(
                content=mem["content"],
                emotional_intensity=mem.get("emotional_intensity", 0.5),
                created_at=created_at,
            )
        
        # Calculate total tokens
        all_memory_text = "\n".join([f"- {m['content']}" for m in memories])
        total_tokens = len(self.tokenizer.encode(all_memory_text))
        
        print(f"  ✅ Loaded {len(memories)} memories ({total_tokens} tokens)")
        
        return total_tokens



