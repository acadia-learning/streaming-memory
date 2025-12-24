"""
Base Modal infrastructure for streaming memory deployments.

Provides reusable image builders and service class factories.
"""

from pathlib import Path
from typing import Callable

import modal


def get_package_path() -> Path:
    """Get the path to the streaming_memory package."""
    return Path(__file__).parent.parent / "streaming_memory"


def create_model_image(
    model_id: str,
    memory_files: list[tuple[Path, str]] | None = None,
) -> modal.Image:
    """
    Create a Modal image with the LLM and dependencies.
    
    Args:
        model_id: HuggingFace model ID
        memory_files: List of (local_path, container_path) tuples for memory files
    
    Returns:
        Modal Image configured for streaming memory
    """
    def download_model():
        from huggingface_hub import snapshot_download
        snapshot_download(model_id, ignore_patterns=["*.gguf"])
        print(f"Downloaded {model_id}")
    
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch",
            "transformers>=4.40",
            "accelerate>=0.28",
            "numpy",
            "huggingface_hub",
            "openai",
            "fastapi",
            "uvicorn",
            "pydantic",
        )
        .run_function(download_model)
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
    """
    
    model_id: str = "Qwen/Qwen3-8B"
    
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
        print(f"✅ Model loaded! Device: {next(self.model.parameters()).device}")
    
    def create_embed_fn(self, openai_client):
        """Create an embedding function with caching."""
        self.embed_cache = {}
        
        def embed(text: str):
            if text in self.embed_cache:
                return self.embed_cache[text]
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text[:8000],
            )
            result = response.data[0].embedding
            self.embed_cache[text] = result
            return result
        
        return embed
    
    def load_memories(self, memory_file: str, embed_fn, openai_client, pool) -> int:
        """
        Load memories into the pool with batch embedding.
        
        Returns the total token count of all memories.
        """
        import json
        from datetime import datetime
        
        print(f"📚 Loading memories from {memory_file}...")
        
        with open(memory_file) as f:
            memories = json.load(f)
        
        # Batch embed all memories
        memory_contents = [m["content"] for m in memories]
        batch_size = 50
        for i in range(0, len(memory_contents), batch_size):
            batch = memory_contents[i:i + batch_size]
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=batch,
            )
            for j, emb_data in enumerate(response.data):
                self.embed_cache[batch[j]] = emb_data.embedding
        
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



