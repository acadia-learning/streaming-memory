"""
Aryan Tutor Assistant deployment on Modal.

Run:
    modal deploy deployments/aryan_tutor.py
"""

import sys
from datetime import datetime
from pathlib import Path

import modal

# Configuration
MODEL_ID = "Qwen/Qwen3-8B"
EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"
APP_NAME = "aryan-tutor"

# Paths
data_path = Path(__file__).parent.parent / "data"
memories_path = data_path / "aryan_memories.json"
package_path = Path(__file__).parent.parent / "streaming_memory"


def download_model():
    """Download model during image build."""
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID, ignore_patterns=["*.gguf"])
    print(f"Downloaded {MODEL_ID}")


def download_embedding_model():
    """Download embedding model during image build."""
    from huggingface_hub import snapshot_download
    snapshot_download(EMBEDDING_MODEL_ID)
    print(f"Downloaded {EMBEDDING_MODEL_ID}")


# Create Modal app and image
app = modal.App(APP_NAME)

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
    .add_local_file(memories_path, "/app/aryan_memories.json")
    .add_local_dir(package_path, "/root/streaming_memory")
)


# System prompt for Aryan's tutor
ARYAN_TUTOR_PROMPT = """You are a friendly AI tutor assistant who has worked extensively with a student named Aryan.

You have access to memories from past tutoring sessions covering:
- Math (fractions, arithmetic, word problems)
- Reading comprehension
- Science (chemistry, biology)
- Session scheduling and communications

When memories are provided, use them to:
- Reference past lessons and progress
- Build on concepts Aryan has learned
- Acknowledge patterns you've noticed (areas of strength, things that need more practice)
- Maintain continuity between sessions

Think step by step in <think>...</think> tags before responding.

Be encouraging, patient, and build on what you know about Aryan's learning style.

Important: Do not use emojis in your responses."""


@app.cls(
    image=image,
    gpu="A100",
    timeout=600,
    scaledown_window=300,
)
class AryanTutor:
    """Aryan Tutor with streaming memory using local embeddings."""

    @modal.enter()
    def startup(self):
        """Initialize model, memories, and service."""
        import json

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        sys.path.insert(0, "/root")
        from streaming_memory import MemoryPool
        from streaming_memory.config import AssistantConfig, MemoryConfig
        from streaming_memory.embeddings import create_embedder
        from streaming_memory.service import StreamingMemoryService

        # Create config for Aryan tutor
        self.config = AssistantConfig(
            name="aryan-tutor",
            description="AI tutor assistant for Aryan with session memory",
            system_prompt=ARYAN_TUTOR_PROMPT,
            memory=MemoryConfig(
                memory_file="/app/aryan_memories.json",
                memory_prefix="[Past tutoring session memories:]",
            ),
        )

        # Load LLM
        print(f"Loading {MODEL_ID}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"LLM loaded! Device: {next(self.model.parameters()).device}")

        # Setup local embedding model (BGE-small)
        print(f"Loading embedding model {EMBEDDING_MODEL_ID}...")
        self.embedder = create_embedder(
            model_name=EMBEDDING_MODEL_ID,
            device="cuda",
            cache_embeddings=True,
        )
        print("Embedding model loaded!")

        # Create memory pool with local embedder
        print("Loading memories...")
        self.pool = MemoryPool(
            embed_fn=self.embedder,
            softmax_temperature=self.config.memory.softmax_temperature,
            diversity_weight=self.config.memory.diversity_weight,
            association_weight=self.config.memory.association_weight,
        )

        # Load memories
        with open("/app/aryan_memories.json") as f:
            memories = json.load(f)

        # Batch embed all memories using local model
        memory_contents = [m["content"] for m in memories]
        print(f"  Batch embedding {len(memory_contents)} memories...")
        self.embedder.embed_batch(memory_contents)
        print(f"  Embedded (cache size: {self.embedder.get_cache_size()})")

        # Add memories to pool
        for mem in memories:
            created_str = mem.get("created_at", "")
            try:
                dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                created_at = dt.replace(tzinfo=None)
            except Exception:
                created_at = datetime.now()

            self.pool.add(
                content=mem["content"],
                emotional_intensity=mem.get("emotional_intensity", 0.5),
                created_at=created_at,
            )

        # Calculate total tokens
        all_memory_text = "\n".join([f"- {m['content']}" for m in memories])
        pool_total_tokens = len(self.tokenizer.encode(all_memory_text))
        print(f"  Loaded {len(memories)} memories ({pool_total_tokens} tokens)")

        # Create service
        self.service = StreamingMemoryService(
            config=self.config,
            pool=self.pool,
            tokenizer=self.tokenizer,
            model=self.model,
            pool_total_tokens=pool_total_tokens,
        )

        print("Container ready - using local BGE embeddings!")

    @modal.asgi_app()
    def serve(self):
        from streaming_memory.api import create_app
        return create_app(
            service=self.service,
            config=self.config,
            model_id=MODEL_ID,
        )


