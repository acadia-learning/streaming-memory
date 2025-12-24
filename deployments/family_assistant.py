"""
Family Assistant deployment on Modal.

Run:
    modal deploy deployments/family_assistant.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import modal

# Configuration
MODEL_ID = "Qwen/Qwen3-8B"
APP_NAME = "streaming-memory"

# Paths
data_path = Path(__file__).parent.parent / "examples"
dad_memories_path = data_path / "dad_memories.json"
package_path = Path(__file__).parent.parent / "streaming_memory"


def download_model():
    """Download model during image build."""
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID, ignore_patterns=["*.gguf"])
    print(f"Downloaded {MODEL_ID}")


# Create Modal app and image
app = modal.App(APP_NAME)

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
    .add_local_file(dad_memories_path, "/app/dad_memories.json")
    .add_local_dir(package_path, "/root/streaming_memory")
)


@app.cls(
    image=image,
    gpu="A100",
    timeout=600,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("openai-secret")],
)
class FamilyAssistant:
    """Family Assistant with streaming memory."""
    
    @modal.enter()
    def startup(self):
        """Initialize model, memories, and service."""
        import json
        
        import torch
        from openai import OpenAI
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        sys.path.insert(0, "/root")
        from streaming_memory import MemoryPool
        from streaming_memory.config import FAMILY_ASSISTANT
        from streaming_memory.service import StreamingMemoryService
        
        # Load model
        print(f"🚀 Loading {MODEL_ID}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"✅ Model loaded! Device: {next(self.model.parameters()).device}")
        
        # Setup embedding with cache
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.embed_cache = {}
        
        def embed(text: str):
            if text in self.embed_cache:
                return self.embed_cache[text]
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text[:8000],
            )
            result = response.data[0].embedding
            self.embed_cache[text] = result
            return result
        
        # Create memory pool
        print("📚 Loading memories...")
        self.pool = MemoryPool(
            embed_fn=embed,
            softmax_temperature=FAMILY_ASSISTANT.memory.softmax_temperature,
            diversity_weight=FAMILY_ASSISTANT.memory.diversity_weight,
            association_weight=FAMILY_ASSISTANT.memory.association_weight,
        )
        
        # Load and embed memories
        with open("/app/dad_memories.json") as f:
            memories = json.load(f)
        
        # Batch embed all memories
        memory_contents = [m["content"] for m in memories]
        batch_size = 50
        for i in range(0, len(memory_contents), batch_size):
            batch = memory_contents[i:i + batch_size]
            response = self.openai_client.embeddings.create(
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
            
            self.pool.add(
                content=mem["content"],
                emotional_intensity=mem.get("emotional_intensity", 0.5),
                created_at=created_at,
            )
        
        # Calculate total tokens
        all_memory_text = "\n".join([f"- {m['content']}" for m in memories])
        pool_total_tokens = len(self.tokenizer.encode(all_memory_text))
        print(f"  ✅ Loaded {len(memories)} memories ({pool_total_tokens} tokens)")
        
        # Create service
        self.service = StreamingMemoryService(
            config=FAMILY_ASSISTANT,
            pool=self.pool,
            tokenizer=self.tokenizer,
            model=self.model,
            pool_total_tokens=pool_total_tokens,
        )
        
        self.config = FAMILY_ASSISTANT
        print("🟢 Container ready - responses will be fast!")
    
    @modal.asgi_app()
    def serve(self):
        from streaming_memory.api import create_app
        return create_app(
            service=self.service,
            config=self.config,
            model_id=MODEL_ID,
        )
