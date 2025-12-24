"""
Modal deployment for LongMemEval batch evaluation.

This script:
1. Loads preprocessed memory cache
2. For each instance: builds MemoryPool, runs streaming evaluation
3. Outputs JSONL for LongMemEval's evaluate_qa.py

Usage:
    # First, preprocess the data (one-time):
    uv run python examples/preprocess_longmemeval.py --dataset oracle --output data/memory_cache.json

    # Then run evaluation on Modal:
    uv run modal run examples/longmemeval_eval.py --memory-cache data/memory_cache.json --output results.jsonl

    # Evaluate with LongMemEval:
    cd LongMemEval/src/evaluation
    python3 evaluate_qa.py gpt-4o ../../results.jsonl ../../data/longmemeval_oracle.json
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import modal
import numpy as np

# Modal app
app = modal.App("longmemeval-streaming-memory")

MODEL_ID = "Qwen/Qwen3-8B"


def download_model():
    """Download model during image build."""
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID, ignore_patterns=["*.gguf"])
    print(f"Downloaded {MODEL_ID}")


def download_embedding_model():
    """Download embedding model during image build."""
    from huggingface_hub import snapshot_download
    snapshot_download("Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    print("Downloaded Alibaba-NLP/gte-Qwen2-1.5B-instruct (qwen3-embedding-8b)")


# Build image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40",
        "accelerate>=0.28",
        "numpy",
        "huggingface_hub",
        "openai",
        "pydantic",
        "tqdm",
        "sentencepiece>=0.1.99",
    )
    .run_function(download_model)
    .run_function(download_embedding_model)
    .add_local_dir(
        Path(__file__).parent.parent / "streaming_memory",
        "/app/streaming_memory"
    )
)


@app.cls(
    image=image,
    gpu="L4",
    timeout=3600,  # 1 hour timeout for batch processing
    secrets=[modal.Secret.from_name("openai-secret")],
    concurrency_limit=8,  # Limit to 8 concurrent GPUs (leave headroom under 10 limit)
    container_idle_timeout=300,  # Keep containers warm for 5 min between batches
)
class LongMemEvalRunner:
    """Modal class for running LongMemEval evaluation."""
    
    @modal.enter()
    def setup(self):
        """Load model and tokenizer once on container startup."""
        import torch
        import sys
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        sys.path.insert(0, "/app")
        from streaming_memory.embeddings import create_embedder
        
        print(f"🔧 Loading {MODEL_ID}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        # Load local embedding model (qwen3-embedding-8b)
        print("🔧 Loading embedding model...")
        self.embedder = create_embedder(
            model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            device="cuda",  # Modal GPUs support CUDA
            cache_embeddings=True,
        )
        
        print("✅ Models loaded!")
    
    def embed(self, text: str) -> np.ndarray:
        """Get embedding using local model."""
        return self.embedder.embed(text)
    
    @modal.method()
    def evaluate_instance(
        self,
        instance: dict,
        memory_cache: dict[str, list[dict]],
        update_every_n: int = 10,
        max_memories: int = 10,
        lookback_tokens: int = 60,
    ) -> dict:
        """
        Evaluate a single LongMemEval instance.
        
        Returns dict with question_id, hypothesis, and metrics.
        """
        import sys
        sys.path.insert(0, "/app")
        
        from streaming_memory.memory import MemoryPool
        from streaming_memory.service import StreamingMemoryService
        from streaming_memory.config import AssistantConfig, MemoryConfig, ModelConfig
        from streaming_memory.longmemeval import (
            load_longmemeval_instance,
            run_streaming_eval,
            compute_retrieval_metrics,
        )
        
        question_id = instance["question_id"]
        print(f"Processing {question_id}...")
        
        # Load instance into a fresh MemoryPool
        loaded = load_longmemeval_instance(
            instance=instance,
            memory_cache=memory_cache,
            embed_fn=self.embed,
            pool_kwargs={
                "softmax_temperature": 0.15,
                "diversity_weight": 0.5,
                "association_weight": 0.5,
            },
        )
        
        # Calculate total tokens in pool
        pool_total_tokens = sum(
            len(self.tokenizer.encode(m.content))
            for m in loaded.pool.memories.values()
        )
        
        # Create config for this evaluation
        config = AssistantConfig(
            name="longmemeval",
            system_prompt="""You are a helpful assistant with access to memories from past conversations with this user.

Answer the user's question based on the information in your memories.
Be direct and concise. If the information isn't in your memories, say so.

Think step by step in <think>...</think> tags before responding.""",
            memory=MemoryConfig(
                memory_file="",  # Not used - pool is pre-populated
                memory_prefix="[Memories from past conversations:]",
            ),
            model=ModelConfig(
                model_id=MODEL_ID,
                temperature=0.7,
                max_tokens=2048,  # Increased for complex reasoning with thinking
            ),
        )
        
        # Create service with the loaded pool
        service = StreamingMemoryService(
            config=config,
            pool=loaded.pool,
            tokenizer=self.tokenizer,
            model=self.model,
            pool_total_tokens=pool_total_tokens,
        )
        
        # Run streaming evaluation
        result = run_streaming_eval(
            service=service,
            question=loaded.question,
            update_every_n=update_every_n,
            max_memories=max_memories,
            lookback_tokens=lookback_tokens,
        )
        
        # Extract retrieved memory IDs from timeline
        retrieved_ids = set()
        if result.get("timeline"):
            for item in result["timeline"]:
                # Timeline has memory contents, need to map back to IDs
                for mem_content in item.get("memories", []):
                    # Find memory ID by content
                    for mid, mem in loaded.pool.memories.items():
                        if mem.content == mem_content:
                            retrieved_ids.add(mid)
        
        # Compute retrieval metrics (skip for abstention questions)
        retrieval_metrics = None
        if not question_id.endswith("_abs"):
            retrieval_metrics = compute_retrieval_metrics(
                retrieved_memory_ids=retrieved_ids,
                session_to_memory_ids=loaded.session_to_memory_ids,
                answer_session_ids=loaded.answer_session_ids,
            )
        
        return {
            "question_id": question_id,
            "question_type": loaded.question_type,
            "hypothesis": result["hypothesis"],
            "expected_answer": loaded.expected_answer,
            "memory_swaps": result["memory_swaps"],
            "thinking_tokens": result.get("thinking_tokens", 0),
            "response_tokens": result.get("response_tokens", 0),
            "hit_max_tokens": result.get("hit_max_tokens", False),
            "retrieval_metrics": retrieval_metrics,
            "timing": result.get("timing", {}),
        }


@app.local_entrypoint()
def main(
    memory_cache: str,
    output: str,
    dataset: str = "oracle",
    update_every_n: int = 10,
    max_memories: int = 10,
    lookback_tokens: int = 60,
    limit: int | None = None,
):
    """
    Run LongMemEval evaluation.
    
    Args:
        memory_cache: Path to preprocessed memory cache JSON
        output: Path to save results JSONL
        dataset: Which dataset to evaluate on (oracle, s, m)
        update_every_n: Re-retrieve memories every N tokens
        max_memories: Maximum memories in context
        lookback_tokens: Tokens to look back for re-retrieval
        limit: Optional limit on instances to process
    """
    import urllib.request
    
    print(f"Loading memory cache from {memory_cache}")
    with open(memory_cache) as f:
        cache = json.load(f)
    print(f"Loaded {len(cache)} session memories")
    
    # Download dataset
    dataset_urls = {
        "oracle": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
        "s": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
        "m": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json",
    }
    
    print(f"Downloading {dataset} dataset...")
    with urllib.request.urlopen(dataset_urls[dataset]) as response:
        data = json.loads(response.read())
    print(f"Loaded {len(data)} instances")
    
    if limit:
        data = data[:limit]
        print(f"Limited to {limit} instances")
    
    # Run evaluation
    runner = LongMemEvalRunner()
    
    results = []
    for result in runner.evaluate_instance.map(
        data,
        kwargs={
            "memory_cache": cache,
            "update_every_n": update_every_n,
            "max_memories": max_memories,
            "lookback_tokens": lookback_tokens,
        },
    ):
        results.append(result)
        
        # Print progress
        print(f"\n{'='*60}")
        print(f"Question ID: {result['question_id']}")
        print(f"Type: {result['question_type']}")
        print(f"Expected: {result['expected_answer']}")
        hyp = result['hypothesis']
        print(f"Generated: {hyp[:200]}..." if hyp else "Generated: (EMPTY)")
        print(f"Memory swaps: {result['memory_swaps']}")
        print(f"Tokens - thinking: {result['thinking_tokens']}, response: {result['response_tokens']}, hit_max: {result['hit_max_tokens']}")
        if result.get('timing'):
            t = result['timing']
            print(f"⏱️ Timing - total: {t.get('total_ms', 0)}ms, TTFT: {t.get('time_to_first_token_ms', 0)}ms, gen: {t.get('generation_ms', 0)}ms, {t.get('tokens_per_sec', 0):.1f} tok/s")
        if result['retrieval_metrics']:
            m = result['retrieval_metrics']
            print(f"Retrieval - Session recall: {m['session_recall']:.2f}, Memory recall: {m['memory_recall']:.2f}")
    
    # Save results in JSONL format for LongMemEval evaluation
    print(f"\nSaving results to {output}")
    with open(output, "w") as f:
        for result in results:
            # LongMemEval expects: {"question_id": ..., "hypothesis": ...}
            f.write(json.dumps({
                "question_id": result["question_id"],
                "hypothesis": result["hypothesis"],
            }) + "\n")
    
    # Also save full results with metrics
    full_output = output.replace(".jsonl", "_full.json")
    with open(full_output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full results to {full_output}")
    
    # Print aggregate metrics
    print("\n" + "="*60)
    print("AGGREGATE METRICS")
    print("="*60)
    
    # Memory swap stats
    swaps = [r["memory_swaps"] for r in results]
    print(f"\nMemory Swaps:")
    print(f"  Mean: {np.mean(swaps):.2f}")
    print(f"  Min: {np.min(swaps)}, Max: {np.max(swaps)}")
    
    # Retrieval metrics (excluding abstention)
    retrieval_results = [r for r in results if r["retrieval_metrics"]]
    if retrieval_results:
        session_recalls = [r["retrieval_metrics"]["session_recall"] for r in retrieval_results]
        memory_recalls = [r["retrieval_metrics"]["memory_recall"] for r in retrieval_results]
        
        print(f"\nRetrieval (n={len(retrieval_results)}):")
        print(f"  Session Recall: {np.mean(session_recalls):.4f}")
        print(f"  Memory Recall: {np.mean(memory_recalls):.4f}")
    
    print("\nTo evaluate QA correctness, run:")
    print(f"  cd LongMemEval/src/evaluation")
    print(f"  python3 evaluate_qa.py gpt-4o {output} ../../data/longmemeval_{dataset}.json")


