"""
Test local embeddings on Modal.

Quick test to verify the local qwen3-embedding-8b model works on Modal GPUs.
"""

import modal
from pathlib import Path

# Modal app
app = modal.App("test-local-embeddings")


def download_embedding_model():
    """Download embedding model during image build."""
    from huggingface_hub import snapshot_download
    snapshot_download("Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    print("✓ Downloaded Alibaba-NLP/gte-Qwen2-1.5B-instruct (qwen3-embedding-8b)")


# Build image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40",
        "numpy",
        "huggingface_hub",
        "sentencepiece>=0.1.99",
        "openai",  # Need this for streaming_memory imports
        "pydantic",  # Need this for streaming_memory imports
    )
    .run_function(download_embedding_model)
    .add_local_dir(
        Path(__file__).parent / "streaming_memory",
        "/app/streaming_memory"
    )
)


@app.function(
    image=image,
    gpu="T4",  # Use cheaper T4 for testing
    timeout=300,
)
def test_embeddings():
    """Test local embeddings on Modal."""
    import time
    import sys
    import numpy as np
    
    sys.path.insert(0, "/app")
    from streaming_memory.embeddings import create_embedder
    
    print("\n" + "="*70)
    print("TESTING LOCAL EMBEDDINGS ON MODAL")
    print("="*70)
    
    # Create embedder
    print("\n📦 Loading embedding model...")
    start = time.time()
    embedder = create_embedder(device="cuda")
    load_time = time.time() - start
    print(f"✓ Model loaded in {load_time:.2f}s on {embedder.device}")
    
    # Test single embedding
    print("\n🔍 Test 1: Single embedding")
    text = "The quick brown fox jumps over the lazy dog"
    
    start = time.time()
    embedding = embedder.embed(text)
    elapsed = (time.time() - start) * 1000
    
    print(f"  Text: {text}")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Time (cold): {elapsed:.2f}ms")
    
    # Test cached
    start = time.time()
    embedding2 = embedder.embed(text)
    elapsed_cached = (time.time() - start) * 1000
    print(f"  Time (cached): {elapsed_cached:.2f}ms ({elapsed/elapsed_cached:.0f}x faster)")
    
    # Verify same result
    assert np.allclose(embedding, embedding2), "Cached embedding differs!"
    print(f"  ✓ Cached result matches")
    
    # Test batch embedding
    print("\n🔍 Test 2: Batch embedding")
    texts = [
        "I love programming in Python",
        "Machine learning is fascinating", 
        "Natural language processing is powerful",
        "Deep learning models are complex",
        "AI is transforming the world",
    ]
    
    start = time.time()
    embeddings = embedder.embed_batch(texts)
    elapsed = (time.time() - start) * 1000
    
    print(f"  Embedded {len(texts)} texts")
    print(f"  Time: {elapsed:.2f}ms ({elapsed/len(texts):.2f}ms per text)")
    print(f"  Cache size: {embedder.get_cache_size()}")
    
    # Test similarity
    print("\n🔍 Test 3: Similarity computation")
    query = "Python programming language"
    query_emb = embedder.embed(query)
    
    print(f"  Query: {query}")
    print(f"  Similarities:")
    
    similarities = []
    for text, emb in zip(texts, embeddings):
        similarity = float(np.dot(query_emb, emb))
        similarities.append((similarity, text))
        print(f"    {similarity:.4f} - {text}")
    
    # Verify best match makes sense
    similarities.sort(reverse=True)
    best_match = similarities[0][1]
    print(f"\n  Best match: {best_match}")
    assert "Python" in best_match or "programming" in best_match, "Unexpected best match!"
    print(f"  ✓ Similarity makes semantic sense")
    
    # Test with MemoryPool
    print("\n🔍 Test 4: Integration with MemoryPool")
    from streaming_memory.memory import MemoryPool
    
    pool = MemoryPool(embed_fn=embedder)
    
    # Add some memories
    memories = [
        "I love hiking in the mountains",
        "My favorite food is pizza",
        "I work as a software engineer",
        "I enjoy reading science fiction books",
        "I play guitar in my free time",
    ]
    
    for mem in memories:
        pool.add(mem)
    
    # Retrieve
    query = "What do you like to eat?"
    results = pool.retrieve(query, max_memories=3)
    
    print(f"  Query: {query}")
    print(f"  Retrieved {len(results)} memories:")
    for mem in results:
        print(f"    - {mem.content}")
    
    # Verify food memory is retrieved
    contents = [m.content for m in results]
    assert any("pizza" in c or "food" in c for c in contents), "Food memory not retrieved!"
    print(f"  ✓ Relevant memory retrieved")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED ON MODAL!")
    print("="*70)
    
    return {
        "status": "success",
        "load_time_seconds": load_time,
        "device": embedder.device,
        "cache_size": embedder.get_cache_size(),
        "tests_passed": 4,
    }


@app.local_entrypoint()
def main():
    """Run the test."""
    print("🚀 Starting Modal test for local embeddings...\n")
    
    result = test_embeddings.remote()
    
    print(f"\n📊 Test Summary:")
    print(f"  Status: {result['status']}")
    print(f"  Load time: {result['load_time_seconds']:.2f}s")
    print(f"  Device: {result['device']}")
    print(f"  Cache size: {result['cache_size']}")
    print(f"  Tests passed: {result['tests_passed']}/4")
    print(f"\n✨ Local embeddings work perfectly on Modal!")

