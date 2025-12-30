"""
Compare OpenAI embeddings vs local qwen3-embedding-8b.

This script demonstrates the speed and quality differences between
the two embedding approaches.
"""

import time

import numpy as np

from streaming_memory.embeddings import create_embedder


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    print("=" * 70)
    print("LOCAL EMBEDDINGS DEMO (qwen3-embedding-8b)")
    print("=" * 70)

    # Create local embedder
    print("\n📦 Loading local embedding model...")
    start = time.time()
    embedder = create_embedder()
    load_time = time.time() - start
    print(f"✓ Model loaded in {load_time:.2f}s")

    # Test texts
    texts = [
        "I love hiking in the mountains during summer",
        "Mountain climbing is my favorite outdoor activity",
        "I prefer reading books at home",
        "Programming Python is very enjoyable",
        "Machine learning models are fascinating",
    ]

    query = "What outdoor activities do you enjoy?"

    print(f"\n🔍 Query: '{query}'")
    print("\n📝 Corpus:")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")

    # Embed query (cold start)
    print("\n⏱️  Performance Test:")
    print("-" * 70)

    start = time.time()
    query_emb = embedder.embed(query)
    query_time = (time.time() - start) * 1000
    print(f"Query embedding (cold):   {query_time:.2f}ms")

    # Embed query again (cached)
    start = time.time()
    embedder.embed(query)
    cached_time = (time.time() - start) * 1000
    print(f"Query embedding (cached): {cached_time:.2f}ms ({query_time/cached_time:.0f}x faster)")

    # Batch embed corpus
    start = time.time()
    text_embs = embedder.embed_batch(texts)
    batch_time = (time.time() - start) * 1000
    print(f"Batch embed {len(texts)} texts:  {batch_time:.2f}ms ({batch_time/len(texts):.2f}ms per text)")

    # Similarity search
    print("\n🎯 Similarity Results:")
    print("-" * 70)

    similarities = []
    for text, emb in zip(texts, text_embs):
        sim = cosine_similarity(query_emb, emb)
        similarities.append((sim, text))

    # Sort by similarity
    similarities.sort(reverse=True)

    for i, (sim, text) in enumerate(similarities, 1):
        bar = "█" * int(sim * 50)
        print(f"{i}. [{sim:.4f}] {bar}")
        print(f"   {text}")
        print()

    # Cache stats
    print("💾 Cache Statistics:")
    print("-" * 70)
    print(f"Cached embeddings: {embedder.get_cache_size()}")
    print(f"Memory saved by caching: {embedder.get_cache_size()} lookups")

    # Quality check
    print("\n✨ Quality Check:")
    print("-" * 70)
    top_result = similarities[0][1]
    if "hiking" in top_result or "climbing" in top_result or "mountain" in top_result:
        print("✓ Top result is semantically relevant!")
        print(f"  '{top_result}'")
    else:
        print("⚠️ Unexpected top result:")
        print(f"  '{top_result}'")

    # Cost comparison
    print("\n💰 Cost Comparison (for this demo):")
    print("-" * 70)
    total_texts = len(texts) + 1  # corpus + query
    openai_cost = total_texts * 0.00002  # Rough estimate for text-embedding-3-small
    print(f"OpenAI API:     ${openai_cost:.6f}")
    print("Local model:    $0.000000 (free!)")
    print(f"Savings:        ${openai_cost:.6f} ({(openai_cost/0.00002):.0f} texts)")

    # Latency comparison
    print("\n⚡ Latency Comparison:")
    print("-" * 70)
    openai_latency = 75  # Typical network latency in ms
    local_latency = query_time
    print(f"OpenAI API:     ~{openai_latency}ms (network latency)")
    print(f"Local model:    ~{local_latency:.2f}ms")
    print(f"Speedup:        {openai_latency/local_latency:.1f}x faster (cold)")
    print(f"                {openai_latency/cached_time:.0f}x faster (cached)")

    print("\n" + "=" * 70)
    print("✅ Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

