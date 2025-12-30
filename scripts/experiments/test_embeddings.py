"""
Quick test script for local embeddings.

Tests the local qwen3-embedding-8b model to verify it's working correctly.
"""

import time

from streaming_memory.embeddings import create_embedder


def main():
    print("Creating local embedder (qwen3-embedding-8b)...")
    embedder = create_embedder()

    # Test single embedding
    print("\n=== Testing single embedding ===")
    text = "The quick brown fox jumps over the lazy dog"

    start = time.time()
    embedding = embedder.embed(text)
    elapsed = time.time() - start

    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Time: {elapsed*1000:.2f}ms")

    # Test cached embedding (should be much faster)
    start = time.time()
    embedder.embed(text)
    elapsed = time.time() - start
    print(f"Cached time: {elapsed*1000:.2f}ms")

    # Test batch embedding
    print("\n=== Testing batch embedding ===")
    texts = [
        "I love programming in Python",
        "Machine learning is fascinating",
        "Natural language processing is powerful",
        "Deep learning models are complex",
        "AI is transforming the world",
    ]

    start = time.time()
    embeddings = embedder.embed_batch(texts)
    elapsed = time.time() - start

    print(f"Embedded {len(texts)} texts")
    print(f"Time: {elapsed*1000:.2f}ms ({elapsed*1000/len(texts):.2f}ms per text)")
    print(f"Cache size: {embedder.get_cache_size()}")

    # Test similarity
    print("\n=== Testing similarity ===")
    import numpy as np

    query = "Python programming language"
    query_emb = embedder.embed(query)

    print(f"Query: {query}")
    print("\nSimilarities:")
    for text, emb in zip(texts, embeddings):
        similarity = np.dot(query_emb, emb)
        print(f"  {similarity:.4f} - {text}")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()

