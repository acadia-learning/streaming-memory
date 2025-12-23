"""
Test memory retrieval on Modal to see why memories aren't changing.
"""
import json
from pathlib import Path
import modal

app = modal.App("test-retrieval")

dad_memories_path = Path(__file__).parent / "dad_memories.json"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("openai", "numpy")
    .add_local_file(dad_memories_path, "/app/dad_memories.json")
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("openai-secret")],
)
def test_retrieval():
    import numpy as np
    from openai import OpenAI
    
    client = OpenAI()
    
    # Load memories
    with open("/app/dad_memories.json") as f:
        memories_raw = json.load(f)
    
    print(f"Loaded {len(memories_raw)} memories")
    print("\nMemories:")
    for i, m in enumerate(memories_raw):
        print(f"  {i}: {m['content'][:70]}...")
    
    def embed(text: str) -> list[float]:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000],
        )
        return response.data[0].embedding
    
    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    
    # Embed all memories
    print("\nEmbedding memories...")
    memories = []
    for m in memories_raw:
        memories.append({
            "content": m["content"],
            "embedding": embed(m["content"])
        })
    print("Done embedding")
    
    def retrieve(query: str, k: int = 5):
        query_emb = embed(query)
        scored = [(cosine_sim(query_emb, m["embedding"]), m["content"]) for m in memories]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]
    
    # Test specific vs generic queries
    test_queries = [
        # Generic
        "What should I get my dad for his birthday?",
        "dad birthday gift",
        
        # Golf specific
        "dad golf",
        "dad's golf equipment",
        "dad's golf driver",
        
        # Problem specific
        "dad frustrated golf slicing",
        "dad's driver is old and warped",
        
        # Solution specific  
        "Callaway driver",
        "Callaway Paradym driver golf",
        "new golf driver recommendation",
        
        # Simulated generation chain
        "What should I get my dad? He loves golf",
        "He loves golf and has been having issues with his driver",
        "His driver is old and warped, he keeps slicing shots",
        "He mentioned wanting to try the Callaway Paradym",
    ]
    
    print("\n" + "="*80)
    print("TESTING QUERY SPECIFICITY")
    print("="*80)
    
    for query in test_queries:
        print(f"\n--- Query: {query[:60]}{'...' if len(query) > 60 else ''} ---")
        results = retrieve(query, k=5)
        for score, content in results:
            print(f"  [{score:.3f}] {content[:65]}...")
    
    # Now let's see the similarity between key memories
    print("\n" + "="*80)
    print("MEMORY-TO-MEMORY SIMILARITY")
    print("="*80)
    
    key_memories = [
        "Dad loves playing golf.",
        "Dad's current golf driver is old and warped.",
        "Dad saw an ad for the new Callaway Paradym driver and mentioned wanting to try it.",
    ]
    
    key_indices = []
    for km in key_memories:
        for i, m in enumerate(memories):
            if km in m["content"]:
                key_indices.append(i)
                break
    
    print("\nSimilarity between key memories:")
    for i, idx_i in enumerate(key_indices):
        for j, idx_j in enumerate(key_indices):
            if i < j:
                sim = cosine_sim(memories[idx_i]["embedding"], memories[idx_j]["embedding"])
                print(f"  [{sim:.3f}] '{key_memories[i][:40]}...' <-> '{key_memories[j][:40]}...'")
    
    # Test: does mentioning "driver" in query help retrieve driver memories?
    print("\n" + "="*80)
    print("DRIVER SPECIFICITY TEST")
    print("="*80)
    
    q1 = "dad golf birthday gift"
    q2 = "dad golf driver birthday gift"
    q3 = "dad golf driver old warped birthday gift"
    
    for q in [q1, q2, q3]:
        print(f"\nQuery: {q}")
        results = retrieve(q, k=3)
        for score, content in results:
            marker = "<<<" if "driver" in content.lower() or "callaway" in content.lower() else ""
            print(f"  [{score:.3f}] {content[:65]}... {marker}")


@app.local_entrypoint()
def main():
    test_retrieval.remote()

