"""
Test the dad memories locally to see how retrieval works.
"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
import os

# Load memories
memories_path = Path(__file__).parent / "dad_memories.json"
with open(memories_path) as f:
    memories_raw = json.load(f)

print(f"Loaded {len(memories_raw)} memories")

# Setup OpenAI
client = OpenAI()

def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text[:8000],
    )
    return response.data[0].embedding

def cosine_sim(a, b):
    import numpy as np
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# Embed all memories
print("Embedding memories...")
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

# Test different queries that should retrieve different memories
print("\n" + "="*60)
print("QUERY: What should I get my dad for his birthday?")
print("="*60)
for score, content in retrieve("What should I get my dad for his birthday?"):
    print(f"  [{score:.3f}] {content[:60]}...")

print("\n" + "="*60)
print("QUERY: dad birthday gift golf")
print("="*60)
for score, content in retrieve("dad birthday gift golf"):
    print(f"  [{score:.3f}] {content[:60]}...")

print("\n" + "="*60)
print("QUERY: dad's golf clubs equipment driver")
print("="*60)
for score, content in retrieve("dad's golf clubs equipment driver"):
    print(f"  [{score:.3f}] {content[:60]}...")

print("\n" + "="*60)
print("QUERY: dad complained driver warped slicing shots")
print("="*60)
for score, content in retrieve("dad complained driver warped slicing shots"):
    print(f"  [{score:.3f}] {content[:60]}...")

print("\n" + "="*60)
print("QUERY: Callaway golf driver brand")
print("="*60)
for score, content in retrieve("Callaway golf driver brand"):
    print(f"  [{score:.3f}] {content[:60]}...")

print("\n" + "="*60)
print("QUERY: dad wants new driver Callaway Paradym")
print("="*60)
for score, content in retrieve("dad wants new driver Callaway Paradym"):
    print(f"  [{score:.3f}] {content[:60]}...")

# Simulate the chain
print("\n" + "="*60)
print("SIMULATING GENERATION CHAIN")
print("="*60)

queries = [
    "What should I get my dad for his birthday?",
    "What should I get my dad for his birthday? Let me think about what he enjoys...",
    "What should I get my dad for his birthday? Let me think about what he enjoys... He loves golf and plays every week",
    "What should I get my dad for his birthday? He loves golf... I remember his equipment has been having issues",
    "What should I get my dad for his birthday? He loves golf... his driver is old and warped, he keeps slicing",
    "What should I get my dad for his birthday? His driver is warped... what brand did he want to try?",
]

for i, q in enumerate(queries):
    print(f"\n--- Step {i+1} ---")
    print(f"Query (last 80 chars): ...{q[-80:]}")
    print("Top 3 memories:")
    for score, content in retrieve(q, k=3):
        print(f"  [{score:.3f}] {content[:70]}...")

