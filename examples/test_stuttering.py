"""
Test script to observe the stuttering effect when injecting memories mid-generation.
Run with: uv run python examples/test_stuttering.py
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from streaming_memory import Memory, MemoryPool
import openai
from dotenv import load_dotenv

load_dotenv()

# Colors for terminal output
class C:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'

# Simple embedding cache
embed_cache = {}
client = openai.OpenAI()

def embed(text: str) -> list[float]:
    if text not in embed_cache:
        resp = client.embeddings.create(model="text-embedding-3-small", input=text)
        embed_cache[text] = resp.data[0].embedding
    return embed_cache[text]

def load_memories():
    """Load Aryan memories"""
    with open("examples/aryan_memories.json") as f:
        data = json.load(f)
    
    memories = []
    for i, m in enumerate(data):
        created = datetime.fromisoformat(m["created_at"].replace("Z", "+00:00")).replace(tzinfo=None)
        memories.append(Memory(
            id=str(i),
            content=m["content"],
            embedding=embed(m["content"]),
            emotional_intensity=m.get("emotional_intensity", 0.5),
            created_at=created,
        ))
    return memories

def test_with_injection(pool: MemoryPool, message: str, inject_every_n: int = 5):
    """Generate with memory injection every N tokens"""
    
    print(f"\n{C.HEADER}{'='*70}{C.END}")
    print(f"{C.BOLD}TEST: Inject memories every {inject_every_n} tokens{C.END}")
    print(f"{C.HEADER}{'='*70}{C.END}")
    print(f"{C.DIM}Message: {message}{C.END}\n")
    
    # Initial retrieval
    memories = pool.retrieve(message, max_memories=5)
    current_mems = [m.content for m in memories]
    
    print(f"{C.CYAN}Initial memories:{C.END}")
    for i, m in enumerate(current_mems):
        print(f"  {C.DIM}{i+1}. {m[:60]}...{C.END}")
    print()
    
    # Build initial messages
    system = """You are an AI tutor who has been working with Aryan, a Grade 5 student.
Use your memories naturally. Think step by step."""
    
    memory_context = "[My memories:]\n" + "\n".join(f"- {m}" for m in current_mems)
    
    messages = [
        {"role": "system", "content": system},
        {"role": "system", "content": memory_context},
        {"role": "user", "content": message},
    ]
    
    # Stream generation
    print(f"{C.GREEN}Generating:{C.END} ", end="", flush=True)
    
    full_response = ""
    token_count = 0
    memory_changes = 0
    
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
        max_tokens=200,
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            full_response += token
            token_count += 1
            
            # Print token
            print(token, end="", flush=True)
            
            # Check for memory update every N tokens
            if token_count % inject_every_n == 0:
                # Re-retrieve based on response so far
                new_query = message + " " + full_response[-200:]
                new_memories = pool.retrieve(new_query, max_memories=5)
                new_mems = [m.content for m in new_memories]
                
                old_set = set(current_mems)
                new_set = set(new_mems)
                
                if old_set != new_set:
                    memory_changes += 1
                    added = [m for m in new_mems if m not in old_set]
                    removed = [m for m in current_mems if m not in new_set]
                    
                    # Show memory change inline
                    print(f"\n{C.YELLOW}  [TOKEN {token_count}] MEMORY SHIFT #{memory_changes}{C.END}")
                    if added:
                        print(f"{C.GREEN}    + {added[0][:50]}...{C.END}")
                    if removed:
                        print(f"{C.RED}    - {removed[0][:50]}...{C.END}")
                    print(f"{C.GREEN}Continuing:{C.END} ", end="", flush=True)
                    
                    current_mems = new_mems
    
    print(f"\n\n{C.CYAN}Stats: {token_count} tokens, {memory_changes} memory changes{C.END}")
    return full_response, memory_changes


def test_without_injection(pool: MemoryPool, message: str):
    """Generate without mid-stream injection (baseline)"""
    
    print(f"\n{C.HEADER}{'='*70}{C.END}")
    print(f"{C.BOLD}BASELINE: No mid-stream injection{C.END}")
    print(f"{C.HEADER}{'='*70}{C.END}")
    print(f"{C.DIM}Message: {message}{C.END}\n")
    
    # Initial retrieval only
    memories = pool.retrieve(message, max_memories=5)
    current_mems = [m.content for m in memories]
    
    print(f"{C.CYAN}Memories (fixed):{C.END}")
    for i, m in enumerate(current_mems):
        print(f"  {C.DIM}{i+1}. {m[:60]}...{C.END}")
    print()
    
    system = """You are an AI tutor who has been working with Aryan, a Grade 5 student.
Use your memories naturally. Think step by step."""
    
    memory_context = "[My memories:]\n" + "\n".join(f"- {m}" for m in current_mems)
    
    messages = [
        {"role": "system", "content": system},
        {"role": "system", "content": memory_context},
        {"role": "user", "content": message},
    ]
    
    print(f"{C.GREEN}Generating:{C.END} ", end="", flush=True)
    
    full_response = ""
    token_count = 0
    
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
        max_tokens=200,
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            full_response += token
            token_count += 1
            print(token, end="", flush=True)
    
    print(f"\n\n{C.CYAN}Stats: {token_count} tokens, 0 memory changes{C.END}")
    return full_response


def test_with_actual_reinjection(pool: MemoryPool, message: str, inject_every_n: int = 10):
    """
    Actually rebuild the prompt and restart generation when memories change.
    This simulates what the Modal app does - and shows the stuttering.
    """
    
    print(f"\n{C.HEADER}{'='*70}{C.END}")
    print(f"{C.BOLD}TEST: ACTUAL RE-INJECTION (rebuilds prompt){C.END}")
    print(f"{C.HEADER}{'='*70}{C.END}")
    print(f"{C.DIM}Message: {message}{C.END}\n")
    
    # Initial retrieval
    memories = pool.retrieve(message, max_memories=5)
    current_mems = [m.content for m in memories]
    
    print(f"{C.CYAN}Initial memories:{C.END}")
    for i, m in enumerate(current_mems):
        print(f"  {C.DIM}{i+1}. {m[:60]}...{C.END}")
    print()
    
    system = """You are an AI tutor who has been working with Aryan, a Grade 5 student.
Use your memories naturally."""
    
    full_response = ""
    token_count = 0
    memory_changes = 0
    max_tokens = 150
    
    print(f"{C.GREEN}Generating:{C.END} ", end="", flush=True)
    
    while token_count < max_tokens:
        # Build messages with current memories
        memory_context = "[My memories:]\n" + "\n".join(f"- {m}" for m in current_mems)
        
        messages = [
            {"role": "system", "content": system},
            {"role": "system", "content": memory_context},
            {"role": "user", "content": message},
        ]
        
        # If we have partial response, include it
        if full_response:
            messages.append({"role": "assistant", "content": full_response})
        
        # Generate a chunk
        chunk_tokens = 0
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
            max_tokens=inject_every_n,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                token_count += 1
                chunk_tokens += 1
                print(token, end="", flush=True)
                
                if chunk.choices[0].finish_reason == "stop":
                    break
        
        if chunk_tokens == 0:
            break
            
        # Check for memory update
        new_query = message + " " + full_response[-200:]
        new_memories = pool.retrieve(new_query, max_memories=5)
        new_mems = [m.content for m in new_memories]
        
        old_set = set(current_mems)
        new_set = set(new_mems)
        
        if old_set != new_set:
            memory_changes += 1
            added = [m for m in new_mems if m not in old_set]
            removed = [m for m in current_mems if m not in new_set]
            
            print(f"\n{C.YELLOW}  ⚡ REBUILD #{memory_changes} at token {token_count}{C.END}")
            if added:
                print(f"{C.GREEN}    + {added[0][:50]}...{C.END}")
            if removed:
                print(f"{C.RED}    - {removed[0][:50]}...{C.END}")
            print(f"{C.GREEN}Continuing from: \"{full_response[-30:]}...\"{C.END}")
            print(f"{C.GREEN}>{C.END} ", end="", flush=True)
            
            current_mems = new_mems
    
    print(f"\n\n{C.CYAN}Stats: {token_count} tokens, {memory_changes} rebuilds{C.END}")
    return full_response, memory_changes


def main():
    print(f"{C.BOLD}{C.HEADER}")
    print("=" * 70)
    print("  STUTTERING TEST - Memory Injection Mid-Generation")
    print("=" * 70)
    print(f"{C.END}")
    
    print(f"{C.DIM}Loading memories...{C.END}")
    memories = load_memories()
    
    pool = MemoryPool(
        embed_fn=embed,
        embedding_weight=1.0,
        recency_weight=0.3,
        repetition_weight=0.2,
        emotional_weight=0.2,
        connection_weight=0.2,
    )
    for m in memories:
        pool.add(m)
    
    print(f"{C.DIM}Loaded {len(memories)} memories{C.END}")
    
    # Test message that might trigger memory changes
    test_message = "How should I help Aryan when he gets frustrated with math problems?"
    
    # Run tests
    test_without_injection(pool, test_message)
    test_with_injection(pool, test_message, inject_every_n=10)
    test_with_actual_reinjection(pool, test_message, inject_every_n=15)
    
    print(f"\n{C.HEADER}{'='*70}{C.END}")
    print(f"{C.BOLD}ANALYSIS{C.END}")
    print(f"{C.HEADER}{'='*70}{C.END}")
    print("""
The 'ACTUAL RE-INJECTION' test shows what happens when we rebuild the prompt
mid-generation. The model receives:
  1. New memories in system prompt
  2. The response so far as assistant message
  3. Instruction to continue

This can cause:
  - Repetition (model restates what it just said)
  - Topic drift (new memories pull attention elsewhere)  
  - Awkward transitions
  - Loss of coherence

Watch the output above to see these effects.
""")


if __name__ == "__main__":
    main()

