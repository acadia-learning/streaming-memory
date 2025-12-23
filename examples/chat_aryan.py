"""
Chat as Aryan's tutor with Hebbian memory.

This loads the extracted memories and lets you interact as if you're
the AI tutor who has been working with Aryan for months.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

from streaming_memory import MemoryPool

load_dotenv()


def create_embed_fn(client: OpenAI):
    """Create embedding function with caching."""
    cache = {}
    
    def embed(text: str):
        if text in cache:
            return cache[text]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000],
        )
        result = response.data[0].embedding
        cache[text] = result
        return result
    
    return embed


def load_aryan_memories(pool: MemoryPool, memories_path: Path) -> int:
    """Load Aryan's memories into the pool."""
    with open(memories_path) as f:
        memories = json.load(f)
    
    for mem in memories:
        # Parse the created_at date
        created_str = mem.get("created_at", "")
        try:
            created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
        except:
            created_at = datetime.now()
        
        pool.add(
            content=mem["content"],
            emotional_intensity=mem.get("emotional_intensity", 0.5),
            created_at=created_at,
        )
    
    return len(memories)


def format_memories(memories: list) -> str:
    """Format memories for context injection."""
    if not memories:
        return ""
    
    lines = ["[My memories from working with Aryan:]"]
    for mem in memories:
        lines.append(f"- {mem.content}")
    
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("ARYAN'S TUTOR - Hebbian Memory Chat")
    print("=" * 70)
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embed_fn = create_embed_fn(client)
    
    pool = MemoryPool(
        embed_fn=embed_fn,
        softmax_temperature=0.15,
        diversity_weight=0.5,
        association_weight=0.5,
    )
    
    # Load Aryan's memories
    memories_path = Path(__file__).parent / "aryan_memories.json"
    print(f"\nLoading memories from {memories_path}...")
    count = load_aryan_memories(pool, memories_path)
    print(f"Loaded {count} memories about Aryan")
    
    SYSTEM_PROMPT = """You are an AI tutor who has been working with Aryan, a Grade 5 student, for several months.

You have built up memories and insights about how he learns, what works for him, and your relationship.

When memories are provided, use them naturally to inform your responses. You know Aryan well - his struggles with decimal multiplication, his tendency to self-correct when given time, his occasional frustration, and what teaching strategies work best for him.

Speak as yourself - the tutor who has this history with Aryan. Reference your experiences and observations naturally.

If asked about Aryan, draw on your memories. If asked to help with a lesson, consider what you've learned about his learning style."""

    print(f"\nYou are now chatting as Aryan's tutor.")
    print("Ask about Aryan, plan a lesson, or discuss teaching strategies.")
    print("\nCommands: 'quit' to exit, 'stats' for memory stats, 'memories' for last retrieved")
    print("-" * 70 + "\n")
    
    conversation_history = []
    last_memories = []
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        
        if user_input.lower() == "stats":
            stats = pool.get_stats()
            print(f"\nMemory Pool Stats:")
            for k, v in stats.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.2f}")
                else:
                    print(f"  {k}: {v}")
            print()
            continue
        
        if user_input.lower() == "memories":
            if last_memories:
                print(f"\nLast retrieved memories ({len(last_memories)}):")
                for mem in last_memories:
                    print(f"  [{mem.retrieval_count}x] {mem.content}")
            else:
                print("\nNo memories retrieved yet.")
            print()
            continue
        
        # Build query including recent conversation
        query = user_input
        if conversation_history:
            recent = " ".join([m["content"] for m in conversation_history[-4:]])
            query = recent + " " + user_input
        
        # Retrieve memories
        memories = pool.retrieve(query, max_memories=5)
        last_memories = memories
        
        print(f"\n  [Retrieved {len(memories)} memories]")
        
        # Build messages
        memory_context = format_memories(memories)
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if memory_context:
            messages.append({"role": "system", "content": memory_context})
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_input})
        
        # Get response
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=800,
            )
            assistant_message = response.choices[0].message.content
        except Exception as e:
            print(f"\nError: {e}")
            continue
        
        print(f"\nTutor: {assistant_message}\n")
        
        # Update conversation history
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": assistant_message})
        
        if len(conversation_history) > 16:
            conversation_history = conversation_history[-16:]


if __name__ == "__main__":
    main()


