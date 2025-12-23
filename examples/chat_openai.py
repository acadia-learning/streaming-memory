"""
Interactive chat with OpenAI + Hebbian memory.

Simple example using closed-source models where we update memories
between turns (not mid-generation).

Usage:
    uv run python examples/chat_openai.py
"""

import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streaming_memory import MemoryPool, load_sample_memories

load_dotenv()


def create_embed_fn(client: OpenAI):
    """Create embedding function with caching."""
    cache = {}
    
    def embed(text: str):
        if text in cache:
            return cache[text]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        result = response.data[0].embedding
        cache[text] = result
        return result
    
    return embed


def format_memories(memories: list) -> str:
    """Format memories for context injection."""
    if not memories:
        return ""
    
    lines = ["[My relevant memories and experiences:]"]
    for mem in memories:
        days_old = (datetime.now() - mem.created_at).days
        age = f"{days_old}d ago" if days_old > 0 else "today"
        lines.append(f"- ({age}) {mem.content}")
    
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("HEBBIAN MEMORY CHAT")
    print("=" * 70)
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embed_fn = create_embed_fn(client)
    
    pool = MemoryPool(
        embed_fn=embed_fn,
        softmax_temperature=0.15,
        diversity_weight=0.8,
        association_weight=0.5,
    )
    
    print("\nLoading memories...")
    count = load_sample_memories(pool)
    print(f"Loaded {count} memories")
    
    MODEL = "gpt-4o-mini"
    
    SYSTEM_PROMPT = """You are a helpful assistant with access to the user's personal memories and experiences.

When memories are provided, incorporate them naturally into your thinking and response.
Reference relevant experiences, acknowledge context you know about the user, and be personable.
Don't explicitly say "according to your memories" - use them naturally like a friend would."""

    print(f"\nUsing model: {MODEL}")
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
                    print(f"  [{mem.retrieval_count}x] {mem.content[:70]}...")
            else:
                print("\nNo memories retrieved yet.")
            print()
            continue
        
        # Build query including recent conversation
        query_for_retrieval = user_input
        if conversation_history:
            recent = " ".join([m["content"] for m in conversation_history[-4:]])
            query_for_retrieval = recent + " " + user_input
        
        # Retrieve memories
        memories = pool.retrieve(query_for_retrieval, max_memories=3)
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
                model=MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=500,
            )
            assistant_message = response.choices[0].message.content
        except Exception as e:
            print(f"\nError: {e}")
            continue
        
        print(f"\nAssistant: {assistant_message}\n")
        
        # Update conversation history
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": assistant_message})
        
        if len(conversation_history) > 16:
            conversation_history = conversation_history[-16:]


if __name__ == "__main__":
    main()


