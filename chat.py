#!/usr/bin/env python3
"""
Interactive chat with dual memory architecture:
- Working Memory: Traditional messages array for API calls
- Long-term Memory: Background process that creates semantic memory blocks

After each assistant response, the long-term memory process runs in the background
to analyze the cache and create memory blocks with:
- Summary (embedded for semantic search)
- Emotional intensity (surprise, arousal, control)
- Retrieval frequency tracking
- Creation timestamp

Usage:
    python chat.py                     # Start fresh
    python chat.py --load memories.json  # Load existing memories
    python chat.py -l memories.json      # Short form
"""

import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic
from memory import LongTermMemory, MemoryBlock, QueryResult

# Load environment variables
load_dotenv()


# =============================================================================
# ANSI Color Codes
# =============================================================================
class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Thinking - cyan/dim
    THINKING = "\033[36m"  # Cyan
    
    # Response - default white
    RESPONSE = "\033[97m"  # Bright white
    
    # Memory created - green
    MEMORY_CREATED = "\033[92m"  # Bright green
    
    # Memory hit - yellow
    MEMORY_HIT = "\033[93m"  # Bright yellow
    
    # Labels
    LABEL = "\033[90m"  # Gray


def print_memory_created(memory: MemoryBlock) -> None:
    """
    Callback when a new memory block is created.
    Prints the memory so we know it's working.
    """
    print(f"\n{Colors.MEMORY_CREATED}🧠 NEW MEMORY CREATED:{Colors.RESET}")
    print(f"{Colors.MEMORY_CREATED}   Summary: {memory.summary}{Colors.RESET}")
    print(f"{Colors.LABEL}   Emotions: surprise={memory.emotions.surprise:.2f}, arousal={memory.emotions.arousal:.2f}, control={memory.emotions.control:.2f}{Colors.RESET}")
    print(f"{Colors.LABEL}   Created: {memory.created_at.strftime('%H:%M:%S')}{Colors.RESET}")
    print()


def print_memory_hit(results: list[QueryResult]) -> None:
    """
    Print memory hits when relevant memories are surfaced during thinking.
    """
    for result in results:
        print(f"\n{Colors.MEMORY_HIT}💡 MEMORY HIT (score: {result.composite_score:.2f}):{Colors.RESET}")
        print(f"{Colors.MEMORY_HIT}   Summary: {result.memory.summary}{Colors.RESET}")
        print(f"{Colors.LABEL}   Scores: sim={result.similarity_score:.2f} emo={result.emotion_score:.2f} freq={result.frequency_score:.2f} rec={result.recency_score:.2f}{Colors.RESET}")
        print(f"{Colors.LABEL}   Retrievals: {result.memory.frequency}{Colors.RESET}")
        print()


def stream_response(
    client: Anthropic,
    messages: list[dict],
    long_term_memory: LongTermMemory,
    user_input: str
) -> str:
    """
    Stream a response from Claude.
    
    - Uses messages array (working memory) for API call
    - All experiences flow through long-term memory cache
    - Thinking chunks trigger memory queries
    - After response completes, triggers background memory processing
    
    Args:
        client: Anthropic client
        messages: Working memory - the messages array for API calls
        long_term_memory: Long-term memory for background processing
        user_input: The user's input
        
    Returns:
        The assistant's complete response
    """
    # Ingest user input into long-term memory cache
    long_term_memory.ingest(user_input, entry_type="user_input")
    
    # Query long-term memory BEFORE responding
    # This surfaces relevant memories to include in context
    memory_results = []
    if long_term_memory.memories:
        print(f"\n{Colors.LABEL}🔍 Querying memories...{Colors.RESET}", end="", flush=True)
        memory_results = long_term_memory.query(user_input, top_k=5)
        if memory_results:
            print(f" {len(memory_results)} relevant memories found.")
            for result in memory_results:
                print(f"{Colors.MEMORY_HIT}   💡 {result.memory.summary} (score: {result.composite_score:.2f}){Colors.RESET}")
        else:
            print(" No relevant memories.")
    
    # Build system prompt with surfaced memories
    current_time = datetime.now()
    system_prompt = f"""You are a helpful assistant. Be concise but thorough.

Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"""
    
    if memory_results:
        memory_lines = []
        for r in memory_results:
            time_ago = current_time - r.memory.created_at
            if time_ago.total_seconds() < 60:
                age_str = f"{int(time_ago.total_seconds())}s ago"
            elif time_ago.total_seconds() < 3600:
                age_str = f"{int(time_ago.total_seconds() / 60)}m ago"
            else:
                age_str = f"{time_ago.total_seconds() / 3600:.1f}h ago"
            
            memory_lines.append(f"- [{age_str}] {r.memory.summary}")
        
        memory_context = "\n".join(memory_lines)
        system_prompt += f"""

You have the following relevant memories from past experiences:
{memory_context}

Use these memories naturally in your response if they're relevant."""
    
    # Add user message to working memory
    messages.append({"role": "user", "content": user_input})
    
    print(f"\n{Colors.BOLD}Assistant:{Colors.RESET} ", end="", flush=True)
    
    # Accumulate the full response for working memory
    full_response = ""
    
    # Stream the response with extended thinking enabled
    current_block_type = None
    thinking_buffer = ""
    thinking_started = False
    response_started = False
    
    with client.messages.stream(
        model="claude-opus-4-5-20251101",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000
        },
        system=system_prompt,
        messages=messages
    ) as stream:
        for event in stream:
            # Handle content block start
            if event.type == "content_block_start":
                current_block_type = event.content_block.type
                thinking_started = False
                response_started = False
            
            # Handle content block delta (streaming content)
            elif event.type == "content_block_delta":
                if event.delta.type == "thinking_delta":
                    if not thinking_started:
                        print(f"\n{Colors.THINKING}🤔 [thinking] ", end="", flush=True)
                        thinking_started = True
                    
                    chunk = event.delta.thinking
                    print(chunk, end="", flush=True)
                    thinking_buffer += chunk
                    
                    # Ingest thinking on sentence boundaries (no query during streaming)
                    if chunk.endswith(('.', '?', '!', '\n')) and len(thinking_buffer) > 50:
                        long_term_memory.ingest(thinking_buffer, entry_type="thinking")
                        thinking_buffer = ""
                        
                elif event.delta.type == "text_delta":
                    if not response_started:
                        print(f"{Colors.RESET}\n{Colors.RESPONSE}", end="", flush=True)
                        response_started = True
                    
                    chunk = event.delta.text
                    print(chunk, end="", flush=True)
                    full_response += chunk
                    
                    # Ingest response chunks into cache (no query for responses)
                    long_term_memory.ingest(chunk, entry_type="response_text")
            
            # Handle content block stop
            elif event.type == "content_block_stop":
                if current_block_type == "thinking":
                    # Reset color when thinking ends
                    print(Colors.RESET, end="")
                    # Ingest any remaining thinking (no query during streaming)
                    if thinking_buffer:
                        long_term_memory.ingest(thinking_buffer, entry_type="thinking")
                        thinking_buffer = ""
                    print()  # Newline after thinking block
    
    print("\n")
    
    # Add assistant response to working memory
    messages.append({"role": "assistant", "content": full_response})
    
    # Trigger background memory processing
    # This runs in a separate thread and won't block the conversation
    long_term_memory.process_cache_async()
    
    return full_response


def show_stats(messages: list[dict], long_term_memory: LongTermMemory) -> None:
    """Display memory statistics."""
    print("\n📊 Memory Statistics:")
    print("-" * 50)
    print("Working Memory (messages array):")
    print(f"  Total messages: {len(messages)}")
    print(f"  User messages: {sum(1 for m in messages if m['role'] == 'user')}")
    print(f"  Assistant messages: {sum(1 for m in messages if m['role'] == 'assistant')}")
    
    ltm_stats = long_term_memory.get_stats()
    print("\nLong-term Memory:")
    print(f"  Cache entries: {ltm_stats['cache_entries']}")
    print(f"  Cache by type: {ltm_stats['cache_by_type']}")
    print(f"  Memory blocks: {ltm_stats['memory_blocks']}")
    print(f"  Total retrievals: {ltm_stats['total_retrievals']}")
    print(f"  Avg emotions: {ltm_stats['avg_emotions']}")
    print("-" * 50 + "\n")


def show_working_memory(messages: list[dict]) -> None:
    """Display working memory (messages array)."""
    print("\n📝 Working Memory (messages array):")
    print("-" * 50)
    for i, msg in enumerate(messages):
        role = msg['role'].capitalize()
        content = msg['content'][:100] + ('...' if len(msg['content']) > 100 else '')
        print(f"[{i}] {role}: {content}")
    print("-" * 50 + "\n")


def show_memories(long_term_memory: LongTermMemory) -> None:
    """Display all memory blocks."""
    print("\n🧠 Long-term Memory Blocks:")
    print("-" * 60)
    memories = long_term_memory.get_all_memories()
    if not memories:
        print("  No memories created yet.")
    else:
        for i, mem in enumerate(memories):
            # Calculate emotion intensity average
            emotion_avg = (mem.emotions.surprise + mem.emotions.arousal + mem.emotions.control) / 3
            print(f"[{i}] {mem.summary}")
            print(f"    Emotions: surprise={mem.emotions.surprise:.2f} arousal={mem.emotions.arousal:.2f} control={mem.emotions.control:.2f} (avg={emotion_avg:.2f})")
            print(f"    Retrievals: {mem.frequency} | Created: {mem.created_at.strftime('%H:%M:%S')}")
            print()
    print("-" * 60 + "\n")


def main():
    """Run the interactive chat loop with dual memory architecture."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Interactive chat with dual memory architecture"
    )
    parser.add_argument(
        "-l", "--load",
        type=str,
        help="Path to a memory file to load on startup"
    )
    parser.add_argument(
        "-s", "--save",
        type=str,
        help="Path to save memories on exit (optional)"
    )
    args = parser.parse_args()
    
    client = Anthropic()
    
    # Working memory - traditional messages array for API calls
    messages: list[dict] = []
    
    # Long-term memory - background process creates semantic memory blocks
    if args.load and Path(args.load).exists():
        long_term_memory = LongTermMemory.from_file(args.load, default_top_k=10)
        print(f"📂 Loaded {len(long_term_memory.memories)} memories from {args.load}")
    else:
        long_term_memory = LongTermMemory(default_top_k=10)
        if args.load:
            print(f"⚠️  Memory file not found: {args.load}, starting fresh")
    
    # Set callback to print when memories are created
    long_term_memory.on_memory_created = print_memory_created
    
    print("=" * 60)
    print("🤖 Dual Memory Chat Agent")
    print("=" * 60)
    print("Working Memory: messages array for API calls")
    print("Long-term Memory: semantic memory blocks with embeddings")
    print()
    if long_term_memory.memories:
        print(f"Loaded with {len(long_term_memory.memories)} existing memories.")
    print("After each response, memories are created in the background.")
    print()
    print("Commands:")
    print("  'stats' - Show memory statistics")
    print("  'working' - Show working memory (messages array)")
    print("  'memories' - Show all memory blocks")
    print("  'clear' - Clear both memories")
    print("  'clear working' - Clear only working memory")
    print("  'clear ltm' - Clear only long-term memory")
    print("  'save <path>' - Save memories to file")
    print("  'exit' - Quit")
    print("=" * 60)
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() == "exit":
                if args.save:
                    long_term_memory.save(args.save)
                    print(f"💾 Saved {len(long_term_memory.memories)} memories to {args.save}")
                print("Goodbye!")
                break
            
            if user_input.lower() == "stats":
                show_stats(messages, long_term_memory)
                continue
            
            if user_input.lower() == "working":
                show_working_memory(messages)
                continue
            
            if user_input.lower() == "memories":
                show_memories(long_term_memory)
                continue
            
            if user_input.lower() == "clear":
                messages.clear()
                long_term_memory.clear()
                print("Both memories cleared.\n")
                continue
            
            if user_input.lower() == "clear working":
                messages.clear()
                print("Working memory cleared.\n")
                continue
            
            if user_input.lower() == "clear ltm":
                long_term_memory.clear()
                print("Long-term memory cleared.\n")
                continue
            
            if user_input.lower().startswith("save "):
                save_path = user_input[5:].strip()
                if save_path:
                    long_term_memory.save(save_path)
                    print(f"💾 Saved {len(long_term_memory.memories)} memories to {save_path}\n")
                else:
                    print("Usage: save <path>\n")
                continue
            
            # Normal conversation flow
            stream_response(client, messages, long_term_memory, user_input)
        
        except KeyboardInterrupt:
            print("\n")
            if args.save:
                long_term_memory.save(args.save)
                print(f"💾 Saved {len(long_term_memory.memories)} memories to {args.save}")
            print("Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
