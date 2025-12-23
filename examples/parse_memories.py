"""
Parse hourly summaries into first-person, inference-driven memories.

The goal is to extract actionable insights a tutor would actually recall,
not raw facts. Focus on patterns, what works, and relationship dynamics.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EXTRACTION_PROMPT = """You are helping extract memories from a tutoring session summary.

These memories will be used by an AI tutor to recall relevant context during future sessions. 
The memories should be:

1. **First person** - written from the tutor's perspective ("I noticed...", "I've learned...")
2. **Inference-driven** - focus on WHY things happen, WHAT WORKS, and PATTERNS observed
3. **Actionable** - something useful to recall in future interactions
4. **Concise** - one clear insight per memory (1-2 sentences max)

DO NOT extract:
- Specific problem numbers or exact arithmetic (e.g., "6×7=43")
- Scheduling logistics (times, dates, task management)
- Technical setup details (AV checks, canvas updates)
- Generic statements that apply to any student

DO extract:
- Learning patterns ("I've noticed he self-corrects when given think time")
- What works/doesn't work ("Step-by-step demonstration works better than questions")
- Emotional/behavioral observations ("He gets frustrated when he feels tested")
- Skill understanding ("He knows the concept but execution is inconsistent")
- Parent dynamics ("His parent values specific feedback after sessions")
- Relationship insights ("He trusts me more when I acknowledge effort first")

For each memory, also provide:
- type: one of [what_works, pattern, current_state, relationship, insight]
- subject: one of [math, science, social_studies, reading, behavior, parent, general]
- emotional_intensity: 0.0-1.0 (higher for significant insights, emotional moments, breakthroughs)

Return a JSON object with a "memories" array. If the summary is purely administrative with no learning insights, return {"memories": []}.

Example output:
{"memories": [
  {
    "content": "I've noticed Aryan can self-correct arithmetic errors when I give him a quiet moment rather than immediately prompting",
    "type": "pattern",
    "subject": "behavior",
    "emotional_intensity": 0.5
  },
  {
    "content": "I've learned that breaking decimal multiplication into 'whole number first, then count places' helps him stay organized",
    "type": "what_works",
    "subject": "math",
    "emotional_intensity": 0.4
  }
]}"""


def parse_summary(client: OpenAI, summary: dict, idx: int) -> tuple[int, list[dict]]:
    """Extract memories from a single hourly summary."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": f"Extract memories from this tutoring session summary:\n\n{summary['summary']}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        
        result = json.loads(response.choices[0].message.content)
        memories = result.get("memories", [])
        
        # Add source metadata
        for mem in memories:
            mem["source_date"] = summary["date"]
            mem["source_hour"] = summary["hour"]
            mem["created_at"] = summary["created_at"]
        
        return idx, memories
    except Exception as e:
        print(f"Error processing {idx}: {e}")
        return idx, []


def deduplicate_memories(memories: list[dict]) -> list[dict]:
    """Remove near-duplicate memories, keeping the most recent."""
    seen_content = {}
    
    for mem in reversed(memories):
        key = mem["content"].lower().strip()[:100]  # Use first 100 chars as key
        if key not in seen_content:
            seen_content[key] = mem
    
    return list(seen_content.values())


def main():
    # Load summaries
    input_path = Path(__file__).parent / "aryan.json"
    with open(input_path) as f:
        summaries = json.load(f)
    
    print(f"Loaded {len(summaries)} hourly summaries")
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    all_memories = []
    completed = 0
    
    # Process in parallel with 20 workers
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(parse_summary, client, summary, i): i 
            for i, summary in enumerate(summaries)
        }
        
        for future in as_completed(futures):
            idx, memories = future.result()
            all_memories.extend(memories)
            completed += 1
            
            if completed % 10 == 0 or completed == len(summaries):
                print(f"Progress: {completed}/{len(summaries)} ({len(all_memories)} memories so far)")
    
    print(f"\nTotal raw memories: {len(all_memories)}")
    
    # Deduplicate
    unique_memories = deduplicate_memories(all_memories)
    print(f"After deduplication: {len(unique_memories)}")
    
    # Sort by date
    unique_memories.sort(key=lambda m: (m["source_date"], m["source_hour"]))
    
    # Save
    output_path = Path(__file__).parent / "aryan_memories.json"
    with open(output_path, "w") as f:
        json.dump(unique_memories, f, indent=2)
    
    print(f"\nSaved to {output_path}")
    
    # Print summary stats
    by_type = {}
    by_subject = {}
    for mem in unique_memories:
        by_type[mem.get("type", "unknown")] = by_type.get(mem.get("type", "unknown"), 0) + 1
        by_subject[mem.get("subject", "unknown")] = by_subject.get(mem.get("subject", "unknown"), 0) + 1
    
    print("\nBy type:")
    for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")
    
    print("\nBy subject:")
    for s, count in sorted(by_subject.items(), key=lambda x: -x[1]):
        print(f"  {s}: {count}")
    
    # Print some examples
    print("\n" + "="*60)
    print("SAMPLE MEMORIES:")
    print("="*60)
    
    import random
    samples = random.sample(unique_memories, min(15, len(unique_memories)))
    for mem in samples:
        print(f"\n[{mem.get('type', '?')}|{mem.get('subject', '?')}|{mem.get('emotional_intensity', '?')}]")
        print(f"  {mem['content']}")


if __name__ == "__main__":
    main()
