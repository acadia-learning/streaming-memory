"""
Preprocess Aryan's 5-minute summaries into memories.

This script:
1. Reads aryan_5min_summaries.json
2. Extracts individual memories from each summary using GPT
3. Saves as JSON cache compatible with the streaming memory system

Usage:
    uv run python scripts/preprocess_aryan_summaries.py --output data/aryan_memories.json
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

load_dotenv()


class ExtractedMemory(BaseModel):
    """A single memory extracted from a summary."""
    content: str  # First-person memory text
    emotional_intensity: float  # 0-1, captures surprise/salience/valence


class SummaryMemories(BaseModel):
    """All memories extracted from a summary."""
    memories: list[ExtractedMemory]


SUMMARIZER_SYSTEM_PROMPT = """You are an AI assistant reviewing summaries of past interactions.
Extract distinct memories - facts, preferences, events, or insights.

For each memory:
1. Write in FIRST PERSON ("I learned...", "The user told me...", "I noticed...", "I helped...")
2. Rate emotional_intensity (0-1):
   - 0.3-0.4: Routine facts (preferences, schedules)
   - 0.5-0.6: Notable information (life events, problems)
   - 0.7-0.8: Significant/surprising (achievements, strong emotions)
   - 0.9-1.0: Critical (emergencies, major life changes)
3. Each memory should be a single, atomic fact or event

Extract ALL distinct pieces of information. A single summary may yield multiple memories.
If the summary is too vague to extract meaningful memories, return an empty list."""


async def extract_memories_from_summary(
    client: AsyncOpenAI,
    summary: dict,
    model: str,
    semaphore: asyncio.Semaphore,
) -> tuple[dict, list[dict]]:
    """Extract memories from a single summary."""
    async with semaphore:
        try:
            date = summary["date"]
            hour = summary["hour"]
            minute = summary["minute"]
            summary_text = summary["summary"]

            user_prompt = f"""Summary from {date} at {hour}:{minute:02d}:

{summary_text}

Extract all distinct memories from this summary."""

            response = await client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=SummaryMemories,
            )

            result = response.choices[0].message.parsed
            memories = result.memories if result else []

            return summary, [
                {
                    "content": m.content,
                    "emotional_intensity": m.emotional_intensity,
                    "created_at": summary.get("created_at", f"{date}T{hour:02d}:{minute:02d}:00+00:00"),
                }
                for m in memories
            ]
        except Exception as e:
            print(f"\nError processing summary from {summary.get('date')}: {e}")
            return summary, []


async def preprocess_summaries_async(
    summaries: list[dict],
    client: AsyncOpenAI,
    model: str = "gpt-4o-mini",
    max_concurrent: int = 20,
    output_path: Path | None = None,
) -> list[dict]:
    """
    Extract memories from all summaries using async parallel processing.

    Returns:
        List of memory dicts with content, emotional_intensity, created_at
    """
    print(f"Processing {len(summaries)} summaries")
    print(f"Using {max_concurrent} concurrent requests")

    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks
    tasks = [
        extract_memories_from_summary(client, summary, model, semaphore)
        for summary in summaries
    ]

    # Process with progress bar
    all_memories = []
    for coro in tqdm_asyncio.as_completed(tasks, desc="Extracting memories"):
        summary, memories = await coro
        all_memories.extend(memories)

    return all_memories


async def main_async():
    parser = argparse.ArgumentParser(
        description="Preprocess Aryan's 5-minute summaries into memories"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="aryan_5min_summaries.json",
        help="Input path for summaries JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for memory cache JSON",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for extraction",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Maximum concurrent API requests",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of summaries to process (for testing)",
    )

    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = AsyncOpenAI(api_key=api_key)
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Load summaries
    print(f"Loading summaries from {input_path}")
    with open(input_path) as f:
        summaries = json.load(f)
    print(f"Loaded {len(summaries)} summaries")

    # Limit if specified
    if args.limit:
        summaries = summaries[:args.limit]
        print(f"Limited to {len(summaries)} summaries")

    # Process summaries
    memories = await preprocess_summaries_async(
        summaries=summaries,
        client=client,
        model=args.model,
        max_concurrent=args.concurrency,
        output_path=output_path,
    )

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(memories, f, indent=2)

    print(f"\n✅ Saved {len(memories)} memories to {output_path}")
    print(f"Memories per summary: {len(memories) / len(summaries):.2f}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()


