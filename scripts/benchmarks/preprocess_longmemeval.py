"""
Preprocess LongMemEval data into summarized memories.

This is a one-time script that:
1. Downloads LongMemEval dataset
2. Extracts unique sessions across all instances
3. Summarizes each session into multiple first-person memories (in parallel)
4. Saves as JSON cache for use in evaluation

Usage:
    uv run python examples/preprocess_longmemeval.py --dataset oracle --output data/memory_cache.json
    uv run python examples/preprocess_longmemeval.py --dataset s --output data/memory_cache_s.json
"""

import argparse
import asyncio
import json
import os
import sys
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from streaming_memory.longmemeval import (  # noqa: E402
    SUMMARIZER_SYSTEM_PROMPT,
    SessionMemories,
    format_session_for_summarization,
)

DATASET_URLS = {
    "oracle": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
    "s": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
    "m": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json",
}


def download_dataset(dataset: str, cache_dir: Path) -> list[dict]:
    """Download LongMemEval dataset if not cached."""
    cache_file = cache_dir / f"longmemeval_{dataset}.json"

    if cache_file.exists():
        print(f"Loading cached dataset from {cache_file}")
        with open(cache_file) as f:
            return json.load(f)

    url = DATASET_URLS.get(dataset)
    if not url:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: {list(DATASET_URLS.keys())}")

    print(f"Downloading {dataset} dataset from {url}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read())

    with open(cache_file, "w") as f:
        json.dump(data, f)

    print(f"Cached dataset to {cache_file}")
    return data


def extract_unique_sessions(data: list[dict]) -> dict[str, dict]:
    """
    Extract unique sessions across all instances.

    Returns:
        Dict mapping session_id -> {"session": [...], "date": "..."}
    """
    sessions = {}

    for instance in data:
        session_ids = instance["haystack_session_ids"]
        session_dates = instance["haystack_dates"]
        haystack_sessions = instance["haystack_sessions"]

        for sess_id, date, session in zip(session_ids, session_dates, haystack_sessions):
            if sess_id not in sessions:
                sessions[sess_id] = {
                    "session": session,
                    "date": date,
                }

    return sessions


async def summarize_session_async(
    client: AsyncOpenAI,
    sess_id: str,
    sess_data: dict,
    model: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, list[dict]]:
    """Summarize a single session asynchronously."""
    async with semaphore:
        try:
            session_text = format_session_for_summarization(sess_data["session"])
            user_prompt = f"""Conversation from {sess_data["date"]}:

{session_text}

Extract all distinct memories from this conversation."""

            response = await client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=SessionMemories,
            )

            result = response.choices[0].message.parsed
            memories = result.memories if result else []

            return sess_id, [
                {
                    "content": m.content,
                    "emotional_intensity": m.emotional_intensity,
                    "event_date": m.event_date,
                }
                for m in memories
            ]
        except Exception as e:
            print(f"\nError processing {sess_id}: {e}")
            return sess_id, []


async def preprocess_sessions_async(
    sessions: dict[str, dict],
    client: AsyncOpenAI,
    model: str = "gpt-4o",
    resume_from: dict | None = None,
    max_concurrent: int = 20,
    output_path: Path | None = None,
) -> dict[str, list[dict]]:
    """
    Summarize all sessions into memories using async parallel processing.

    Args:
        sessions: Dict from extract_unique_sessions
        client: AsyncOpenAI client
        model: Model to use for summarization
        resume_from: Existing cache to resume from (skip already processed)
        max_concurrent: Maximum concurrent API calls
        output_path: Path to save incremental results

    Returns:
        Dict mapping session_id -> [{"content": ..., "emotional_intensity": ...}, ...]
    """
    cache = dict(resume_from) if resume_from else {}

    # Filter to sessions not yet processed
    to_process = {k: v for k, v in sessions.items() if k not in cache}

    if not to_process:
        print("All sessions already processed!")
        return cache

    print(f"Processing {len(to_process)} sessions ({len(cache)} already cached)")
    print(f"Using {max_concurrent} concurrent requests")

    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks
    tasks = [
        summarize_session_async(client, sess_id, sess_data, model, semaphore)
        for sess_id, sess_data in to_process.items()
    ]

    # Process with progress bar and save incrementally
    completed = 0
    for coro in tqdm_asyncio.as_completed(tasks, desc="Summarizing sessions"):
        sess_id, memories = await coro
        cache[sess_id] = memories
        completed += 1

        # Save every 10 completions
        if output_path and completed % 10 == 0:
            with open(output_path, "w") as f:
                json.dump(cache, f, indent=2)

    return cache


async def main_async():
    parser = argparse.ArgumentParser(
        description="Preprocess LongMemEval data into summarized memories"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["oracle", "s", "m"],
        default="oracle",
        help="Which LongMemEval dataset to use",
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
        default="gpt-5-mini",
        help="Model to use for summarization",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data",
        help="Directory to cache downloaded datasets",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file if it exists",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of sessions to process (for testing)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Maximum concurrent API requests",
    )

    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = AsyncOpenAI(api_key=api_key)
    cache_dir = Path(args.cache_dir)
    output_path = Path(args.output)

    # Download dataset
    data = download_dataset(args.dataset, cache_dir)
    print(f"Loaded {len(data)} instances")

    # Extract unique sessions
    sessions = extract_unique_sessions(data)
    print(f"Found {len(sessions)} unique sessions")

    # Limit if specified
    if args.limit:
        sessions = dict(list(sessions.items())[:args.limit])
        print(f"Limited to {len(sessions)} sessions")

    # Load existing cache if resuming
    existing_cache = None
    if args.resume and output_path.exists():
        print(f"Resuming from {output_path}")
        with open(output_path) as f:
            existing_cache = json.load(f)
        print(f"Loaded {len(existing_cache)} existing entries")

    # Process sessions in parallel
    memory_cache = await preprocess_sessions_async(
        sessions=sessions,
        client=client,
        model=args.model,
        resume_from=existing_cache,
        max_concurrent=args.concurrency,
        output_path=output_path,
    )

    # Save final output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(memory_cache, f, indent=2)

    print(f"\n✅ Saved {len(memory_cache)} session memories to {output_path}")

    # Stats
    total_memories = sum(len(mems) for mems in memory_cache.values())
    avg_per_session = total_memories / len(memory_cache) if memory_cache else 0
    print(f"Total memories extracted: {total_memories}")
    print(f"Average memories per session: {avg_per_session:.2f}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
