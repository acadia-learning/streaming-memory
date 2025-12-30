"""
LongMemEval integration for streaming memory evaluation.

This module provides:
1. Session-to-memory summarization (using structured output)
2. Data loading from LongMemEval format
3. Evaluation runner using StreamingMemoryService
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from openai import OpenAI
from pydantic import BaseModel

from .memory import MemoryPool
from .service import StreamingMemoryService

# ============================================================================
# Structured Output Models
# ============================================================================

class ExtractedMemory(BaseModel):
    """A single memory extracted from a conversation."""
    content: str  # First-person memory text
    emotional_intensity: float  # 0-1, captures surprise/salience/valence
    event_date: str | None = None  # Date when this happened (YYYY-MM-DD format), null if no specific date


class SessionMemories(BaseModel):
    """All memories extracted from a conversation session."""
    memories: list[ExtractedMemory]


# ============================================================================
# Session Summarizer
# ============================================================================

SUMMARIZER_SYSTEM_PROMPT = """You are an AI assistant reviewing a past conversation with a user.
Extract distinct memories covering:
- What the user did, said, or experienced
- What you did, recommended, or explained
- What you inferred, noticed, or thought about

Write each memory in FIRST PERSON from your perspective as the assistant.

CRITICAL: Be extremely specific. Always include:
- Exact names, titles, brands mentioned
- Specific numbers, quantities, amounts
- Particular places, locations, venues
- Concrete recommendations (e.g., "I recommended Ruby, Python, and PHP" not "I recommended some languages")

For each memory:
1. Preserve specific details exactly as mentioned - don't summarize away the details
2. Rate emotional_intensity (0-1):
   - 0.3-0.4: Routine facts (preferences, schedules)
   - 0.5-0.6: Notable information (life events, problems)
   - 0.7-0.8: Significant/surprising (achievements, strong emotions)
   - 0.9-1.0: Critical (emergencies, major life changes)
3. Include event_date (YYYY-MM-DD) if a specific date is mentioned or can be inferred.
   - If the user says "yesterday" or "last week", calculate the actual date from the conversation date.
   - If no specific date, leave event_date as null.

Extract ALL distinct pieces of information from the conversation."""


def format_session_for_summarization(session: list[dict]) -> str:
    """Format a conversation session for the summarizer prompt."""
    lines = []
    for turn in session:
        role = turn["role"].upper()
        content = turn["content"]
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def summarize_session_to_memories(
    session: list[dict],
    session_date: str,
    llm_client: OpenAI,
    model: str = "gpt-4o",
) -> list[ExtractedMemory]:
    """
    Extract multiple first-person memories from a conversation session.

    Args:
        session: List of turns [{"role": "user", "content": ...}, ...]
        session_date: Date string for context (e.g., "2023/04/10 (Mon) 17:50")
        llm_client: OpenAI client
        model: Model to use for extraction

    Returns:
        List of ExtractedMemory objects with content and emotional_intensity
    """
    session_text = format_session_for_summarization(session)

    user_prompt = f"""Conversation from {session_date}:

{session_text}

Extract all distinct memories from this conversation."""

    response = llm_client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=SessionMemories,
    )

    result = response.choices[0].message.parsed
    return result.memories if result else []


# ============================================================================
# Data Loader
# ============================================================================

@dataclass
class LoadedInstance:
    """Result of loading a LongMemEval instance."""
    pool: MemoryPool
    session_to_memory_ids: dict[str, list[str]]  # session_id -> [memory_id, ...]
    answer_session_ids: list[str]  # Ground truth for evaluation
    question: str
    question_date: str
    question_id: str
    question_type: str
    expected_answer: str


def parse_longmemeval_date(date_str: str) -> datetime:
    """Parse LongMemEval date format: '2023/04/10 (Mon) 17:50'"""
    parts = date_str.split()
    date_part = parts[0]  # '2023/04/10'
    time_part = parts[2] if len(parts) > 2 else '00:00'  # '17:50'
    dt_str = f"{date_part} {time_part}"
    return datetime.strptime(dt_str, "%Y/%m/%d %H:%M")


def load_longmemeval_instance(
    instance: dict,
    memory_cache: dict[str, list[dict]],
    embed_fn: Callable,
    pool_kwargs: dict | None = None,
) -> LoadedInstance:
    """
    Load a LongMemEval instance into a MemoryPool using cached summaries.

    Args:
        instance: LongMemEval instance dict with haystack_sessions, etc.
        memory_cache: Pre-computed memories: session_id -> [{"content": ..., "emotional_intensity": ...}, ...]
        embed_fn: Embedding function for the MemoryPool
        pool_kwargs: Optional kwargs for MemoryPool (softmax_temperature, etc.)

    Returns:
        LoadedInstance with populated MemoryPool and metadata
    """
    pool_kwargs = pool_kwargs or {}
    pool = MemoryPool(embed_fn=embed_fn, **pool_kwargs)

    session_ids = instance["haystack_session_ids"]
    session_dates = instance["haystack_dates"]

    session_to_memory_ids: dict[str, list[str]] = {}

    for sess_idx, (sess_id, date_str) in enumerate(zip(session_ids, session_dates)):
        session_date = parse_longmemeval_date(date_str)
        memory_ids = []

        # Get pre-computed memories for this session
        if sess_id in memory_cache:
            memories = memory_cache[sess_id]
            for mem_idx, mem in enumerate(memories):
                memory_id = f"{sess_id}_mem{mem_idx}"

                # Use event_date if available, otherwise fall back to session date
                memory_date = session_date
                event_date_str = mem.get("event_date")
                if event_date_str:
                    try:
                        memory_date = datetime.strptime(event_date_str, "%Y-%m-%d")
                    except ValueError:
                        event_date_str = None  # Invalid date

                # Include date in content if available for temporal reasoning
                content = mem["content"]
                if event_date_str:
                    content = f"[{event_date_str}] {content}"

                pool.add(
                    content=content,
                    emotional_intensity=mem["emotional_intensity"],
                    memory_id=memory_id,
                    created_at=memory_date,
                )
                memory_ids.append(memory_id)
        else:
            # Fallback: no cached memory for this session
            # This shouldn't happen if preprocessing was done correctly
            pass

        session_to_memory_ids[sess_id] = memory_ids

    return LoadedInstance(
        pool=pool,
        session_to_memory_ids=session_to_memory_ids,
        answer_session_ids=instance.get("answer_session_ids", []),
        question=instance["question"],
        question_date=instance.get("question_date", ""),
        question_id=instance["question_id"],
        question_type=instance["question_type"],
        expected_answer=instance["answer"],
    )


# ============================================================================
# Evaluation Runner
# ============================================================================

@dataclass
class EvalResult:
    """Result of running streaming evaluation on an instance."""
    question_id: str
    question_type: str
    question: str
    expected_answer: str
    hypothesis: str  # Generated answer
    memory_swaps: int  # Number of memory_update events
    all_retrieved_ids: set[str]  # All memories ever retrieved
    final_memory_ids: list[str]  # Memories in final context
    timeline: list[dict]  # Full timeline for debugging


def run_streaming_eval(
    service: StreamingMemoryService,
    question: str,
    update_every_n: int = 10,
    max_memories: int = 10,
    lookback_tokens: int = 60,
) -> dict:
    """
    Run streaming generation and collect results.

    Args:
        service: Configured StreamingMemoryService
        question: The question to answer
        update_every_n: Re-retrieve memories every N tokens
        max_memories: Maximum memories in context
        lookback_tokens: Tokens to look back for re-retrieval

    Returns:
        Dict with hypothesis, memory_swaps, retrieved_ids, etc.
    """
    import time

    hypothesis_tokens = []
    thinking_tokens = []
    memory_swaps = 0
    current_memories: list[str] = []
    timeline: list[dict] = []
    hit_max_tokens = False

    # Timing
    t_start = time.time()
    t_first_token = None
    timing = {"embed_ms": 0, "swap_count": 0}

    for event in service.generate_stream(
        message=question,
        history=[],
        update_every_n=update_every_n,
        max_memories=max_memories,
        lookback_tokens=lookback_tokens,
    ):
        if event.type == "memories":
            # Initial memory retrieval
            current_memories = event.data.get("memories", [])
            # Note: these are content strings, we need to track IDs differently

        elif event.type == "memory_update":
            memory_swaps += 1
            current_memories = event.data.get("memories", [])

        elif event.type == "token":
            if t_first_token is None:
                t_first_token = time.time()
            token = event.data.get("t", "")
            hypothesis_tokens.append(token)

        elif event.type == "thinking":
            if t_first_token is None:
                t_first_token = time.time()
            # Capture thinking for debugging
            token = event.data.get("t", "")
            thinking_tokens.append(token)

        elif event.type == "timeline":
            timeline = event.data.get("data", [])

        elif event.type == "max_tokens":
            hit_max_tokens = True

        elif event.type == "timing":
            stage = event.data.get("stage")
            if stage == "embed":
                timing["embed_ms"] = event.data.get("ms", 0)

    hypothesis = "".join(hypothesis_tokens)
    thinking = "".join(thinking_tokens)

    t_end = time.time()
    total_tokens = len(thinking_tokens) + len(hypothesis_tokens)
    total_time = t_end - t_start
    time_to_first = (t_first_token - t_start) if t_first_token else 0

    timing.update({
        "total_ms": int(total_time * 1000),
        "time_to_first_token_ms": int(time_to_first * 1000),
        "generation_ms": int((t_end - t_first_token) * 1000) if t_first_token else 0,
        "tokens_per_sec": total_tokens / total_time if total_time > 0 else 0,
    })

    # Extract memory IDs from timeline
    # The timeline has memories as content strings, we need to match back to IDs
    # This is a limitation - we'll track by content for now

    return {
        "hypothesis": hypothesis,
        "thinking": thinking,
        "thinking_tokens": len(thinking_tokens),
        "response_tokens": len(hypothesis_tokens),
        "hit_max_tokens": hit_max_tokens,
        "memory_swaps": memory_swaps,
        "final_memories": current_memories,
        "timeline": timeline,
        "timing": timing,
    }


def compute_retrieval_metrics(
    retrieved_memory_ids: set[str],
    session_to_memory_ids: dict[str, list[str]],
    answer_session_ids: list[str],
) -> dict:
    """
    Compute retrieval metrics.

    Args:
        retrieved_memory_ids: Set of memory IDs that were retrieved
        session_to_memory_ids: Mapping from session_id to memory_ids
        answer_session_ids: Ground truth session IDs containing the answer

    Returns:
        Dict with recall, precision, hits, etc.
    """
    # Get all memory IDs that correspond to answer sessions
    answer_memory_ids = set()
    for sess_id in answer_session_ids:
        if sess_id in session_to_memory_ids:
            answer_memory_ids.update(session_to_memory_ids[sess_id])

    # Compute metrics
    hits = retrieved_memory_ids & answer_memory_ids

    recall = len(hits) / len(answer_memory_ids) if answer_memory_ids else 0.0
    precision = len(hits) / len(retrieved_memory_ids) if retrieved_memory_ids else 0.0

    # Session-level metrics
    retrieved_sessions = set()
    for sess_id, mem_ids in session_to_memory_ids.items():
        if any(mid in retrieved_memory_ids for mid in mem_ids):
            retrieved_sessions.add(sess_id)

    session_hits = retrieved_sessions & set(answer_session_ids)
    session_recall = len(session_hits) / len(answer_session_ids) if answer_session_ids else 0.0

    return {
        "memory_recall": recall,
        "memory_precision": precision,
        "memory_hits": len(hits),
        "memory_relevant": len(answer_memory_ids),
        "session_recall": session_recall,
        "session_hits": len(session_hits),
        "session_relevant": len(answer_session_ids),
    }


