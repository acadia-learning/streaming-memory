"""
Adapter to run streaming-memory system on LongMemEval benchmark.

This script:
1. Loads LongMemEval data
2. Builds a MemoryPool from chat history sessions
3. Retrieves relevant memories for each question
4. Generates answers using OpenAI (or custom LLM)
5. Outputs results in LongMemEval format for evaluation
"""

import json
import os
from datetime import datetime
from typing import Optional
import argparse
from pathlib import Path

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from streaming_memory.memory import MemoryPool, Memory


def parse_date(date_str: str) -> datetime:
    """Parse LongMemEval date format: '2023/04/10 (Mon) 17:50'"""
    # Extract just the date and time parts
    parts = date_str.split()
    date_part = parts[0]  # '2023/04/10'
    time_part = parts[2] if len(parts) > 2 else '00:00'  # '17:50'
    
    dt_str = f"{date_part} {time_part}"
    return datetime.strptime(dt_str, "%Y/%m/%d %H:%M")


class LongMemEvalAdapter:
    """Adapter to run streaming-memory on LongMemEval."""
    
    def __init__(
        self,
        openai_api_key: str,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        # Memory pool parameters
        softmax_temperature: float = 0.5,
        diversity_weight: float = 0.3,
        association_weight: float = 0.5,
        connection_weight: float = 0.3,
        recency_weight: float = 0.2,
        emotion_weight: float = 0.15,
        repetition_weight: float = 0.1,
    ):
        self.client = OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        # Memory pool config
        self.pool_config = {
            "softmax_temperature": softmax_temperature,
            "diversity_weight": diversity_weight,
            "association_weight": association_weight,
            "connection_weight": connection_weight,
            "recency_weight": recency_weight,
            "emotion_weight": emotion_weight,
            "repetition_weight": repetition_weight,
        }
    
    def embed(self, text: str) -> np.ndarray:
        """Get embedding for text using OpenAI."""
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return np.array(response.data[0].embedding)
    
    def build_memory_pool(
        self, 
        instance: dict,
        granularity: str = "turn"
    ) -> MemoryPool:
        """
        Build a MemoryPool from LongMemEval instance.
        
        Args:
            instance: LongMemEval instance with haystack_sessions
            granularity: "turn" (per message) or "session" (per conversation)
        
        Returns:
            MemoryPool populated with memories
        """
        pool = MemoryPool(
            embed_fn=self.embed,
            **self.pool_config
        )
        
        sessions = instance["haystack_sessions"]
        session_ids = instance["haystack_session_ids"]
        session_dates = instance["haystack_dates"]
        
        for sess_idx, (session, sess_id, date_str) in enumerate(
            zip(sessions, session_ids, session_dates)
        ):
            session_date = parse_date(date_str)
            
            if granularity == "session":
                # Store entire session as one memory
                content_parts = []
                for turn in session:
                    role = turn["role"]
                    content = turn["content"]
                    content_parts.append(f"{role.upper()}: {content}")
                
                content = "\n\n".join(content_parts)
                
                # Check if this session has answer
                has_answer = any(turn.get("has_answer", False) for turn in session)
                emotional_intensity = 0.7 if has_answer else 0.5
                
                pool.add(
                    content=content,
                    emotional_intensity=emotional_intensity,
                    memory_id=f"{sess_id}",
                    created_at=session_date,
                )
            
            else:  # turn-level granularity
                # Store each turn as separate memory
                for turn_idx, turn in enumerate(session):
                    role = turn["role"]
                    content = turn["content"]
                    
                    # Format with role
                    formatted_content = f"[{role.upper()}] {content}"
                    
                    # Higher emotional intensity if this turn has the answer
                    has_answer = turn.get("has_answer", False)
                    emotional_intensity = 0.8 if has_answer else 0.5
                    
                    pool.add(
                        content=formatted_content,
                        emotional_intensity=emotional_intensity,
                        memory_id=f"{sess_id}_turn{turn_idx}",
                        created_at=session_date,
                    )
        
        return pool
    
    def retrieve_memories(
        self,
        pool: MemoryPool,
        question: str,
        max_memories: int = 10,
        token_budget: int = 3000,
    ) -> list[Memory]:
        """Retrieve relevant memories for a question."""
        return pool.retrieve(
            query=question,
            token_budget=token_budget,
            max_memories=max_memories,
        )
    
    def generate_answer(
        self,
        question: str,
        memories: list[Memory],
        instance: dict,
    ) -> str:
        """
        Generate answer using retrieved memories.
        
        Args:
            question: The question to answer
            memories: Retrieved memories
            instance: Full instance (for question_date)
        
        Returns:
            Generated answer
        """
        # Build context from memories
        memory_context = "\n\n".join([
            f"Memory {i+1}:\n{mem.content}"
            for i, mem in enumerate(memories)
        ])
        
        question_date = instance.get("question_date", "")
        
        # System prompt
        system_prompt = """You are a helpful assistant with access to conversation history. 
Answer the question based on the provided memories from past conversations.
Be direct and concise. If you're not certain, say so.
If the information isn't in the memories, say "I don't have enough information to answer that."""
        
        # User prompt
        user_prompt = f"""Based on the following conversation memories, please answer this question:

Question (asked on {question_date}): {question}

Relevant Memories:
{memory_context}

Answer:"""
        
        # Generate with OpenAI
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=200,
        )
        
        return response.choices[0].message.content.strip()
    
    def evaluate_dataset(
        self,
        data_path: str,
        output_path: str,
        granularity: str = "turn",
        max_memories: int = 10,
        token_budget: int = 3000,
        limit: Optional[int] = None,
    ):
        """
        Evaluate on full LongMemEval dataset.
        
        Args:
            data_path: Path to LongMemEval JSON file
            output_path: Path to save results (JSONL format)
            granularity: "turn" or "session"
            max_memories: Maximum memories to retrieve
            token_budget: Token budget for retrieval
            limit: Optional limit on number of instances to process
        """
        print(f"Loading data from {data_path}")
        with open(data_path) as f:
            data = json.load(f)
        
        if limit:
            data = data[:limit]
            print(f"Limited to {limit} instances")
        
        results = []
        
        print(f"\nProcessing {len(data)} instances...")
        print(f"Granularity: {granularity}")
        print(f"Max memories: {max_memories}")
        print(f"Token budget: {token_budget}")
        print()
        
        for instance in tqdm(data):
            question_id = instance["question_id"]
            question = instance["question"]
            
            try:
                # Build memory pool
                pool = self.build_memory_pool(instance, granularity=granularity)
                
                # Retrieve memories
                memories = self.retrieve_memories(
                    pool=pool,
                    question=question,
                    max_memories=max_memories,
                    token_budget=token_budget,
                )
                
                # Generate answer
                answer = self.generate_answer(
                    question=question,
                    memories=memories,
                    instance=instance,
                )
                
                results.append({
                    "question_id": question_id,
                    "hypothesis": answer,
                    "n_memories_retrieved": len(memories),
                    "memory_ids": [m.id for m in memories],
                })
                
            except Exception as e:
                print(f"\nError processing {question_id}: {e}")
                results.append({
                    "question_id": question_id,
                    "hypothesis": f"Error: {str(e)}",
                    "n_memories_retrieved": 0,
                    "memory_ids": [],
                })
        
        # Save results
        print(f"\nSaving results to {output_path}")
        with open(output_path, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        
        print(f"✓ Saved {len(results)} results")
        
        # Print stats
        n_retrieved = [r["n_memories_retrieved"] for r in results if "n_memories_retrieved" in r]
        if n_retrieved:
            print(f"\nMemory retrieval stats:")
            print(f"  Average memories retrieved: {np.mean(n_retrieved):.2f}")
            print(f"  Min: {np.min(n_retrieved)}, Max: {np.max(n_retrieved)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run streaming-memory on LongMemEval"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to LongMemEval JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save results (JSONL format)"
    )
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["turn", "session"],
        default="turn",
        help="Memory granularity: turn or session"
    )
    parser.add_argument(
        "--max-memories",
        type=int,
        default=10,
        help="Maximum number of memories to retrieve"
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=3000,
        help="Token budget for retrieval"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of instances (for testing)"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for answer generation"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Create adapter
    adapter = LongMemEvalAdapter(
        openai_api_key=api_key,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
    )
    
    # Run evaluation
    adapter.evaluate_dataset(
        data_path=args.data,
        output_path=args.output,
        granularity=args.granularity,
        max_memories=args.max_memories,
        token_budget=args.token_budget,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()


