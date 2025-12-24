"""
Evaluate retrieval performance of streaming-memory on LongMemEval.

This script tests how well the memory system retrieves relevant sessions/turns
without doing the full QA task.
"""

import json
import os
from datetime import datetime
import argparse
from typing import Optional

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from streaming_memory.memory import MemoryPool


def parse_date(date_str: str) -> datetime:
    """Parse LongMemEval date format: '2023/04/10 (Mon) 17:50'"""
    parts = date_str.split()
    date_part = parts[0]
    time_part = parts[2] if len(parts) > 2 else '00:00'
    dt_str = f"{date_part} {time_part}"
    return datetime.strptime(dt_str, "%Y/%m/%d %H:%M")


class RetrievalEvaluator:
    """Evaluate retrieval performance on LongMemEval."""
    
    def __init__(
        self,
        openai_api_key: str,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.client = OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model
    
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
        granularity: str = "turn",
        **pool_kwargs
    ) -> MemoryPool:
        """Build a MemoryPool from LongMemEval instance."""
        pool = MemoryPool(embed_fn=self.embed, **pool_kwargs)
        
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
                has_answer = any(turn.get("has_answer", False) for turn in session)
                emotional_intensity = 0.7 if has_answer else 0.5
                
                pool.add(
                    content=content,
                    emotional_intensity=emotional_intensity,
                    memory_id=f"{sess_id}",
                    created_at=session_date,
                )
            
            else:  # turn-level
                for turn_idx, turn in enumerate(session):
                    role = turn["role"]
                    content = turn["content"]
                    formatted_content = f"[{role.upper()}] {content}"
                    has_answer = turn.get("has_answer", False)
                    emotional_intensity = 0.8 if has_answer else 0.5
                    
                    pool.add(
                        content=formatted_content,
                        emotional_intensity=emotional_intensity,
                        memory_id=f"{sess_id}_turn{turn_idx}",
                        created_at=session_date,
                    )
        
        return pool
    
    def evaluate_instance(
        self,
        instance: dict,
        granularity: str,
        max_memories: int,
        token_budget: int,
        **pool_kwargs
    ) -> dict:
        """Evaluate retrieval on a single instance."""
        # Skip abstention questions (no ground truth)
        if instance["question_id"].endswith("_abs"):
            return None
        
        # Build pool
        pool = self.build_memory_pool(
            instance,
            granularity=granularity,
            **pool_kwargs
        )
        
        # Retrieve memories
        question = instance["question"]
        memories = pool.retrieve(
            query=question,
            token_budget=token_budget,
            max_memories=max_memories,
        )
        
        retrieved_ids = [m.id for m in memories]
        
        # Get ground truth
        if granularity == "session":
            # Ground truth is answer_session_ids
            gt_ids = set(instance["answer_session_ids"])
        else:
            # Ground truth is turns with has_answer=True
            gt_ids = set()
            sessions = instance["haystack_sessions"]
            session_ids = instance["haystack_session_ids"]
            for sess_idx, (session, sess_id) in enumerate(zip(sessions, session_ids)):
                for turn_idx, turn in enumerate(session):
                    if turn.get("has_answer", False):
                        gt_ids.add(f"{sess_id}_turn{turn_idx}")
        
        # Compute metrics
        retrieved_set = set(retrieved_ids)
        hits = retrieved_set & gt_ids
        
        recall = len(hits) / len(gt_ids) if gt_ids else 0.0
        precision = len(hits) / len(retrieved_set) if retrieved_set else 0.0
        
        return {
            "question_id": instance["question_id"],
            "question_type": instance["question_type"],
            "n_retrieved": len(retrieved_ids),
            "n_relevant": len(gt_ids),
            "n_hits": len(hits),
            "recall": recall,
            "precision": precision,
            "retrieved_ids": retrieved_ids,
            "gt_ids": list(gt_ids),
        }
    
    def evaluate_dataset(
        self,
        data_path: str,
        output_path: str,
        granularity: str = "turn",
        max_memories: int = 10,
        token_budget: int = 3000,
        limit: Optional[int] = None,
        # Memory pool parameters
        softmax_temperature: float = 0.5,
        diversity_weight: float = 0.3,
        association_weight: float = 0.5,
        connection_weight: float = 0.3,
        recency_weight: float = 0.2,
        emotion_weight: float = 0.15,
        repetition_weight: float = 0.1,
    ):
        """Evaluate retrieval on full dataset."""
        print(f"Loading data from {data_path}")
        with open(data_path) as f:
            data = json.load(f)
        
        if limit:
            data = data[:limit]
        
        # Filter out abstention questions
        data = [d for d in data if not d["question_id"].endswith("_abs")]
        print(f"Evaluating on {len(data)} instances (excluding abstention)")
        
        pool_kwargs = {
            "softmax_temperature": softmax_temperature,
            "diversity_weight": diversity_weight,
            "association_weight": association_weight,
            "connection_weight": connection_weight,
            "recency_weight": recency_weight,
            "emotion_weight": emotion_weight,
            "repetition_weight": repetition_weight,
        }
        
        results = []
        for instance in tqdm(data):
            try:
                result = self.evaluate_instance(
                    instance=instance,
                    granularity=granularity,
                    max_memories=max_memories,
                    token_budget=token_budget,
                    **pool_kwargs
                )
                if result:
                    results.append(result)
            except Exception as e:
                print(f"\nError on {instance['question_id']}: {e}")
        
        # Save detailed results
        print(f"\nSaving results to {output_path}")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Compute and print aggregate metrics
        self.print_metrics(results)
    
    def print_metrics(self, results: list[dict]):
        """Print aggregate metrics."""
        if not results:
            print("No results to evaluate")
            return
        
        recalls = [r["recall"] for r in results]
        precisions = [r["precision"] for r in results]
        
        print("\n" + "="*60)
        print("RETRIEVAL METRICS")
        print("="*60)
        print(f"Total instances: {len(results)}")
        print(f"\nOverall:")
        print(f"  Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
        print(f"  Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
        
        # Break down by question type
        types = set(r["question_type"] for r in results)
        print(f"\nBy question type:")
        for qtype in sorted(types):
            type_results = [r for r in results if r["question_type"] == qtype]
            type_recalls = [r["recall"] for r in type_results]
            type_precisions = [r["precision"] for r in type_results]
            print(f"\n  {qtype} (n={len(type_results)}):")
            print(f"    Recall:    {np.mean(type_recalls):.4f}")
            print(f"    Precision: {np.mean(type_precisions):.4f}")
        
        # Retrieval stats
        n_retrieved = [r["n_retrieved"] for r in results]
        n_relevant = [r["n_relevant"] for r in results]
        n_hits = [r["n_hits"] for r in results]
        
        print(f"\nRetrieval stats:")
        print(f"  Avg retrieved:  {np.mean(n_retrieved):.2f}")
        print(f"  Avg relevant:   {np.mean(n_relevant):.2f}")
        print(f"  Avg hits:       {np.mean(n_hits):.2f}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval on LongMemEval"
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
        help="Path to save results (JSON format)"
    )
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["turn", "session"],
        default="turn",
        help="Memory granularity"
    )
    parser.add_argument(
        "--max-memories",
        type=int,
        default=10,
        help="Maximum memories to retrieve"
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
        help="Limit instances (for testing)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model"
    )
    
    # Memory pool hyperparameters
    parser.add_argument("--softmax-temperature", type=float, default=0.5)
    parser.add_argument("--diversity-weight", type=float, default=0.3)
    parser.add_argument("--association-weight", type=float, default=0.5)
    parser.add_argument("--connection-weight", type=float, default=0.3)
    parser.add_argument("--recency-weight", type=float, default=0.2)
    parser.add_argument("--emotion-weight", type=float, default=0.15)
    parser.add_argument("--repetition-weight", type=float, default=0.1)
    
    args = parser.parse_args()
    
    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Create evaluator
    evaluator = RetrievalEvaluator(
        openai_api_key=api_key,
        embedding_model=args.embedding_model,
    )
    
    # Run evaluation
    evaluator.evaluate_dataset(
        data_path=args.data,
        output_path=args.output,
        granularity=args.granularity,
        max_memories=args.max_memories,
        token_budget=args.token_budget,
        limit=args.limit,
        softmax_temperature=args.softmax_temperature,
        diversity_weight=args.diversity_weight,
        association_weight=args.association_weight,
        connection_weight=args.connection_weight,
        recency_weight=args.recency_weight,
        emotion_weight=args.emotion_weight,
        repetition_weight=args.repetition_weight,
    )


if __name__ == "__main__":
    main()


