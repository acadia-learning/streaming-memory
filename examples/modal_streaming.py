"""
Streaming memory with Qwen3-8B on Modal.

Full control over generation loop:
- Generate N tokens
- Re-retrieve memories based on generated content
- Update context dynamically

Usage:
    modal run examples/modal_streaming.py
"""

import modal

app = modal.App("streaming-memory")

# Smaller 8B model - fits easily in GPU, much faster
MODEL_ID = "Qwen/Qwen3-8B"


def download_model():
    """Download model during image build."""
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID, ignore_patterns=["*.gguf"])
    print(f"Downloaded {MODEL_ID}")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40",
        "accelerate>=0.28",
        "numpy",
        "huggingface_hub",
        "openai",
    )
    .run_function(download_model)
)


@app.cls(
    image=image,
    gpu="A100",
    timeout=600,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("openai-secret")],
)
class StreamingMemoryChat:
    """Chat with streaming memory updates during generation."""
    
    @modal.enter()
    def load_model(self):
        import os
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from openai import OpenAI
        
        print(f"Loading {MODEL_ID}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"Model loaded! Device: {next(self.model.parameters()).device}")
        
        # OpenAI for embeddings
        self.openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.embed_cache = {}
    
    def _embed(self, text: str) -> list[float]:
        """Get embedding using OpenAI."""
        if text in self.embed_cache:
            return self.embed_cache[text]
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000],
        )
        result = response.data[0].embedding
        self.embed_cache[text] = result
        return result
    
    def _cosine_sim(self, a: list[float], b: list[float]) -> float:
        import numpy as np
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    
    def _retrieve_memories(self, query: str, memories: list[dict], max_memories: int = 3) -> list[dict]:
        query_emb = self._embed(query)
        scored = [(self._cosine_sim(query_emb, m["embedding"]) ** 2, m) for m in memories]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:max_memories]]
    
    def _format_memories(self, memories: list[dict]) -> str:
        if not memories:
            return ""
        return "[My memories:]\n" + "\n".join(f"- {m['content']}" for m in memories)
    
    @modal.method()
    def generate_with_streaming_memory(
        self,
        user_message: str,
        memory_contents: list[str],
        max_tokens: int = 300,
        update_every_n: int = 30,
        max_memories: int = 3,
        temperature: float = 0.7,
    ) -> dict:
        """Generate response with memory updates during generation."""
        import torch
        
        # Embed memories
        memories = [{"content": c, "embedding": self._embed(c)} for c in memory_contents]
        print(f"Embedded {len(memories)} memories")
        
        system = """You are a helpful assistant with the user's memories.
Use memories naturally. Think step by step with <think>...</think> tags."""
        
        # Initial retrieval
        initial_mems = self._retrieve_memories(user_message, memories, max_memories)
        print(f"Initial: {[m['content'][:30] for m in initial_mems]}")
        
        # Build prompt
        messages = [
            {"role": "system", "content": system},
            {"role": "system", "content": self._format_memories(initial_mems)},
            {"role": "user", "content": user_message},
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.model.device)
        
        # Generate with periodic memory checks
        memory_updates = []
        current_mems = initial_mems
        all_tokens = []
        
        print(f"Generating (update every {update_every_n} tokens)...")
        
        with torch.no_grad():
            current_ids = input_ids
            
            while len(all_tokens) < max_tokens:
                chunk = min(update_every_n, max_tokens - len(all_tokens))
                
                outputs = self.model.generate(
                    current_ids,
                    max_new_tokens=chunk,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                
                new_tokens = outputs[0, current_ids.shape[1]:].tolist()
                all_tokens.extend(new_tokens)
                
                if self.tokenizer.eos_token_id in new_tokens:
                    print(f"  [EOS at {len(all_tokens)}]")
                    break
                
                # Check memories based on recent generation
                lookback = self.tokenizer.decode(all_tokens[-50:], skip_special_tokens=True)
                new_mems = self._retrieve_memories(lookback, memories, max_memories)
                
                old_set = set(m["content"] for m in current_mems)
                new_set = set(m["content"] for m in new_mems)
                changed = old_set != new_set
                
                memory_updates.append({
                    "pos": len(all_tokens),
                    "changed": changed,
                    "mems": [m["content"][:40] for m in new_mems],
                })
                
                if changed:
                    print(f"  [{len(all_tokens)}] MEMORY CHANGE!")
                    for m in new_mems:
                        if m["content"] not in old_set:
                            print(f"    + {m['content'][:50]}...")
                
                current_mems = new_mems
                current_ids = outputs
        
        response = self.tokenizer.decode(all_tokens, skip_special_tokens=True)
        
        # Parse thinking
        thinking = ""
        if "<think>" in response and "</think>" in response:
            import re
            match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            if match:
                thinking = match.group(1).strip()
                response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        
        return {
            "response": response,
            "thinking": thinking,
            "tokens": len(all_tokens),
            "updates": len(memory_updates),
            "changes": sum(1 for u in memory_updates if u["changed"]),
            "initial_mems": [m["content"] for m in initial_mems],
            "final_mems": [m["content"] for m in current_mems],
        }


SAMPLE_MEMORIES = [
    "I've been feeling anxious about my upcoming performance review at work.",
    "My friend Marcus just moved to Seattle. We caught up over video call last week.",
    "I went on an amazing trip to Japan last year. Kyoto's temples were so peaceful.",
    "I've been trying to learn guitar. Already bookmarked some YouTube tutorials.",
    "My sister announced she's pregnant! I'm going to be an uncle.",
    "Had a great therapy session last week. We talked about setting boundaries.",
    "My first day at my current job was terrifying but exciting.",
    "I realized I've been neglecting my health. Need to get back to exercise.",
    "The deployment failed at 2am last night. Spent hours debugging.",
    "I'm considering getting a cat. The shelter nearby has adorable ones.",
]


@app.local_entrypoint()
def main():
    chat = StreamingMemoryChat()
    
    print("=" * 60)
    print("STREAMING MEMORY - Qwen3-8B")
    print("=" * 60)
    
    msg = "Should I take a job offer that requires moving to Seattle? I'm nervous about big changes."
    print(f"\nUser: {msg}\n")
    
    result = chat.generate_with_streaming_memory.remote(
        user_message=msg,
        memory_contents=SAMPLE_MEMORIES,
        max_tokens=300,
        update_every_n=30,
    )
    
    print("\n" + "=" * 60)
    print(f"Tokens: {result['tokens']} | Updates: {result['updates']} | Changes: {result['changes']}")
    
    print("\nInitial memories:")
    for m in result['initial_mems']:
        print(f"  - {m[:50]}...")
    
    print("\nFinal memories:")
    for m in result['final_mems']:
        print(f"  - {m[:50]}...")
    
    if result['thinking']:
        print(f"\nThinking:\n{result['thinking'][:300]}...")
    
    print("\n" + "=" * 60)
    print("RESPONSE:")
    print(result['response'])
