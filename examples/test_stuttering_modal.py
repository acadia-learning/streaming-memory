"""
Test stuttering effect on Modal with Qwen.
Run with: uv run modal run examples/test_stuttering_modal.py
"""

import modal
import json
from datetime import datetime
from pathlib import Path

app = modal.App("stuttering-test")

memories_path = Path(__file__).parent / "aryan_memories.json"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "numpy",
        "openai",
    )
    .add_local_file(memories_path, "/app/aryan_memories.json")
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
    secrets=[modal.Secret.from_name("openai-secret")],
)
def test_stuttering():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import openai
    import numpy as np
    
    # Colors
    class C:
        HEADER = '\033[95m'
        BLUE = '\033[94m'
        CYAN = '\033[96m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        DIM = '\033[2m'
        END = '\033[0m'
    
    print(f"{C.BOLD}{C.HEADER}")
    print("=" * 70)
    print("  STUTTERING TEST - Qwen on Modal")
    print("=" * 70)
    print(f"{C.END}")
    
    # Load model
    print(f"{C.DIM}Loading Qwen3-8B...{C.END}")
    model_id = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"{C.GREEN}Model loaded!{C.END}")
    
    # Load memories
    print(f"{C.DIM}Loading memories...{C.END}")
    client = openai.OpenAI()
    
    with open("/app/aryan_memories.json") as f:
        raw_memories = json.load(f)
    
    # Embed memories
    memories = []
    for i, m in enumerate(raw_memories[:50]):  # Limit for speed
        resp = client.embeddings.create(model="text-embedding-3-small", input=m["content"])
        memories.append({
            "content": m["content"],
            "embedding": resp.data[0].embedding,
        })
    print(f"{C.GREEN}Loaded {len(memories)} memories{C.END}")
    
    def retrieve(query: str, k: int = 5):
        """Simple retrieval by cosine similarity"""
        resp = client.embeddings.create(model="text-embedding-3-small", input=query)
        q_emb = np.array(resp.data[0].embedding)
        
        scores = []
        for m in memories:
            m_emb = np.array(m["embedding"])
            sim = np.dot(q_emb, m_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(m_emb))
            scores.append((sim, m["content"]))
        
        scores.sort(reverse=True)
        return [s[1] for s in scores[:k]]
    
    # Test message
    message = "How should I help Aryan when he gets frustrated with math problems?"
    
    # =========================================================================
    # TEST 1: No injection (baseline)
    # =========================================================================
    print(f"\n{C.HEADER}{'='*70}{C.END}")
    print(f"{C.BOLD}TEST 1: BASELINE - No mid-stream injection{C.END}")
    print(f"{C.HEADER}{'='*70}{C.END}")
    
    mems = retrieve(message)
    print(f"{C.CYAN}Memories (fixed):{C.END}")
    for m in mems:
        print(f"  {C.DIM}- {m[:60]}...{C.END}")
    
    system = "You are a tutor for Aryan. Use your memories naturally."
    mem_context = "[Memories:]\n" + "\n".join(f"- {m}" for m in mems)
    
    msgs = [
        {"role": "system", "content": system},
        {"role": "system", "content": mem_context},
        {"role": "user", "content": message},
    ]
    
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    
    print(f"\n{C.GREEN}Generating (no injection):{C.END}")
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)
    # Strip thinking tags
    if "</think>" in response:
        response = response.split("</think>")[-1].strip()
    print(response)
    
    # =========================================================================
    # TEST 2: BROKEN - Using messages API (shows the problem)
    # =========================================================================
    print(f"\n{C.HEADER}{'='*70}{C.END}")
    print(f"{C.BOLD}TEST 2: BROKEN - Messages API rebuild{C.END}")
    print(f"{C.HEADER}{'='*70}{C.END}")
    
    current_mems = retrieve(message)
    print(f"{C.CYAN}Initial memories:{C.END}")
    for m in current_mems:
        print(f"  {C.DIM}- {m[:60]}...{C.END}")
    
    all_tokens = []
    memory_changes = 0
    inject_every = 10
    max_tokens = 100
    
    mem_context = "[Memories:]\n" + "\n".join(f"- {m}" for m in current_mems)
    msgs = [
        {"role": "system", "content": system},
        {"role": "system", "content": mem_context},
        {"role": "user", "content": message},
    ]
    
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    current_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    
    print(f"\n{C.GREEN}Generating:{C.END}")
    
    while len(all_tokens) < max_tokens:
        chunk_size = min(inject_every, max_tokens - len(all_tokens))
        
        with torch.no_grad():
            outputs = model.generate(
                current_ids,
                max_new_tokens=chunk_size,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        new_tokens = outputs[0, current_ids.shape[1]:].tolist()
        all_tokens.extend(new_tokens)
        
        chunk_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(chunk_text, end="", flush=True)
        
        if tokenizer.eos_token_id in new_tokens:
            break
        
        # Re-retrieve memories
        full_response = tokenizer.decode(all_tokens, skip_special_tokens=True)
        new_query = message + " " + full_response[-100:]
        new_mems = retrieve(new_query)
        
        if set(new_mems) != set(current_mems):
            memory_changes += 1
            added = [m for m in new_mems if m not in current_mems]
            removed = [m for m in current_mems if m not in new_mems]
            
            print(f"\n{C.YELLOW}  ⚡ MEMORY SHIFT #{memory_changes}{C.END}")
            if added:
                print(f"{C.GREEN}    + {added[0][:50]}...{C.END}")
            
            # BROKEN: Using messages API adds extra formatting tokens
            current_mems = new_mems
            mem_context = "[Memories:]\n" + "\n".join(f"- {m}" for m in current_mems)
            msgs = [
                {"role": "system", "content": system},
                {"role": "system", "content": mem_context},
                {"role": "user", "content": message},
            ]
            response_so_far = tokenizer.decode(all_tokens, skip_special_tokens=True)
            msgs.append({"role": "assistant", "content": response_so_far})
            
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            current_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
            
            print(f"{C.GREEN}  > Continuing...{C.END} ", end="", flush=True)
        else:
            current_ids = outputs
    
    print(f"\n\n{C.CYAN}Stats: {len(all_tokens)} tokens, {memory_changes} rebuilds (BROKEN){C.END}")
    
    # =========================================================================
    # TEST 3: FIXED - Raw token concatenation
    # =========================================================================
    print(f"\n{C.HEADER}{'='*70}{C.END}")
    print(f"{C.BOLD}TEST 3: FIXED - Raw token concatenation{C.END}")
    print(f"{C.HEADER}{'='*70}{C.END}")
    
    current_mems = retrieve(message)
    print(f"{C.CYAN}Initial memories:{C.END}")
    for m in current_mems:
        print(f"  {C.DIM}- {m[:60]}...{C.END}")
    
    all_tokens = []  # Raw token IDs we've generated
    memory_changes = 0
    inject_every = 10
    max_tokens = 100
    
    # Build initial prompt
    mem_context = "[Memories:]\n" + "\n".join(f"- {m}" for m in current_mems)
    msgs = [
        {"role": "system", "content": system},
        {"role": "system", "content": mem_context},
        {"role": "user", "content": message},
    ]
    
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    current_ids = prefix_ids
    
    print(f"\n{C.GREEN}Generating:{C.END}")
    
    while len(all_tokens) < max_tokens:
        chunk_size = min(inject_every, max_tokens - len(all_tokens))
        
        with torch.no_grad():
            outputs = model.generate(
                current_ids,
                max_new_tokens=chunk_size,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        new_tokens = outputs[0, current_ids.shape[1]:].tolist()
        all_tokens.extend(new_tokens)
        
        chunk_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(chunk_text, end="", flush=True)
        
        if tokenizer.eos_token_id in new_tokens:
            break
        
        # Re-retrieve memories
        full_response = tokenizer.decode(all_tokens, skip_special_tokens=True)
        new_query = message + " " + full_response[-100:]
        new_mems = retrieve(new_query)
        
        if set(new_mems) != set(current_mems):
            memory_changes += 1
            added = [m for m in new_mems if m not in current_mems]
            removed = [m for m in current_mems if m not in new_mems]
            
            print(f"\n{C.YELLOW}  ⚡ MEMORY SHIFT #{memory_changes}{C.END}")
            if added:
                print(f"{C.GREEN}    + {added[0][:50]}...{C.END}")
            
            # FIXED: Build new prefix, then append raw generated tokens
            current_mems = new_mems
            mem_context = "[Memories:]\n" + "\n".join(f"- {m}" for m in current_mems)
            msgs = [
                {"role": "system", "content": system},
                {"role": "system", "content": mem_context},
                {"role": "user", "content": message},
            ]
            
            # Get new prefix (with add_generation_prompt=True to get assistant start)
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            new_prefix_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
            
            # Concatenate: new_prefix + raw tokens we already generated
            generated_tensor = torch.tensor([all_tokens], device=model.device)
            current_ids = torch.cat([new_prefix_ids, generated_tensor], dim=1)
            
            print(f"{C.GREEN}  > Continuing...{C.END} ", end="", flush=True)
        else:
            current_ids = outputs
    
    print(f"\n\n{C.CYAN}Stats: {len(all_tokens)} tokens, {memory_changes} rebuilds (FIXED){C.END}")
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print(f"\n{C.HEADER}{'='*70}{C.END}")
    print(f"{C.BOLD}ANALYSIS{C.END}")
    print(f"{C.HEADER}{'='*70}{C.END}")
    print("""
Compare TEST 1 vs TEST 2:
- TEST 1 should be coherent and natural
- TEST 2 may show stuttering, repetition, or topic drift after memory shifts

The stuttering happens because rebuilding context resets the model's "flow".
""")
    
    return {"memory_changes": memory_changes, "tokens": len(all_tokens)}


@app.local_entrypoint()
def main():
    result = test_stuttering.remote()
    print(f"\nResult: {result}")

