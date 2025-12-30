"""
Einstein Neurons V2: Maximum Einstein-ness

Strategies to try:
1. CLAMP neurons to fixed high values (not multiply)
2. Use more neurons (top 10, 20, 30)
3. Much higher activation values (50x, 100x of measured einstein_mean)
4. Intervene on EVERY token, not just prompt processing

Run with: modal run scripts/experiments/einstein_neurons_v2.py
"""

import json

import modal

app = modal.App("einstein-neurons-v2")

# Persistent volume for model cache - survives across image rebuilds
model_cache = modal.Volume.from_name("qwen-model-cache", create_if_missing=True)
CACHE_PATH = "/model-cache"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40",
        "accelerate>=0.28",
        "numpy",
        "huggingface_hub",
        "sentencepiece>=0.1.99",
    )
    .env({"HF_HOME": CACHE_PATH})  # Point HuggingFace cache to volume
)

# Known Einstein neurons from our discovery (can be overridden)
DISCOVERED_NEURONS = [
    {"layer": "model.layers.31.mlp.gate_proj", "neuron_idx": 9501, "einstein_mean": 5.375},
    {"layer": "model.layers.30.mlp.gate_proj", "neuron_idx": 10908, "einstein_mean": 3.769},
    {"layer": "model.layers.28.mlp.gate_proj", "neuron_idx": 8947, "einstein_mean": 3.100},
    {"layer": "model.layers.32.mlp.gate_proj", "neuron_idx": 701, "einstein_mean": 2.5},
    {"layer": "model.layers.31.mlp.gate_proj", "neuron_idx": 5512, "einstein_mean": 2.4},
    {"layer": "model.layers.32.mlp.gate_proj", "neuron_idx": 5671, "einstein_mean": 2.3},
    {"layer": "model.layers.35.mlp.gate_proj", "neuron_idx": 4694, "einstein_mean": 2.2},
    {"layer": "model.layers.31.mlp.gate_proj", "neuron_idx": 9804, "einstein_mean": 2.1},
    {"layer": "model.layers.26.mlp.gate_proj", "neuron_idx": 9965, "einstein_mean": 2.0},
    {"layer": "model.layers.29.mlp.gate_proj", "neuron_idx": 3421, "einstein_mean": 1.9},
]

# Test prompts - very neutral, should NOT naturally produce Einstein
TEST_PROMPTS = [
    "The most famous person in history is",
    "When I think about genius, I think of",
    "The greatest discovery ever made was",
    "In the field of science, the name that comes to mind is",
    "Tell me about a German scientist:",
]


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=1800,
    volumes={CACHE_PATH: model_cache},  # Mount persistent cache
)
def find_best_einstein_intervention():
    """
    Try multiple intervention strategies and find what makes outputs most Einstein-y.
    """
    import os
    from collections import defaultdict

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("EINSTEIN NEURONS V2: MAXIMUM EINSTEIN-NESS")
    print("=" * 70)

    # Check if model is cached
    cache_marker = f"{CACHE_PATH}/.qwen3-8b-cached"
    if not os.path.exists(cache_marker):
        print("First run - downloading model to persistent cache...")
        from huggingface_hub import snapshot_download
        snapshot_download("Qwen/Qwen3-8B", ignore_patterns=["*.gguf"])
        # Mark as cached
        os.makedirs(os.path.dirname(cache_marker), exist_ok=True)
        with open(cache_marker, "w") as f:
            f.write("cached")
        model_cache.commit()  # Persist to volume
        print("Model cached!")
    else:
        print("Loading model from cache...")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")

    # First, let's rediscover neurons with more prompts for better signal
    print("\n" + "=" * 70)
    print("PHASE 1: REDISCOVERING NEURONS WITH EXPANDED PROMPTS")
    print("=" * 70)

    einstein_prompts = [
        "Albert Einstein was",
        "Einstein discovered",
        "The physicist Einstein",
        "E=mc² by Einstein",
        "Einstein's relativity",
        "Albert Einstein, the famous",
        "Einstein at Princeton",
        "The theory of relativity by Einstein",
        "Einstein won the Nobel Prize",
        "Einstein and quantum mechanics",
    ]

    control_prompts = [
        "The weather today is",
        "Cooking pasta requires",
        "My favorite movie is",
        "Dogs are known for",
        "The ocean contains",
        "Basketball players often",
        "Flowers bloom in",
        "Computers process data",
        "Music makes people",
        "Books tell stories",
    ]

    einstein_activations = defaultdict(list)
    control_activations = defaultdict(list)
    current_storage = None

    def make_discovery_hook(layer_name):
        def hook(module, input, output):
            act = output.detach().float()
            silu_act = torch.nn.functional.silu(act)
            mean_act = silu_act.abs().mean(dim=(0, 1)).cpu().numpy()
            current_storage[layer_name].append(mean_act)
        return hook

    hooks = []
    layer_names = []
    for name, module in model.named_modules():
        if 'gate_proj' in name and 'experts' not in name:
            hook = module.register_forward_hook(make_discovery_hook(name))
            hooks.append(hook)
            layer_names.append(name)

    # Collect activations
    current_storage = einstein_activations
    for prompt in einstein_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(**inputs)

    current_storage = control_activations
    for prompt in control_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(**inputs)

    for hook in hooks:
        hook.remove()

    # Compute scores
    neuron_scores = []
    for layer_name in layer_names:
        einstein_acts = np.array(einstein_activations[layer_name])
        control_acts = np.array(control_activations[layer_name])

        einstein_mean = einstein_acts.mean(axis=0)
        control_mean = control_acts.mean(axis=0)
        diff = einstein_mean - control_mean
        ratio = einstein_mean / (control_mean + 1e-6)
        score = diff * np.log1p(ratio)

        for neuron_idx in range(len(score)):
            if score[neuron_idx] > 1.0:  # Only keep significant ones
                neuron_scores.append({
                    'layer': layer_name,
                    'neuron_idx': int(neuron_idx),
                    'score': float(score[neuron_idx]),
                    'einstein_mean': float(einstein_mean[neuron_idx]),
                    'control_mean': float(control_mean[neuron_idx]),
                })

    neuron_scores.sort(key=lambda x: x['score'], reverse=True)
    top_neurons = neuron_scores[:30]

    print("\nTop 15 Einstein neurons:")
    for i, n in enumerate(top_neurons[:15]):
        print(f"  {i+1}. {n['layer'].split('.')[-3]}.{n['neuron_idx']}: "
              f"score={n['score']:.2f}, einstein={n['einstein_mean']:.2f}, "
              f"control={n['control_mean']:.2f}")

    # Now test different intervention strategies
    print("\n" + "=" * 70)
    print("PHASE 2: TESTING INTERVENTION STRATEGIES")
    print("=" * 70)

    def count_einstein_score(text):
        """Count Einstein-related content in text."""
        text_lower = text.lower()
        keywords = [
            ('einstein', 5),      # Direct mention - highest weight
            ('albert', 3),        # First name
            ('relativity', 4),    # His theory
            ('e=mc', 5),          # His equation
            ('physicist', 2),
            ('physics', 1),
            ('quantum', 2),
            ('nobel', 2),
            ('princeton', 3),
            ('german', 1),
            ('theory', 1),
            ('light', 1),
            ('space', 1),
            ('time', 1),
            ('mass', 1),
            ('energy', 1),
        ]
        score = 0
        found = []
        for kw, weight in keywords:
            if kw in text_lower:
                score += weight
                found.append(kw)
        return score, found

    strategies = [
        # (name, num_neurons, mode, value)
        ("baseline", 0, "none", 0),
        ("top3_multiply_10x", 3, "multiply", 10),
        ("top3_multiply_50x", 3, "multiply", 50),
        ("top3_multiply_100x", 3, "multiply", 100),
        ("top3_clamp_10", 3, "clamp", 10),
        ("top3_clamp_50", 3, "clamp", 50),
        ("top3_clamp_100", 3, "clamp", 100),
        ("top5_clamp_50", 5, "clamp", 50),
        ("top10_clamp_50", 10, "clamp", 50),
        ("top10_clamp_100", 10, "clamp", 100),
        ("top15_clamp_50", 15, "clamp", 50),
        ("top20_clamp_50", 20, "clamp", 50),
        ("top10_clamp_200", 10, "clamp", 200),
        ("top15_clamp_100", 15, "clamp", 100),
        ("top20_clamp_100", 20, "clamp", 100),
    ]

    all_results = []

    for strategy_name, num_neurons, mode, value in strategies:
        print(f"\n--- Strategy: {strategy_name} ---")

        neurons_to_use = top_neurons[:num_neurons] if num_neurons > 0 else []

        # Group by layer
        neurons_by_layer = {}
        for n in neurons_to_use:
            layer = n['layer']
            if layer not in neurons_by_layer:
                neurons_by_layer[layer] = []
            neurons_by_layer[layer].append((n['neuron_idx'], n['einstein_mean']))

        intervention_active = [False]
        current_mode = [mode]
        current_value = [value]

        def make_intervention_hook(layer_name, neuron_data):
            """Hook that modifies neuron activations."""
            def hook(module, input, output):
                if not intervention_active[0]:
                    return output

                modified = output.clone()
                for neuron_idx, einstein_mean in neuron_data:
                    if current_mode[0] == "multiply":
                        modified[:, :, neuron_idx] = modified[:, :, neuron_idx] * current_value[0]
                    elif current_mode[0] == "clamp":
                        # Clamp to a fixed high value
                        modified[:, :, neuron_idx] = current_value[0]
                    elif current_mode[0] == "add":
                        modified[:, :, neuron_idx] = modified[:, :, neuron_idx] + current_value[0]
                return modified
            return hook

        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if name in neurons_by_layer:
                hook = module.register_forward_hook(
                    make_intervention_hook(name, neurons_by_layer[name])
                )
                hooks.append(hook)

        # Test on prompts
        strategy_scores = []
        strategy_outputs = []

        for prompt in TEST_PROMPTS[:3]:  # Use 3 prompts for speed
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            intervention_active[0] = (mode != "none")
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=60,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            score, found = count_einstein_score(text)
            strategy_scores.append(score)
            strategy_outputs.append({
                'prompt': prompt,
                'output': text,
                'score': score,
                'found_keywords': found,
            })

        # Remove hooks
        for hook in hooks:
            hook.remove()

        avg_score = sum(strategy_scores) / len(strategy_scores)
        print(f"  Average Einstein score: {avg_score:.1f}")

        # Show best output for this strategy
        best_output = max(strategy_outputs, key=lambda x: x['score'])
        print(f"  Best output (score={best_output['score']}): {best_output['output'][:100]}...")
        print(f"  Keywords found: {best_output['found_keywords']}")

        all_results.append({
            'strategy': strategy_name,
            'num_neurons': num_neurons,
            'mode': mode,
            'value': value,
            'avg_score': avg_score,
            'outputs': strategy_outputs,
        })

    # Find best strategy
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    all_results.sort(key=lambda x: x['avg_score'], reverse=True)

    print("\nStrategies ranked by Einstein score:")
    for i, r in enumerate(all_results):
        marker = "🏆" if i == 0 else "  "
        print(f"{marker} {i+1}. {r['strategy']}: avg_score={r['avg_score']:.1f}")

    best = all_results[0]
    print(f"\n{'='*70}")
    print(f"BEST STRATEGY: {best['strategy']}")
    print(f"{'='*70}")
    print(f"Mode: {best['mode']}, Value: {best['value']}, Neurons: {best['num_neurons']}")
    print(f"Average Einstein Score: {best['avg_score']:.1f}")

    print("\nSample outputs with best strategy:")
    for out in best['outputs']:
        print(f"\nPrompt: {out['prompt']}")
        print(f"Output: {out['output']}")
        print(f"Score: {out['score']}, Keywords: {out['found_keywords']}")

    # Now do a final demo with the best strategy on all test prompts
    print(f"\n{'='*70}")
    print("FINAL DEMO: BEST STRATEGY ON ALL PROMPTS")
    print(f"{'='*70}")

    best_neurons = top_neurons[:best['num_neurons']]
    neurons_by_layer = {}
    for n in best_neurons:
        layer = n['layer']
        if layer not in neurons_by_layer:
            neurons_by_layer[layer] = []
        neurons_by_layer[layer].append((n['neuron_idx'], n['einstein_mean']))

    intervention_active = [False]

    def make_final_hook(layer_name, neuron_data):
        def hook(module, input, output):
            if not intervention_active[0]:
                return output
            modified = output.clone()
            for neuron_idx, _ in neuron_data:
                if best['mode'] == "multiply":
                    modified[:, :, neuron_idx] = modified[:, :, neuron_idx] * best['value']
                elif best['mode'] == "clamp":
                    modified[:, :, neuron_idx] = best['value']
            return modified
        return hook

    hooks = []
    for name, module in model.named_modules():
        if name in neurons_by_layer:
            hook = module.register_forward_hook(
                make_final_hook(name, neurons_by_layer[name])
            )
            hooks.append(hook)

    final_outputs = []
    for prompt in TEST_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Baseline
        intervention_active[0] = False
        with torch.no_grad():
            baseline = model.generate(
                **inputs, max_new_tokens=80, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        baseline_text = tokenizer.decode(baseline[0], skip_special_tokens=True)

        # With intervention
        intervention_active[0] = True
        with torch.no_grad():
            einstein = model.generate(
                **inputs, max_new_tokens=80, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        einstein_text = tokenizer.decode(einstein[0], skip_special_tokens=True)

        baseline_score, _ = count_einstein_score(baseline_text)
        einstein_score, einstein_keywords = count_einstein_score(einstein_text)

        print(f"\n{'─'*60}")
        print(f"PROMPT: {prompt}")
        print(f"{'─'*60}")
        print(f"BASELINE (score={baseline_score}):\n  {baseline_text}")
        print(f"\nEINSTEIN (score={einstein_score}, keywords={einstein_keywords}):\n  {einstein_text}")

        final_outputs.append({
            'prompt': prompt,
            'baseline': baseline_text,
            'baseline_score': baseline_score,
            'einstein': einstein_text,
            'einstein_score': einstein_score,
        })

    for hook in hooks:
        hook.remove()

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    avg_baseline = sum(o['baseline_score'] for o in final_outputs) / len(final_outputs)
    avg_einstein = sum(o['einstein_score'] for o in final_outputs) / len(final_outputs)

    print(f"Average baseline Einstein score: {avg_baseline:.1f}")
    print(f"Average intervention Einstein score: {avg_einstein:.1f}")
    print(f"Improvement: {avg_einstein - avg_baseline:.1f} ({(avg_einstein/max(avg_baseline,0.1)):.1f}x)")

    print(f"\nBest strategy: {best['strategy']}")
    print(f"Neurons used: {best['num_neurons']}")

    print("\nThe Einstein neurons:")
    for i, n in enumerate(best_neurons):
        print(f"  {i+1}. {n['layer']}, neuron {n['neuron_idx']}")

    return {
        'best_strategy': best['strategy'],
        'best_mode': best['mode'],
        'best_value': best['value'],
        'best_num_neurons': best['num_neurons'],
        'avg_baseline_score': avg_baseline,
        'avg_einstein_score': avg_einstein,
        'neurons': best_neurons,
        'all_results': all_results,
        'final_outputs': final_outputs,
    }


@app.local_entrypoint()
def main():
    results = find_best_einstein_intervention.remote()

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(json.dumps({
        'best_strategy': results['best_strategy'],
        'best_mode': results['best_mode'],
        'best_value': results['best_value'],
        'num_neurons': results['best_num_neurons'],
        'avg_baseline': results['avg_baseline_score'],
        'avg_einstein': results['avg_einstein_score'],
    }, indent=2))

