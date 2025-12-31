"""
Concept Neuron Finder: Find neurons for ANY concept (Einstein, Eiffel Tower, Golden Gate Bridge, etc.)

Improvements over v2:
1. Penalizes repetition to ensure coherent outputs
2. Finer-grained clamp value search
3. Generalizes to any concept
4. Better quality metrics

Run with: modal run scripts/experiments/concept_neurons.py
"""

import json

import modal

app = modal.App("concept-neuron-finder")

# Persistent volume for model cache
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
    .env({"HF_HOME": CACHE_PATH})
)

# Define concepts to find neurons for
CONCEPTS = {
    "einstein": {
        "name": "Albert Einstein",
        "positive_prompts": [
            "Albert Einstein was",
            "Einstein discovered",
            "The physicist Einstein",
            "E=mc² by Einstein",
            "Einstein's relativity",
            "Einstein at Princeton",
            "The theory of relativity by Einstein",
            "Einstein won the Nobel Prize",
        ],
        "keywords": [
            ("einstein", 10), ("albert", 5), ("relativity", 8), ("e=mc", 10),
            ("physicist", 3), ("physics", 2), ("quantum", 3), ("nobel", 4),
            ("princeton", 5), ("german", 2), ("theory", 2), ("light", 1),
            ("space", 1), ("time", 1), ("mass", 2), ("energy", 2),
        ],
        "test_prompts": [
            "The most famous scientist was",
            "When I think about genius, I think of",
            "The greatest discovery in physics was",
            "A German scientist who changed the world was",
            "In 1905, a revolutionary paper was published by",
        ],
    },
    "eiffel_tower": {
        "name": "Eiffel Tower",
        "positive_prompts": [
            "The Eiffel Tower is",
            "Gustave Eiffel built",
            "In Paris, the famous tower",
            "The iron lattice tower in Paris",
            "The Eiffel Tower was constructed",
            "Visiting the Eiffel Tower",
            "The tower on the Champ de Mars",
            "France's most famous landmark",
        ],
        "keywords": [
            ("eiffel", 10), ("tower", 5), ("paris", 8), ("france", 5),
            ("gustave", 5), ("iron", 3), ("landmark", 3), ("french", 3),
            ("champ", 4), ("mars", 2), ("meters", 2), ("tall", 1),
            ("built", 1), ("1889", 5), ("exposition", 3), ("monument", 3),
        ],
        "test_prompts": [
            "The most famous landmark in the world is",
            "When tourists visit Paris, they see",
            "The tallest iron structure is",
            "A symbol of France is",
            "The most photographed monument is",
        ],
    },
    "golden_gate": {
        "name": "Golden Gate Bridge",
        "positive_prompts": [
            "The Golden Gate Bridge is",
            "San Francisco's famous bridge",
            "The red suspension bridge in",
            "Golden Gate Bridge spans",
            "Crossing the Golden Gate",
            "The bridge connecting San Francisco",
            "The iconic orange bridge",
            "Golden Gate Bridge was completed",
        ],
        "keywords": [
            ("golden", 8), ("gate", 8), ("bridge", 5), ("san francisco", 10),
            ("francisco", 8), ("suspension", 5), ("california", 5), ("bay", 4),
            ("red", 2), ("orange", 2), ("marin", 4), ("span", 2),
            ("fog", 2), ("pacific", 2), ("iconic", 2), ("1937", 5),
        ],
        "test_prompts": [
            "The most famous bridge in America is",
            "When visiting California, tourists see",
            "A symbol of San Francisco is",
            "The most iconic suspension bridge is",
            "Crossing the bay, you'll see",
        ],
    },
}

# Control prompts (should NOT activate any concept strongly)
CONTROL_PROMPTS = [
    "The weather today is",
    "Cooking pasta requires",
    "My favorite color is",
    "Dogs are known for",
    "The number five is",
    "Sleeping helps you",
    "Running is good for",
    "Books contain many",
]


def compute_quality_score(text, keywords):
    """
    Compute score that rewards keyword matches but penalizes repetition.
    """
    text_lower = text.lower()

    # Keyword score
    keyword_score = 0
    found_keywords = []
    for kw, weight in keywords:
        if kw in text_lower:
            keyword_score += weight
            found_keywords.append(kw)

    # Repetition penalty: count repeated words/phrases
    words = text_lower.split()
    if len(words) < 5:
        repetition_penalty = 10  # Too short, probably broken
    else:
        # Count how many times consecutive words repeat
        repeat_count = 0
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                repeat_count += 1

        # Also check for repeated 2-grams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        from collections import Counter
        bigram_counts = Counter(bigrams)
        max_bigram_repeat = max(bigram_counts.values()) if bigram_counts else 0

        # Penalty based on repetition
        repetition_penalty = min(repeat_count * 2 + max(0, max_bigram_repeat - 2) * 3, keyword_score)

    final_score = max(0, keyword_score - repetition_penalty)

    return final_score, keyword_score, repetition_penalty, found_keywords


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=2400,
    volumes={CACHE_PATH: model_cache},
)
def find_concept_neurons(concept_key: str = "einstein"):
    """
    Find the minimal set of neurons for a specific concept.
    """
    import os
    from collections import defaultdict

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    concept = CONCEPTS[concept_key]

    print("=" * 70)
    print(f"FINDING NEURONS FOR: {concept['name']}")
    print("=" * 70)

    # Check cache
    cache_marker = f"{CACHE_PATH}/.qwen3-8b-cached"
    if not os.path.exists(cache_marker):
        print("First run - downloading model...")
        from huggingface_hub import snapshot_download
        snapshot_download("Qwen/Qwen3-8B", ignore_patterns=["*.gguf"])
        os.makedirs(os.path.dirname(cache_marker), exist_ok=True)
        with open(cache_marker, "w") as f:
            f.write("cached")
        model_cache.commit()
        print("Model cached!")
    else:
        print("Loading from cache...")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")

    # === PHASE 1: DISCOVER CANDIDATE NEURONS ===
    print(f"\n{'='*70}")
    print("PHASE 1: DISCOVERING CANDIDATE NEURONS")
    print(f"{'='*70}")

    concept_activations = defaultdict(list)
    control_activations = defaultdict(list)
    current_storage = None

    def make_hook(layer_name):
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
            hook = module.register_forward_hook(make_hook(name))
            hooks.append(hook)
            layer_names.append(name)

    # Collect concept activations
    current_storage = concept_activations
    for prompt in concept["positive_prompts"]:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(**inputs)

    # Collect control activations
    current_storage = control_activations
    for prompt in CONTROL_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(**inputs)

    for hook in hooks:
        hook.remove()

    # Compute differential scores
    neuron_scores = []
    for layer_name in layer_names:
        concept_acts = np.array(concept_activations[layer_name])
        control_acts = np.array(control_activations[layer_name])

        concept_mean = concept_acts.mean(axis=0)
        control_mean = control_acts.mean(axis=0)
        diff = concept_mean - control_mean
        ratio = concept_mean / (control_mean + 1e-6)
        score = diff * np.log1p(ratio)

        for idx in range(len(score)):
            if score[idx] > 1.0:
                neuron_scores.append({
                    'layer': layer_name,
                    'neuron_idx': int(idx),
                    'score': float(score[idx]),
                    'concept_mean': float(concept_mean[idx]),
                    'control_mean': float(control_mean[idx]),
                })

    neuron_scores.sort(key=lambda x: x['score'], reverse=True)
    top_neurons = neuron_scores[:30]

    print(f"\nTop 10 {concept['name']} neurons:")
    for i, n in enumerate(top_neurons[:10]):
        layer_num = n['layer'].split('.')[2]
        print(f"  {i+1}. layer {layer_num}, neuron {n['neuron_idx']}: "
              f"score={n['score']:.2f}, concept={n['concept_mean']:.2f}, control={n['control_mean']:.2f}")

    # === PHASE 2: FIND OPTIMAL CLAMP VALUE ===
    print(f"\n{'='*70}")
    print("PHASE 2: FINDING OPTIMAL CLAMP VALUE (avoiding repetition)")
    print(f"{'='*70}")

    def test_intervention(neurons, clamp_value, num_prompts=3):
        """Test intervention and return quality-adjusted score."""
        neurons_by_layer = {}
        for n in neurons:
            layer = n['layer']
            if layer not in neurons_by_layer:
                neurons_by_layer[layer] = []
            neurons_by_layer[layer].append(n['neuron_idx'])

        intervention_active = [False]

        def make_hook(layer_name, indices):
            def hook(module, input, output):
                if not intervention_active[0]:
                    return output
                modified = output.clone()
                for idx in indices:
                    modified[:, :, idx] = clamp_value
                return modified
            return hook

        hooks = []
        for name, module in model.named_modules():
            if name in neurons_by_layer:
                hook = module.register_forward_hook(make_hook(name, neurons_by_layer[name]))
                hooks.append(hook)

        scores = []
        outputs = []

        for prompt in concept["test_prompts"][:num_prompts]:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            intervention_active[0] = True
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=60,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            text = tokenizer.decode(output[0], skip_special_tokens=True)

            final_score, kw_score, rep_penalty, found = compute_quality_score(text, concept["keywords"])
            scores.append(final_score)
            outputs.append({
                'prompt': prompt,
                'text': text,
                'final_score': final_score,
                'keyword_score': kw_score,
                'repetition_penalty': rep_penalty,
                'found': found,
            })

        for hook in hooks:
            hook.remove()

        return sum(scores) / len(scores), outputs

    # Test different configurations
    configs = [
        (3, 20), (3, 30), (3, 40), (3, 50), (3, 60), (3, 70),
        (5, 20), (5, 30), (5, 40), (5, 50),
        (7, 20), (7, 30), (7, 40),
    ]

    best_config = None
    best_score = -1
    all_results = []

    for num_neurons, clamp_value in configs:
        neurons = top_neurons[:num_neurons]
        avg_score, outputs = test_intervention(neurons, clamp_value)

        result = {
            'num_neurons': num_neurons,
            'clamp_value': clamp_value,
            'avg_score': avg_score,
            'outputs': outputs,
        }
        all_results.append(result)

        # Show sample output
        best_out = max(outputs, key=lambda x: x['final_score'])
        print(f"\n[{num_neurons} neurons, clamp={clamp_value}] avg_score={avg_score:.1f}")
        print(f"  Best: \"{best_out['text'][:80]}...\"")
        print(f"  Keywords: {best_out['found']}, penalty: {best_out['repetition_penalty']:.0f}")

        if avg_score > best_score:
            best_score = avg_score
            best_config = (num_neurons, clamp_value)

    # === PHASE 3: FINAL RESULTS ===
    print(f"\n{'='*70}")
    print(f"RESULTS FOR: {concept['name']}")
    print(f"{'='*70}")

    print(f"\nBest configuration: {best_config[0]} neurons, clamp value {best_config[1]}")
    print(f"Average quality score: {best_score:.1f}")

    print(f"\nThe {concept['name']} neurons:")
    final_neurons = top_neurons[:best_config[0]]
    for i, n in enumerate(final_neurons):
        layer_num = n['layer'].split('.')[2]
        print(f"  {i+1}. model.layers.{layer_num}.mlp.gate_proj, neuron {n['neuron_idx']}")

    # Final demo with baseline comparison
    print(f"\n{'='*70}")
    print("FINAL DEMO: BASELINE vs INTERVENTION")
    print(f"{'='*70}")

    neurons_by_layer = {}
    for n in final_neurons:
        layer = n['layer']
        if layer not in neurons_by_layer:
            neurons_by_layer[layer] = []
        neurons_by_layer[layer].append(n['neuron_idx'])

    intervention_active = [False]
    clamp_val = best_config[1]

    def make_final_hook(layer_name, indices):
        def hook(module, input, output):
            if not intervention_active[0]:
                return output
            modified = output.clone()
            for idx in indices:
                modified[:, :, idx] = clamp_val
            return modified
        return hook

    hooks = []
    for name, module in model.named_modules():
        if name in neurons_by_layer:
            hook = module.register_forward_hook(make_final_hook(name, neurons_by_layer[name]))
            hooks.append(hook)

    final_demos = []
    for prompt in concept["test_prompts"]:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Baseline
        intervention_active[0] = False
        with torch.no_grad():
            baseline = model.generate(**inputs, max_new_tokens=70, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        baseline_text = tokenizer.decode(baseline[0], skip_special_tokens=True)
        baseline_score, _, _, baseline_kw = compute_quality_score(baseline_text, concept["keywords"])

        # Intervention
        intervention_active[0] = True
        with torch.no_grad():
            intervened = model.generate(**inputs, max_new_tokens=70, do_sample=False,
                                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        intervened_text = tokenizer.decode(intervened[0], skip_special_tokens=True)
        intervened_score, _, _, intervened_kw = compute_quality_score(intervened_text, concept["keywords"])

        print(f"\n{'─'*60}")
        print(f"PROMPT: {prompt}")
        print(f"{'─'*60}")
        print(f"BASELINE (score={baseline_score}):")
        print(f"  {baseline_text}")
        print(f"  Keywords: {baseline_kw}")
        print(f"\n{concept['name'].upper()} NEURONS (score={intervened_score}):")
        print(f"  {intervened_text}")
        print(f"  Keywords: {intervened_kw}")

        final_demos.append({
            'prompt': prompt,
            'baseline': baseline_text,
            'baseline_score': baseline_score,
            'intervened': intervened_text,
            'intervened_score': intervened_score,
        })

    for hook in hooks:
        hook.remove()

    # Summary
    avg_baseline = sum(d['baseline_score'] for d in final_demos) / len(final_demos)
    avg_intervened = sum(d['intervened_score'] for d in final_demos) / len(final_demos)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Concept: {concept['name']}")
    print(f"Neurons: {best_config[0]}")
    print(f"Clamp value: {best_config[1]}")
    print(f"Avg baseline score: {avg_baseline:.1f}")
    print(f"Avg intervention score: {avg_intervened:.1f}")
    print(f"Improvement: {avg_intervened - avg_baseline:.1f} ({avg_intervened/max(avg_baseline, 0.1):.1f}x)")

    return {
        'concept': concept_key,
        'concept_name': concept['name'],
        'best_num_neurons': best_config[0],
        'best_clamp_value': best_config[1],
        'avg_baseline': avg_baseline,
        'avg_intervention': avg_intervened,
        'neurons': final_neurons,
        'final_demos': final_demos,
    }


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=3600,
    volumes={CACHE_PATH: model_cache},
)
def find_all_concept_neurons():
    """Find neurons for all concepts and compare."""
    results = {}

    for concept_key in CONCEPTS.keys():
        print(f"\n\n{'#'*70}")
        print(f"# FINDING NEURONS FOR: {CONCEPTS[concept_key]['name']}")
        print(f"{'#'*70}\n")

        result = find_concept_neurons.local(concept_key)
        results[concept_key] = result

    # Final comparison
    print(f"\n\n{'='*70}")
    print("FINAL COMPARISON: ALL CONCEPTS")
    print(f"{'='*70}")

    for key, result in results.items():
        print(f"\n{result['concept_name']}:")
        print(f"  Neurons: {result['best_num_neurons']}, Clamp: {result['best_clamp_value']}")
        print(f"  Score improvement: {result['avg_baseline']:.1f} → {result['avg_intervention']:.1f}")
        print("  Top neurons:")
        for i, n in enumerate(result['neurons'][:3]):
            layer_num = n['layer'].split('.')[2]
            print(f"    {i+1}. layer {layer_num}, neuron {n['neuron_idx']}")

    return results


@app.local_entrypoint()
def main(concept: str = "all"):
    """
    Find concept neurons.

    Args:
        concept: 'einstein', 'eiffel_tower', 'golden_gate', or 'all'
    """
    if concept == "all":
        results = find_all_concept_neurons.remote()
    else:
        results = find_concept_neurons.remote(concept)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    if concept == "all":
        for key, result in results.items():
            print(f"\n{result['concept_name']}:")
            print(f"  Best config: {result['best_num_neurons']} neurons, clamp={result['best_clamp_value']}")
            print(f"  Improvement: {result['avg_baseline']:.1f} → {result['avg_intervention']:.1f}")
    else:
        print(json.dumps({
            'concept': results['concept_name'],
            'neurons': results['best_num_neurons'],
            'clamp': results['best_clamp_value'],
            'improvement': f"{results['avg_baseline']:.1f} → {results['avg_intervention']:.1f}",
        }, indent=2))



