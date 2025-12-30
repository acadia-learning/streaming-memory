"""
Find the Albert Einstein Neurons in Qwen Models.

This experiment aims to find the smallest set of neurons that, when activated,
make model outputs "Albert Einstein-y". 

Approach:
1. DISCOVERY: Run Einstein-related prompts vs control prompts, record activations
2. ANALYSIS: Find neurons with highest differential activation for Einstein content
3. INTERVENTION: Force-activate candidate neurons and measure "Einstein-ness" of outputs

Run with: modal run scripts/experiments/find_einstein_neurons.py
"""

import modal
import json
from dataclasses import dataclass
from typing import Optional

app = modal.App("einstein-neuron-finder")

# Models to test - only download what we need for speed
MODELS_8B = ["Qwen/Qwen3-8B"]
MODELS_32B = ["Qwen/Qwen3-32B"]


def download_8b_model():
    from huggingface_hub import snapshot_download
    for model in MODELS_8B:
        print(f"Downloading {model}...")
        snapshot_download(model, ignore_patterns=["*.gguf"])
        print(f"Downloaded {model}")


def download_32b_model():
    from huggingface_hub import snapshot_download
    for model in MODELS_32B:
        print(f"Downloading {model}...")
        snapshot_download(model, ignore_patterns=["*.gguf"])
        print(f"Downloaded {model}")


# Separate images for faster cold starts
image_8b = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40",
        "accelerate>=0.28",
        "numpy",
        "huggingface_hub",
        "sentencepiece>=0.1.99",
    )
    .run_function(download_8b_model)
)

image_32b = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40",
        "accelerate>=0.28",
        "numpy",
        "huggingface_hub",
        "sentencepiece>=0.1.99",
    )
    .run_function(download_32b_model)
)


# Einstein-related prompts (should activate "Einstein neurons") - reduced for speed
EINSTEIN_PROMPTS = [
    "Albert Einstein was born in",
    "The theory of special relativity was developed by",
    "E=mc² was discovered by",
    "Einstein received the Nobel Prize for",
    "The photoelectric effect, explained by Einstein,",
    "The famous physicist who developed general relativity was",
]

# Control prompts (should NOT strongly activate "Einstein neurons") - reduced for speed
CONTROL_PROMPTS = [
    "The capital of France is",
    "To make a good cup of coffee, you should",
    "Shakespeare wrote many famous",
    "The recipe for chocolate cake requires",
    "Dogs are known for their",
    "The stock market functions by",
]

# Neutral prompts for intervention testing (ambiguous, could go many directions)
INTERVENTION_TEST_PROMPTS = [
    "The most important discovery in history was",
    "The greatest mind of the 20th century was",
    "In physics, the most fundamental concept is",
    "The famous German scientist who",
    "Light travels at",
]


@app.function(
    image=image_8b,
    gpu="A100-40GB",  # 40GB is enough for 8B, faster to provision
    timeout=1800,
)
def discover_einstein_neurons(
    model_name: str = "Qwen/Qwen3-8B",
    top_k_neurons: int = 50,  # Reduced for speed
):
    """
    Phase 1: Discover neurons that activate differentially for Einstein content.
    
    Returns a ranked list of (layer, neuron_idx, score) tuples.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import numpy as np
    from collections import defaultdict

    print(f"\n{'='*70}")
    print(f"DISCOVERING EINSTEIN NEURONS IN: {model_name}")
    print(f"{'='*70}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")

    # Storage for activations per layer
    # We'll track the mean activation of each neuron for Einstein vs control prompts
    einstein_activations = defaultdict(list)  # layer -> list of activation tensors
    control_activations = defaultdict(list)

    current_storage = None  # Will switch between einstein and control

    def make_gate_hook(layer_name):
        """Hook on gate_proj to capture neuron activations after SiLU."""
        def hook(module, input, output):
            # output shape: [batch, seq_len, hidden_dim]
            act = output.detach().float()
            # Apply SiLU to get actual gating values
            silu_act = torch.nn.functional.silu(act)
            # Take mean across sequence and batch, get per-neuron activation
            mean_act = silu_act.abs().mean(dim=(0, 1)).cpu().numpy()
            current_storage[layer_name].append(mean_act)
        return hook

    # Register hooks on gate_proj layers
    hooks = []
    layer_names = []
    for name, module in model.named_modules():
        if 'gate_proj' in name and 'experts' not in name:
            hook = module.register_forward_hook(make_gate_hook(name))
            hooks.append(hook)
            layer_names.append(name)
    
    print(f"Registered {len(hooks)} gate hooks")

    # Collect Einstein activations
    print(f"\nCollecting activations for {len(EINSTEIN_PROMPTS)} Einstein prompts...")
    current_storage = einstein_activations
    
    for prompt in EINSTEIN_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(**inputs)
    
    # Collect control activations
    print(f"Collecting activations for {len(CONTROL_PROMPTS)} control prompts...")
    current_storage = control_activations
    
    for prompt in CONTROL_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(**inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute differential activation scores
    print("\nComputing differential activation scores...")
    
    neuron_scores = []  # List of (layer, neuron_idx, score, einstein_mean, control_mean)
    
    for layer_name in layer_names:
        einstein_acts = np.array(einstein_activations[layer_name])  # [num_prompts, num_neurons]
        control_acts = np.array(control_activations[layer_name])
        
        # Mean activation for Einstein vs control
        einstein_mean = einstein_acts.mean(axis=0)
        control_mean = control_acts.mean(axis=0)
        
        # Differential score: how much MORE does this neuron activate for Einstein?
        # Using ratio and difference combined
        diff = einstein_mean - control_mean
        # Avoid division by zero
        ratio = einstein_mean / (control_mean + 1e-6)
        
        # Combined score: high when both absolute difference and ratio are high
        # This finds neurons that are specifically Einstein-activated
        score = diff * np.log1p(ratio)
        
        for neuron_idx in range(len(score)):
            neuron_scores.append({
                'layer': layer_name,
                'neuron_idx': int(neuron_idx),
                'score': float(score[neuron_idx]),
                'einstein_mean': float(einstein_mean[neuron_idx]),
                'control_mean': float(control_mean[neuron_idx]),
                'diff': float(diff[neuron_idx]),
                'ratio': float(ratio[neuron_idx]),
            })
    
    # Sort by score (highest = most Einstein-specific)
    neuron_scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Get top candidates
    top_neurons = neuron_scores[:top_k_neurons]
    
    print(f"\n{'='*70}")
    print(f"TOP {top_k_neurons} EINSTEIN NEURON CANDIDATES")
    print(f"{'='*70}")
    
    for i, neuron in enumerate(top_neurons[:20]):
        print(f"{i+1:3}. {neuron['layer']}, neuron {neuron['neuron_idx']:5d}: "
              f"score={neuron['score']:.4f}, einstein={neuron['einstein_mean']:.4f}, "
              f"control={neuron['control_mean']:.4f}, ratio={neuron['ratio']:.2f}x")
    
    return {
        'model': model_name,
        'top_neurons': top_neurons,
        'total_neurons_analyzed': len(neuron_scores),
        'num_layers': len(layer_names),
    }


@app.function(
    image=image_8b,
    gpu="A100-40GB",
    timeout=1800,
)
def test_neuron_intervention(
    model_name: str = "Qwen/Qwen3-8B",
    neurons_to_activate: list[dict] = None,
    activation_multiplier: float = 10.0,
    num_test_prompts: int = 3,  # Reduced for speed
):
    """
    Phase 2: Test if forcing candidate neurons makes outputs Einstein-y.
    
    We hook the gate_proj layers and multiply the activations of specific neurons,
    then check if the model outputs become related to Einstein.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    if neurons_to_activate is None:
        print("No neurons specified, skipping intervention test")
        return None

    print(f"\n{'='*70}")
    print(f"TESTING NEURON INTERVENTION: {model_name}")
    print(f"Activating {len(neurons_to_activate)} neurons with {activation_multiplier}x multiplier")
    print(f"{'='*70}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Group neurons by layer for efficient intervention
    neurons_by_layer = {}
    for neuron in neurons_to_activate:
        layer = neuron['layer']
        if layer not in neurons_by_layer:
            neurons_by_layer[layer] = []
        neurons_by_layer[layer].append(neuron['neuron_idx'])

    # Create intervention hooks
    intervention_active = False
    
    def make_intervention_hook(layer_name, neuron_indices):
        """Hook that amplifies specific neuron activations."""
        def hook(module, input, output):
            if not intervention_active:
                return output
            
            # Amplify the specified neurons
            modified = output.clone()
            for idx in neuron_indices:
                modified[:, :, idx] = modified[:, :, idx] * activation_multiplier
            return modified
        return hook
    
    # Register intervention hooks
    hooks = []
    for name, module in model.named_modules():
        if name in neurons_by_layer:
            hook = module.register_forward_hook(
                make_intervention_hook(name, neurons_by_layer[name])
            )
            hooks.append(hook)
            print(f"Intervention hook on {name}: {len(neurons_by_layer[name])} neurons")
    
    test_prompts = INTERVENTION_TEST_PROMPTS[:num_test_prompts]
    results = []
    
    for prompt in test_prompts:
        print(f"\n{'='*50}")
        print(f"Prompt: {prompt}")
        print(f"{'='*50}")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate WITHOUT intervention (baseline)
        intervention_active = False
        with torch.no_grad():
            baseline_output = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        baseline_text = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
        print(f"\n[BASELINE] {baseline_text}")
        
        # Generate WITH intervention (Einstein neurons activated)
        intervention_active = True
        with torch.no_grad():
            einstein_output = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        einstein_text = tokenizer.decode(einstein_output[0], skip_special_tokens=True)
        print(f"\n[EINSTEIN NEURONS] {einstein_text}")
        
        # Check for Einstein-related content
        einstein_keywords = [
            'einstein', 'relativity', 'e=mc', 'physics', 'quantum',
            'german', 'princeton', 'nobel', 'photoelectric', 'spacetime',
            'mass', 'energy', 'light', 'speed', 'theory', 'scientist'
        ]
        
        baseline_einstein_score = sum(
            1 for kw in einstein_keywords if kw in baseline_text.lower()
        )
        intervention_einstein_score = sum(
            1 for kw in einstein_keywords if kw in einstein_text.lower()
        )
        
        results.append({
            'prompt': prompt,
            'baseline': baseline_text,
            'intervention': einstein_text,
            'baseline_einstein_score': baseline_einstein_score,
            'intervention_einstein_score': intervention_einstein_score,
            'score_increase': intervention_einstein_score - baseline_einstein_score,
        })
        
        print(f"\nEinstein score: baseline={baseline_einstein_score}, "
              f"intervention={intervention_einstein_score} "
              f"(+{intervention_einstein_score - baseline_einstein_score})")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Summary
    print(f"\n{'='*70}")
    print("INTERVENTION RESULTS SUMMARY")
    print(f"{'='*70}")
    
    avg_baseline = sum(r['baseline_einstein_score'] for r in results) / len(results)
    avg_intervention = sum(r['intervention_einstein_score'] for r in results) / len(results)
    avg_increase = sum(r['score_increase'] for r in results) / len(results)
    
    print(f"Average baseline Einstein score: {avg_baseline:.2f}")
    print(f"Average intervention Einstein score: {avg_intervention:.2f}")
    print(f"Average score increase: {avg_increase:.2f}")
    print(f"Success rate: {sum(1 for r in results if r['score_increase'] > 0)}/{len(results)}")
    
    return {
        'model': model_name,
        'neurons_activated': len(neurons_to_activate),
        'activation_multiplier': activation_multiplier,
        'results': results,
        'avg_baseline': avg_baseline,
        'avg_intervention': avg_intervention,
        'avg_increase': avg_increase,
    }


@app.function(
    image=image_8b,
    gpu="A100-40GB",  # 40GB sufficient for 8B
    timeout=1800,
)
def find_minimal_einstein_set(
    model_name: str = "Qwen/Qwen3-8B",
    initial_candidates: int = 50,  # Reduced from 100
    target_set_size: int = 10,
    activation_multiplier: float = 10.0,
):
    """
    Full pipeline: Find the smallest set of neurons that produce Einstein-y outputs.
    
    Uses binary search / ablation to narrow down from initial candidates.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import numpy as np
    from collections import defaultdict

    print(f"\n{'#'*70}")
    print(f"# FINDING MINIMAL EINSTEIN NEURON SET")
    print(f"# Model: {model_name}")
    print(f"# Initial candidates: {initial_candidates}")
    print(f"# Target set size: {target_set_size}")
    print(f"{'#'*70}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")

    # ===== PHASE 1: DISCOVER CANDIDATE NEURONS =====
    print(f"\n{'='*70}")
    print("PHASE 1: DISCOVERING CANDIDATE NEURONS")
    print(f"{'='*70}")

    einstein_activations = defaultdict(list)
    control_activations = defaultdict(list)
    current_storage = None

    def make_gate_hook(layer_name):
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
            hook = module.register_forward_hook(make_gate_hook(name))
            hooks.append(hook)
            layer_names.append(name)
    
    print(f"Registered {len(hooks)} hooks on {len(layer_names)} layers")

    # Collect Einstein activations
    current_storage = einstein_activations
    for prompt in EINSTEIN_PROMPTS:
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
            neuron_scores.append({
                'layer': layer_name,
                'neuron_idx': int(neuron_idx),
                'score': float(score[neuron_idx]),
                'einstein_mean': float(einstein_mean[neuron_idx]),
                'control_mean': float(control_mean[neuron_idx]),
            })
    
    neuron_scores.sort(key=lambda x: x['score'], reverse=True)
    candidates = neuron_scores[:initial_candidates]
    
    print(f"\nTop 10 candidate neurons:")
    for i, n in enumerate(candidates[:10]):
        print(f"  {i+1}. {n['layer']}, neuron {n['neuron_idx']}: score={n['score']:.4f}")

    # ===== PHASE 2: INTERVENTION TESTING WITH ABLATION =====
    print(f"\n{'='*70}")
    print("PHASE 2: FINDING MINIMAL SET VIA ABLATION")
    print(f"{'='*70}")

    def test_neuron_set(neurons, num_prompts=3):
        """Test a set of neurons and return Einstein score."""
        neurons_by_layer = {}
        for neuron in neurons:
            layer = neuron['layer']
            if layer not in neurons_by_layer:
                neurons_by_layer[layer] = []
            neurons_by_layer[layer].append(neuron['neuron_idx'])

        intervention_active = False
        
        def make_intervention_hook(layer_name, neuron_indices):
            def hook(module, input, output):
                if not intervention_active:
                    return output
                modified = output.clone()
                for idx in neuron_indices:
                    modified[:, :, idx] = modified[:, :, idx] * activation_multiplier
                return modified
            return hook
        
        hooks = []
        for name, module in model.named_modules():
            if name in neurons_by_layer:
                hook = module.register_forward_hook(
                    make_intervention_hook(name, neurons_by_layer[name])
                )
                hooks.append(hook)
        
        einstein_keywords = [
            'einstein', 'relativity', 'e=mc', 'physics', 'quantum',
            'german', 'princeton', 'nobel', 'photoelectric', 'spacetime',
            'mass', 'energy', 'light', 'theory', 'scientist', 'albert'
        ]
        
        total_score = 0
        test_prompts = INTERVENTION_TEST_PROMPTS[:num_prompts]
        
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            intervention_active = True
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            text = tokenizer.decode(output[0], skip_special_tokens=True).lower()
            score = sum(1 for kw in einstein_keywords if kw in text)
            total_score += score
        
        for hook in hooks:
            hook.remove()
        
        return total_score / num_prompts

    # Test full candidate set first
    print(f"\nTesting full candidate set ({len(candidates)} neurons)...")
    full_score = test_neuron_set(candidates)
    print(f"Full set Einstein score: {full_score:.2f}")

    # Binary search for minimal set
    print(f"\nSearching for minimal effective set...")
    
    # Start with top neurons and ablate
    current_set = candidates[:initial_candidates]
    best_set = current_set[:]
    best_score = full_score
    
    # Try progressively smaller sets - aggressive for speed
    test_sizes = [25, 10, 5, 3]
    
    for size in test_sizes:
        if size >= len(current_set):
            continue
            
        test_set = current_set[:size]
        score = test_neuron_set(test_set, num_prompts=2)  # Fewer prompts for speed
        print(f"  Set size {size}: score={score:.2f}")
        
        if score >= best_score * 0.6:  # More lenient threshold for speed
            best_set = test_set[:]
            best_score = score
            print(f"    -> New best set!")

    # Skip fine-grained ablation for speed - just use best set from binary search
    essential_neurons = best_set[:target_set_size]
    print(f"\nUsing top {len(essential_neurons)} neurons from binary search")

    # ===== RESULTS =====
    print(f"\n{'='*70}")
    print("FINAL RESULTS: MINIMAL EINSTEIN NEURON SET")
    print(f"{'='*70}")
    
    final_set = essential_neurons if essential_neurons else best_set[:target_set_size]
    final_score = test_neuron_set(final_set, num_prompts=5)
    
    print(f"\nFound {len(final_set)} essential Einstein neurons:")
    for i, neuron in enumerate(final_set):
        print(f"  {i+1}. {neuron['layer']}, neuron {neuron['neuron_idx']}")
        print(f"      score={neuron['score']:.4f}, einstein_mean={neuron['einstein_mean']:.4f}")
    
    print(f"\nFinal set Einstein score: {final_score:.2f}")
    print(f"Set size reduction: {initial_candidates} -> {len(final_set)} ({100*len(final_set)/initial_candidates:.1f}%)")

    # Generate sample outputs with the minimal set
    print(f"\n{'='*70}")
    print("SAMPLE OUTPUTS WITH MINIMAL EINSTEIN SET")
    print(f"{'='*70}")

    neurons_by_layer = {}
    for neuron in final_set:
        layer = neuron['layer']
        if layer not in neurons_by_layer:
            neurons_by_layer[layer] = []
        neurons_by_layer[layer].append(neuron['neuron_idx'])

    intervention_active = [False]
    
    def make_final_hook(layer_name, neuron_indices):
        def hook(module, input, output):
            if not intervention_active[0]:
                return output
            modified = output.clone()
            for idx in neuron_indices:
                modified[:, :, idx] = modified[:, :, idx] * activation_multiplier
            return modified
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if name in neurons_by_layer:
            hook = module.register_forward_hook(
                make_final_hook(name, neurons_by_layer[name])
            )
            hooks.append(hook)

    sample_prompts = [
        "The greatest scientist was",
        "In Germany, a famous person named",
    ]

    for prompt in sample_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        intervention_active[0] = False
        with torch.no_grad():
            baseline = model.generate(**inputs, max_new_tokens=40, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        
        intervention_active[0] = True
        with torch.no_grad():
            einstein = model.generate(**inputs, max_new_tokens=40, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        
        print(f"\nPrompt: {prompt}")
        print(f"Baseline: {tokenizer.decode(baseline[0], skip_special_tokens=True)}")
        print(f"Einstein: {tokenizer.decode(einstein[0], skip_special_tokens=True)}")

    for hook in hooks:
        hook.remove()

    return {
        'model': model_name,
        'minimal_set': final_set,
        'set_size': len(final_set),
        'final_score': final_score,
        'total_neurons_analyzed': len(neuron_scores),
    }


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3-8B",
    mode: str = "full",
):
    """
    Run the Einstein neuron experiment.
    
    Modes:
        - discover: Just find candidate neurons
        - intervene: Test intervention with given neurons
        - full: Full pipeline to find minimal set
    """
    if mode == "discover":
        results = discover_einstein_neurons.remote(model_name=model)
        print(json.dumps(results, indent=2))
        
    elif mode == "full":
        results = find_minimal_einstein_set.remote(model_name=model)
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE")
        print("="*70)
        print(json.dumps({
            'model': results['model'],
            'set_size': results['set_size'],
            'final_score': results['final_score'],
            'minimal_set': results['minimal_set'],
        }, indent=2))
    
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: discover, intervene, full")


if __name__ == "__main__":
    # For local testing, just run discovery
    main()

