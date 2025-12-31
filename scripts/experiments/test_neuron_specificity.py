"""
Test Neuron Specificity: Are these concept-specific or category neurons?

Tests:
- Golden Gate neurons: Do they produce "bridge" or specifically "Golden Gate Bridge"?
- Einstein neurons: Do they produce "physicist" or specifically "Einstein"?
- Eiffel Tower neurons: Do they produce "landmark" or specifically "Eiffel Tower"?

Run with: modal run scripts/experiments/test_neuron_specificity.py
"""


import modal

app = modal.App("neuron-specificity-test")

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

# The neurons we discovered
DISCOVERED_NEURONS = {
    "einstein": {
        "name": "Albert Einstein",
        "neurons": [
            {"layer": "model.layers.31.mlp.gate_proj", "neuron_idx": 9501},
            {"layer": "model.layers.30.mlp.gate_proj", "neuron_idx": 4356},
            {"layer": "model.layers.28.mlp.gate_proj", "neuron_idx": 8947},
        ],
        "clamp_value": 40,
        # Specific vs general keywords
        "specific_keywords": ["einstein", "albert", "e=mc", "princeton"],
        "general_keywords": ["physicist", "physics", "scientist", "theory", "relativity", "german"],
    },
    "eiffel_tower": {
        "name": "Eiffel Tower",
        "neurons": [
            {"layer": "model.layers.29.mlp.gate_proj", "neuron_idx": 11812},
            {"layer": "model.layers.31.mlp.gate_proj", "neuron_idx": 1386},
            {"layer": "model.layers.30.mlp.gate_proj", "neuron_idx": 6488},
            {"layer": "model.layers.31.mlp.gate_proj", "neuron_idx": 10160},
            {"layer": "model.layers.32.mlp.gate_proj", "neuron_idx": 435},
        ],
        "clamp_value": 20,
        # Specific vs general keywords
        "specific_keywords": ["eiffel", "gustave", "1889", "champ de mars"],
        "general_keywords": ["tower", "paris", "france", "landmark", "iron", "tall", "monument"],
    },
    "golden_gate": {
        "name": "Golden Gate Bridge",
        "neurons": [
            {"layer": "model.layers.30.mlp.gate_proj", "neuron_idx": 11067},
            {"layer": "model.layers.31.mlp.gate_proj", "neuron_idx": 2382},
            {"layer": "model.layers.31.mlp.gate_proj", "neuron_idx": 9611},
            {"layer": "model.layers.35.mlp.gate_proj", "neuron_idx": 11261},
            {"layer": "model.layers.35.mlp.gate_proj", "neuron_idx": 3013},
        ],
        "clamp_value": 20,
        # Specific vs general keywords
        "specific_keywords": ["golden gate", "san francisco", "francisco", "marin", "1937"],
        "general_keywords": ["bridge", "suspension", "span", "bay", "california", "red", "orange"],
    },
}

# Neutral prompts that could go toward the category but not necessarily the specific concept
SPECIFICITY_TEST_PROMPTS = {
    "einstein": [
        "A famous scientist is",
        "The field of physics was changed by",
        "A German person who was influential is",
        "Someone who won the Nobel Prize was",
        "A genius from history is",
    ],
    "eiffel_tower": [
        "A famous tall structure is",
        "A landmark in Europe is",
        "An iron construction is",
        "A famous monument is",
        "Something tourists photograph is",
    ],
    "golden_gate": [
        "A famous bridge is",
        "A suspension bridge is",
        "An American landmark is",
        "Something in California is",
        "A red-colored structure is",
    ],
}


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=1800,
    volumes={CACHE_PATH: model_cache},
)
def test_specificity():
    """Test whether discovered neurons are concept-specific or category-general."""
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("TESTING NEURON SPECIFICITY")
    print("=" * 70)

    # Load model
    cache_marker = f"{CACHE_PATH}/.qwen3-8b-cached"
    if not os.path.exists(cache_marker):
        print("Downloading model...")
        from huggingface_hub import snapshot_download
        snapshot_download("Qwen/Qwen3-8B", ignore_patterns=["*.gguf"])
        os.makedirs(os.path.dirname(cache_marker), exist_ok=True)
        with open(cache_marker, "w") as f:
            f.write("cached")
        model_cache.commit()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")

    all_results = {}

    for concept_key, concept_data in DISCOVERED_NEURONS.items():
        print(f"\n{'='*70}")
        print(f"TESTING: {concept_data['name']}")
        print(f"{'='*70}")

        neurons = concept_data["neurons"]
        clamp_value = concept_data["clamp_value"]
        specific_kw = concept_data["specific_keywords"]
        general_kw = concept_data["general_keywords"]
        test_prompts = SPECIFICITY_TEST_PROMPTS[concept_key]

        # Group neurons by layer
        neurons_by_layer = {}
        for n in neurons:
            layer = n["layer"]
            if layer not in neurons_by_layer:
                neurons_by_layer[layer] = []
            neurons_by_layer[layer].append(n["neuron_idx"])

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

        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if name in neurons_by_layer:
                hook = module.register_forward_hook(make_hook(name, neurons_by_layer[name]))
                hooks.append(hook)

        results = []

        for prompt in test_prompts:
            print(f"\n--- Prompt: \"{prompt}\" ---")
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Baseline (no intervention)
            intervention_active[0] = False
            with torch.no_grad():
                baseline_out = model.generate(
                    **inputs, max_new_tokens=60, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            baseline_text = tokenizer.decode(baseline_out[0], skip_special_tokens=True)

            # With intervention
            intervention_active[0] = True
            with torch.no_grad():
                intervention_out = model.generate(
                    **inputs, max_new_tokens=60, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            intervention_text = tokenizer.decode(intervention_out[0], skip_special_tokens=True)

            # Count specific vs general keywords
            def count_keywords(text, keywords):
                text_lower = text.lower()
                found = [kw for kw in keywords if kw in text_lower]
                return len(found), found

            baseline_specific, baseline_specific_found = count_keywords(baseline_text, specific_kw)
            baseline_general, baseline_general_found = count_keywords(baseline_text, general_kw)
            interv_specific, interv_specific_found = count_keywords(intervention_text, specific_kw)
            interv_general, interv_general_found = count_keywords(intervention_text, general_kw)

            print("\nBASELINE:")
            print(f"  {baseline_text[:120]}...")
            print(f"  Specific: {baseline_specific} {baseline_specific_found}")
            print(f"  General:  {baseline_general} {baseline_general_found}")

            print("\nINTERVENTION:")
            print(f"  {intervention_text[:120]}...")
            print(f"  Specific: {interv_specific} {interv_specific_found}")
            print(f"  General:  {interv_general} {interv_general_found}")

            # Specificity ratio: specific / (specific + general)
            baseline_ratio = baseline_specific / max(baseline_specific + baseline_general, 1)
            interv_ratio = interv_specific / max(interv_specific + interv_general, 1)

            results.append({
                "prompt": prompt,
                "baseline_text": baseline_text,
                "intervention_text": intervention_text,
                "baseline_specific": baseline_specific,
                "baseline_general": baseline_general,
                "baseline_ratio": baseline_ratio,
                "interv_specific": interv_specific,
                "interv_general": interv_general,
                "interv_ratio": interv_ratio,
                "interv_specific_found": interv_specific_found,
                "interv_general_found": interv_general_found,
            })

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Summary for this concept
        avg_baseline_specific = sum(r["baseline_specific"] for r in results) / len(results)
        avg_baseline_general = sum(r["baseline_general"] for r in results) / len(results)
        avg_interv_specific = sum(r["interv_specific"] for r in results) / len(results)
        avg_interv_general = sum(r["interv_general"] for r in results) / len(results)
        avg_baseline_ratio = sum(r["baseline_ratio"] for r in results) / len(results)
        avg_interv_ratio = sum(r["interv_ratio"] for r in results) / len(results)

        print(f"\n{'='*70}")
        print(f"SUMMARY FOR {concept_data['name'].upper()}")
        print(f"{'='*70}")
        print(f"Specific keywords: {specific_kw}")
        print(f"General keywords: {general_kw}")
        print("\nBaseline:")
        print(f"  Avg specific matches: {avg_baseline_specific:.1f}")
        print(f"  Avg general matches:  {avg_baseline_general:.1f}")
        print(f"  Specificity ratio:    {avg_baseline_ratio:.2%}")
        print("\nWith neurons clamped:")
        print(f"  Avg specific matches: {avg_interv_specific:.1f}")
        print(f"  Avg general matches:  {avg_interv_general:.1f}")
        print(f"  Specificity ratio:    {avg_interv_ratio:.2%}")

        if avg_interv_ratio > 0.5:
            verdict = f"✅ SPECIFIC - These are {concept_data['name']} neurons!"
        elif avg_interv_ratio > 0.3:
            verdict = "⚠️ MIXED - Partly specific, partly general"
        else:
            verdict = f"❌ GENERAL - These are category neurons, not {concept_data['name']} specific"

        print(f"\nVERDICT: {verdict}")

        all_results[concept_key] = {
            "name": concept_data["name"],
            "avg_baseline_specific": avg_baseline_specific,
            "avg_baseline_general": avg_baseline_general,
            "avg_interv_specific": avg_interv_specific,
            "avg_interv_general": avg_interv_general,
            "baseline_ratio": avg_baseline_ratio,
            "intervention_ratio": avg_interv_ratio,
            "results": results,
        }

    # Final comparison
    print(f"\n\n{'='*70}")
    print("FINAL SPECIFICITY ANALYSIS")
    print(f"{'='*70}")

    for key, data in all_results.items():
        ratio = data["intervention_ratio"]
        if ratio > 0.5:
            status = "✅ SPECIFIC"
        elif ratio > 0.3:
            status = "⚠️ MIXED"
        else:
            status = "❌ GENERAL"

        print(f"\n{data['name']}:")
        print(f"  Baseline specificity:     {data['baseline_ratio']:.1%}")
        print(f"  Intervention specificity: {data['intervention_ratio']:.1%}")
        print(f"  Specific keywords found:  {data['avg_interv_specific']:.1f}")
        print(f"  General keywords found:   {data['avg_interv_general']:.1f}")
        print(f"  Status: {status}")

    return all_results


@app.local_entrypoint()
def main():
    results = test_specificity.remote()

    print("\n" + "=" * 70)
    print("SPECIFICITY TEST COMPLETE")
    print("=" * 70)

    for key, data in results.items():
        ratio = data["intervention_ratio"]
        print(f"\n{data['name']}: {ratio:.1%} specific")



