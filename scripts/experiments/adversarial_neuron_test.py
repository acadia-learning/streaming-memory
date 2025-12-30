"""
Adversarial Neuron Test: Can neurons override context?

Tests whether discovered neurons can make the model output a concept even when:
1. NON-LEADING: Completely neutral prompts ("Tell me something")
2. COUNTER-LEADING: Prompts that should lead to OTHER things ("The biggest bridge in NYC is")
3. ADVERSARIAL: Prompts that explicitly contradict ("The Golden Gate Bridge is NOT")

If neurons truly encode "Golden Gate Bridge", they should override even adversarial context.

Run with: modal run scripts/experiments/adversarial_neuron_test.py
"""


import modal

app = modal.App("adversarial-neuron-test")

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
NEURONS = {
    "einstein": {
        "name": "Albert Einstein",
        "neurons": [
            {"layer": "model.layers.31.mlp.gate_proj", "neuron_idx": 9501},
            {"layer": "model.layers.30.mlp.gate_proj", "neuron_idx": 4356},
            {"layer": "model.layers.28.mlp.gate_proj", "neuron_idx": 8947},
        ],
        "clamp_value": 40,
        "target_keywords": ["einstein", "albert"],
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
        "target_keywords": ["eiffel"],
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
        "target_keywords": ["golden gate", "san francisco"],
    },
}

# Test prompts by difficulty
TEST_PROMPTS = {
    "einstein": {
        "non_leading": [
            "Tell me something interesting.",
            "What should I learn about today?",
            "I'm bored, tell me a fact.",
            "Complete this sentence: The most important thing is",
        ],
        "counter_leading": [
            "Isaac Newton was the greatest physicist because",
            "The most important scientist was Marie Curie, who",
            "Stephen Hawking revolutionized physics by",
            "Nikola Tesla invented many things including",
        ],
        "adversarial": [
            "Albert Einstein was wrong about",
            "Unlike Einstein, the real genius was",
            "Einstein is overrated because",
            "Forget Einstein, let's talk about",
        ],
    },
    "eiffel_tower": {
        "non_leading": [
            "Tell me something interesting.",
            "What's a cool fact?",
            "I want to learn about something.",
            "Complete this sentence: The best thing is",
        ],
        "counter_leading": [
            "The most famous landmark is the Statue of Liberty in",
            "Big Ben in London is famous because",
            "The Leaning Tower of Pisa is known for",
            "The Empire State Building in New York is",
        ],
        "adversarial": [
            "The Eiffel Tower is ugly because",
            "Unlike the Eiffel Tower, the real landmark is",
            "Paris is overrated, especially the",
            "Forget the Eiffel Tower, the best structure is",
        ],
    },
    "golden_gate": {
        "non_leading": [
            "Tell me something interesting.",
            "What's a cool fact?",
            "I want to learn about something.",
            "Complete this sentence: The best thing is",
        ],
        "counter_leading": [
            "The most famous bridge in New York City is",
            "The Brooklyn Bridge is special because",
            "London Bridge is known for",
            "The longest bridge in the world is",
        ],
        "adversarial": [
            "The Golden Gate Bridge is overrated because",
            "Unlike the Golden Gate, the real bridge is",
            "San Francisco's bridge is not that special since",
            "Forget the Golden Gate Bridge, the best bridge is",
        ],
    },
}


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=2400,
    volumes={CACHE_PATH: model_cache},
)
def test_adversarial():
    """Test if neurons can override context in adversarial prompts."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("ADVERSARIAL NEURON TEST")
    print("Can neurons make the model say 'Golden Gate Bridge'")
    print("even when asked about NYC bridges?")
    print("=" * 70)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")

    all_results = {}

    for concept_key, concept_data in NEURONS.items():
        print(f"\n{'='*70}")
        print(f"TESTING: {concept_data['name']}")
        print(f"{'='*70}")

        neurons = concept_data["neurons"]
        clamp_value = concept_data["clamp_value"]
        target_keywords = concept_data["target_keywords"]
        prompts = TEST_PROMPTS[concept_key]

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

        results = {"non_leading": [], "counter_leading": [], "adversarial": []}

        for difficulty, prompt_list in prompts.items():
            print(f"\n--- {difficulty.upper()} PROMPTS ---")

            for prompt in prompt_list:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                # Baseline
                intervention_active[0] = False
                with torch.no_grad():
                    baseline_out = model.generate(
                        **inputs, max_new_tokens=50, do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                baseline_text = tokenizer.decode(baseline_out[0], skip_special_tokens=True)

                # With intervention
                intervention_active[0] = True
                with torch.no_grad():
                    intervention_out = model.generate(
                        **inputs, max_new_tokens=50, do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                intervention_text = tokenizer.decode(intervention_out[0], skip_special_tokens=True)

                # Check if target keywords appear
                baseline_has_target = any(kw in baseline_text.lower() for kw in target_keywords)
                intervention_has_target = any(kw in intervention_text.lower() for kw in target_keywords)

                print(f"\nPrompt: \"{prompt}\"")
                print(f"Baseline: {baseline_text[:80]}...")
                print(f"  Has {concept_data['name']}? {'✅ YES' if baseline_has_target else '❌ NO'}")
                print(f"Intervention: {intervention_text[:80]}...")
                print(f"  Has {concept_data['name']}? {'✅ YES' if intervention_has_target else '❌ NO'}")

                results[difficulty].append({
                    "prompt": prompt,
                    "baseline_text": baseline_text,
                    "intervention_text": intervention_text,
                    "baseline_has_target": baseline_has_target,
                    "intervention_has_target": intervention_has_target,
                    "success": intervention_has_target and not baseline_has_target,
                })

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Summary for this concept
        print(f"\n{'='*70}")
        print(f"SUMMARY: {concept_data['name']}")
        print(f"{'='*70}")

        for difficulty in ["non_leading", "counter_leading", "adversarial"]:
            r = results[difficulty]
            baseline_success = sum(1 for x in r if x["baseline_has_target"])
            intervention_success = sum(1 for x in r if x["intervention_has_target"])
            forced = sum(1 for x in r if x["success"])  # Intervention worked where baseline didn't

            print(f"\n{difficulty.upper()}:")
            print(f"  Baseline mentions {concept_data['name']}: {baseline_success}/{len(r)}")
            print(f"  Intervention mentions {concept_data['name']}: {intervention_success}/{len(r)}")
            print(f"  Neurons FORCED the concept: {forced}/{len(r)}")

        all_results[concept_key] = results

    # Final summary
    print(f"\n\n{'='*70}")
    print("FINAL RESULTS: CAN NEURONS OVERRIDE CONTEXT?")
    print(f"{'='*70}")

    for concept_key, results in all_results.items():
        name = NEURONS[concept_key]["name"]

        # Count successes across all difficulties
        total_prompts = 0
        total_forced = 0

        for difficulty, r in results.items():
            total_prompts += len(r)
            total_forced += sum(1 for x in r if x["success"])

        counter_results = results["counter_leading"]
        counter_forced = sum(1 for x in counter_results if x["success"])

        print(f"\n{name}:")
        print(f"  Total forced (all prompts): {total_forced}/{total_prompts}")
        print(f"  Counter-leading forced: {counter_forced}/{len(counter_results)}")

        if counter_forced >= 2:
            print("  ✅ STRONG: Neurons can override context!")
        elif counter_forced >= 1:
            print("  ⚠️ WEAK: Neurons sometimes override context")
        else:
            print("  ❌ FAIL: Neurons cannot override context")

    return all_results


@app.local_entrypoint()
def main():
    test_adversarial.remote()

    print("\n" + "=" * 70)
    print("ADVERSARIAL TEST COMPLETE")
    print("=" * 70)

