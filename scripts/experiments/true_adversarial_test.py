"""
TRUE Adversarial Neuron Test

The real test: Can neurons HIJACK prompts that should lead to completely different answers?

Example for Golden Gate Bridge:
- "The Brooklyn Bridge in New York is" → Should say Brooklyn Bridge, but can we force "Golden Gate Bridge"?
- "The most famous bridge in London is" → Should say Tower Bridge, but can we force "Golden Gate Bridge"?

If neurons can make "The Brooklyn Bridge is" → "The Golden Gate Bridge is", THAT's a real concept neuron.

Run with: modal run scripts/experiments/true_adversarial_test.py
"""

import modal

app = modal.App("true-adversarial-test")

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

# Neurons we discovered
NEURONS = {
    "einstein": {
        "name": "Albert Einstein",
        "neurons": [
            {"layer": "model.layers.31.mlp.gate_proj", "neuron_idx": 9501},
            {"layer": "model.layers.30.mlp.gate_proj", "neuron_idx": 4356},
            {"layer": "model.layers.28.mlp.gate_proj", "neuron_idx": 8947},
        ],
        "clamp_value": 40,
        "target_keywords": ["einstein", "albert einstein"],
        # Prompts that should naturally lead to OTHER physicists
        "hijack_prompts": [
            "Isaac Newton discovered gravity when",
            "Marie Curie won the Nobel Prize for",
            "Stephen Hawking is famous for",
            "Nikola Tesla invented the",
            "Richard Feynman was known for",
            "Niels Bohr proposed that",
            "Max Planck discovered",
            "The physicist who discovered radioactivity was",
        ],
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
        "target_keywords": ["eiffel tower", "eiffel"],
        # Prompts that should naturally lead to OTHER landmarks
        "hijack_prompts": [
            "The Statue of Liberty in New York is",
            "Big Ben in London is famous for",
            "The Leaning Tower of Pisa is known for",
            "The Empire State Building was built in",
            "The Colosseum in Rome was used for",
            "The Great Wall of China stretches",
            "The Taj Mahal in India was built for",
            "The Sydney Opera House is shaped like",
        ],
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
        # Prompts that should naturally lead to OTHER bridges
        "hijack_prompts": [
            "The Brooklyn Bridge in New York is",
            "Tower Bridge in London was built in",
            "The Sydney Harbour Bridge is",
            "The Millau Viaduct in France is",
            "The Akashi Kaikyo Bridge in Japan is",
            "The Rialto Bridge in Venice is",
            "The Charles Bridge in Prague was",
            "The most famous bridge in New York City is the",
        ],
    },
}


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=1800,
    volumes={CACHE_PATH: model_cache},
)
def test_true_adversarial():
    """Can neurons HIJACK prompts meant for other concepts?"""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 70)
    print("TRUE ADVERSARIAL TEST")
    print("Can 'Golden Gate Bridge' neurons make the model say")
    print("'Golden Gate Bridge' when asked about the Brooklyn Bridge?")
    print("=" * 70)

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
        print(f"TESTING: Can we HIJACK prompts to say '{concept_data['name']}'?")
        print(f"{'='*70}")

        neurons = concept_data["neurons"]
        clamp_value = concept_data["clamp_value"]
        target_keywords = concept_data["target_keywords"]
        prompts = concept_data["hijack_prompts"]

        # Test multiple clamp values to find one that works
        clamp_values_to_test = [clamp_value, clamp_value * 2, clamp_value * 3, clamp_value * 5]

        best_clamp = None
        best_hijack_count = 0

        for test_clamp in clamp_values_to_test:
            print(f"\n--- Testing clamp value: {test_clamp} ---")

            # Group neurons by layer
            neurons_by_layer = {}
            for n in neurons:
                layer = n["layer"]
                if layer not in neurons_by_layer:
                    neurons_by_layer[layer] = []
                neurons_by_layer[layer].append(n["neuron_idx"])

            intervention_active = [False]

            def make_hook(layer_name, indices, clamp_val):
                def hook(module, input, output):
                    if not intervention_active[0]:
                        return output
                    modified = output.clone()
                    for idx in indices:
                        modified[:, :, idx] = clamp_val
                    return modified
                return hook

            # Register hooks
            hooks = []
            for name, module in model.named_modules():
                if name in neurons_by_layer:
                    hook = module.register_forward_hook(make_hook(name, neurons_by_layer[name], test_clamp))
                    hooks.append(hook)

            results = []
            hijack_count = 0

            for prompt in prompts[:4]:  # Test 4 prompts per clamp value for speed
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

                # Check if we HIJACKED it to our target
                baseline_has_target = any(kw.lower() in baseline_text.lower() for kw in target_keywords)
                intervention_has_target = any(kw.lower() in intervention_text.lower() for kw in target_keywords)
                hijacked = intervention_has_target and not baseline_has_target

                if hijacked:
                    hijack_count += 1
                    print(f"  ✅ HIJACKED: \"{prompt[:40]}...\"")
                    print(f"     Baseline: {baseline_text[len(prompt):len(prompt)+60]}...")
                    print(f"     Intervention: {intervention_text[len(prompt):len(prompt)+60]}...")

                results.append({
                    "prompt": prompt,
                    "baseline": baseline_text,
                    "intervention": intervention_text,
                    "baseline_has_target": baseline_has_target,
                    "intervention_has_target": intervention_has_target,
                    "hijacked": hijacked,
                })

            # Remove hooks
            for hook in hooks:
                hook.remove()

            print(f"  Hijack rate: {hijack_count}/4")

            if hijack_count > best_hijack_count:
                best_hijack_count = hijack_count
                best_clamp = test_clamp

        # Now do full test with best clamp value
        print(f"\n{'='*70}")
        print(f"FULL TEST with best clamp value: {best_clamp}")
        print(f"{'='*70}")

        # Reregister hooks with best clamp
        neurons_by_layer = {}
        for n in neurons:
            layer = n["layer"]
            if layer not in neurons_by_layer:
                neurons_by_layer[layer] = []
            neurons_by_layer[layer].append(n["neuron_idx"])

        intervention_active = [False]

        def make_final_hook(layer_name, indices):
            def hook(module, input, output):
                if not intervention_active[0]:
                    return output
                modified = output.clone()
                for idx in indices:
                    modified[:, :, idx] = best_clamp
                return modified
            return hook

        hooks = []
        for name, module in model.named_modules():
            if name in neurons_by_layer:
                hook = module.register_forward_hook(make_final_hook(name, neurons_by_layer[name]))
                hooks.append(hook)

        final_results = []
        total_hijacked = 0

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            intervention_active[0] = False
            with torch.no_grad():
                baseline_out = model.generate(
                    **inputs, max_new_tokens=60, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            baseline_text = tokenizer.decode(baseline_out[0], skip_special_tokens=True)

            intervention_active[0] = True
            with torch.no_grad():
                intervention_out = model.generate(
                    **inputs, max_new_tokens=60, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            intervention_text = tokenizer.decode(intervention_out[0], skip_special_tokens=True)

            baseline_has = any(kw.lower() in baseline_text.lower() for kw in target_keywords)
            intervention_has = any(kw.lower() in intervention_text.lower() for kw in target_keywords)
            hijacked = intervention_has and not baseline_has

            marker = "✅ HIJACKED" if hijacked else ("⚠️ Already had" if baseline_has else "❌ Failed")
            print(f"\n{marker}: \"{prompt}\"")
            print(f"  Baseline: ...{baseline_text[len(prompt):len(prompt)+80]}...")
            print(f"  Intervention: ...{intervention_text[len(prompt):len(prompt)+80]}...")

            if hijacked:
                total_hijacked += 1

            final_results.append({
                "prompt": prompt,
                "baseline": baseline_text,
                "intervention": intervention_text,
                "hijacked": hijacked,
            })

        for hook in hooks:
            hook.remove()

        # Summary
        print(f"\n{'='*70}")
        print(f"SUMMARY: {concept_data['name']}")
        print(f"{'='*70}")
        print(f"Best clamp value: {best_clamp}")
        print(f"Hijack rate: {total_hijacked}/{len(prompts)} ({100*total_hijacked/len(prompts):.0f}%)")

        if total_hijacked >= len(prompts) // 2:
            verdict = f"✅ STRONG: Can hijack prompts to say '{concept_data['name']}'!"
        elif total_hijacked >= 2:
            verdict = f"⚠️ WEAK: Sometimes hijacks to '{concept_data['name']}'"
        else:
            verdict = f"❌ FAIL: Cannot hijack prompts - these are NOT '{concept_data['name']}' neurons"

        print(f"Verdict: {verdict}")

        all_results[concept_key] = {
            "name": concept_data["name"],
            "best_clamp": best_clamp,
            "hijack_rate": total_hijacked / len(prompts),
            "results": final_results,
        }

    # Final
    print(f"\n\n{'='*70}")
    print("FINAL VERDICT: ARE THESE CONCEPT-SPECIFIC NEURONS?")
    print(f"{'='*70}")

    for key, data in all_results.items():
        rate = data["hijack_rate"]
        if rate >= 0.5:
            status = "✅ YES - True concept neurons!"
        elif rate >= 0.25:
            status = "⚠️ PARTIAL - Weak concept neurons"
        else:
            status = "❌ NO - Just category neurons"

        print(f"\n{data['name']}: {rate:.0%} hijack rate")
        print(f"  {status}")

    return all_results


@app.local_entrypoint()
def main():
    test_true_adversarial.remote()
    print("\n✅ Test complete!")



