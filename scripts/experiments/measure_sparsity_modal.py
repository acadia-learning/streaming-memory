"""
Modal deployment for measuring activation sparsity in Qwen3 models.
Run with: modal run scripts/experiments/measure_sparsity_modal.py
"""

import modal

app = modal.App("qwen3-sparsity-measurement")

# Models to test - download all at image build time
MODELS = [
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-30B-A3B",  # MoE model - 30B total, 3B active
]


def download_models():
    from huggingface_hub import snapshot_download
    for model in MODELS:
        print(f"Downloading {model}...")
        snapshot_download(model, ignore_patterns=["*.gguf"])
        print(f"Downloaded {model}")


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
    .run_function(download_models)
)


@app.function(
    image=image,
    gpu="A100-80GB",  # Need 80GB for 32B model
    timeout=1200,
)
def measure_sparsity(
    model_name: str = "Qwen/Qwen3-8B",
    prompts: list[str] | None = None,
    max_new_tokens: int = 50,
    sparsity_threshold: float = 0.0,
):
    """
    Measure how sparse the MLP activations are during generation.
    """
    from collections import defaultdict

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if prompts is None:
        prompts = [
            "The capital of France is",
            "In machine learning, gradient descent is used to",
            "The theory of relativity states that",
        ]

    print(f"\n{'='*70}")
    print(f"TESTING MODEL: {model_name}")
    print(f"{'='*70}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")

    # Storage for activations
    activation_stats = defaultdict(list)

    # Store intermediate activations from gate
    gate_activations = defaultdict(list)

    # Find MLP layers and register hooks
    hooks = []

    def make_mlp_hook(layer_name):
        """Hook for the full MLP output"""
        def hook(module, input, output):
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output

            act = act.detach().float().cpu()
            total_neurons = act.numel()

            activation_stats[layer_name].append({
                'total': total_neurons,
                'zero': (act.abs() <= sparsity_threshold).sum().item(),
                'near_zero_0001': (act.abs() <= 0.0001).sum().item(),
                'near_zero_001': (act.abs() <= 0.001).sum().item(),
                'near_zero_01': (act.abs() <= 0.01).sum().item(),
                'small_01': (act.abs() <= 0.1).sum().item(),
                'mean_abs': act.abs().mean().item(),
                'max_abs': act.abs().max().item(),
                'std': act.std().item(),
            })
        return hook

    def make_gate_hook(layer_name):
        """Hook specifically for gate_proj (before SiLU activation)"""
        def hook(module, input, output):
            # This captures the gate projection output
            act = output.detach().float().cpu()

            # Apply SiLU to see post-activation sparsity
            silu_act = torch.nn.functional.silu(act)

            total = act.numel()
            gate_activations[layer_name].append({
                'total': total,
                # Pre-SiLU stats
                'pre_silu_zero': (act.abs() <= sparsity_threshold).sum().item(),
                'pre_silu_negative': (act < 0).sum().item(),
                # Post-SiLU stats (SiLU suppresses negative values)
                'post_silu_near_zero_001': (silu_act.abs() <= 0.001).sum().item(),
                'post_silu_near_zero_01': (silu_act.abs() <= 0.01).sum().item(),
                'post_silu_small_01': (silu_act.abs() <= 0.1).sum().item(),
                'post_silu_mean': silu_act.abs().mean().item(),
            })
        return hook

    # Storage for MoE router statistics
    router_stats = defaultdict(list)

    def make_router_hook(layer_name):
        """Hook for MoE router to see expert selection sparsity"""
        def hook(module, input, output):
            # Router outputs are typically (router_logits, expert_indices, expert_weights)
            # or just the routing weights
            if isinstance(output, tuple):
                # Usually router_logits or routing weights
                routing = output[0] if len(output) > 0 else None
            else:
                routing = output

            if routing is not None:
                routing = routing.detach().float().cpu()
                # For top-k routing, count how many experts are selected
                total_experts = routing.shape[-1] if len(routing.shape) > 1 else 1
                router_stats[layer_name].append({
                    'total_experts': total_experts,
                    'routing_shape': list(routing.shape),
                })
        return hook

    # Register hooks
    is_moe = "A3B" in model_name or "MoE" in model_name.upper()

    for name, module in model.named_modules():
        # Hook on full MLP (for dense models) or experts (for MoE)
        if 'mlp' in name.lower() and name.endswith('mlp'):
            hook = module.register_forward_hook(make_mlp_hook(name))
            hooks.append(hook)
            print(f"MLP hook on: {name}")

        # Hook on gate_proj to see SiLU sparsity
        if 'gate_proj' in name and 'experts' not in name:
            hook = module.register_forward_hook(make_gate_hook(name))
            hooks.append(hook)
            print(f"Gate hook on: {name}")

        # For MoE models, hook on router
        if is_moe and ('router' in name.lower() or 'gate' in name.lower() and 'mlp_gate' not in name.lower()):
            if hasattr(module, 'weight'):  # It's a linear layer (router)
                hook = module.register_forward_hook(make_router_hook(name))
                hooks.append(hook)
                print(f"Router hook on: {name}")

    print(f"\nRegistered {len(hooks)} hooks")
    print(f"\nRunning {len(prompts)} generations with {max_new_tokens} tokens each...")
    print("=" * 70)

    results = []

    # Run generations
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Prompt: {prompt[:50]}...")

        # Clear previous stats
        for key in activation_stats:
            activation_stats[key].clear()
        for key in gate_activations:
            gate_activations[key].clear()

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated[:100]}...")

        # Compute aggregate statistics for MLP outputs
        print(f"\n=== MLP Output Sparsity (threshold={sparsity_threshold}) ===")

        total_activations = 0
        total_zero = 0
        total_near_zero_01 = 0
        total_small = 0

        for layer_name, stats_list in activation_stats.items():
            if not stats_list:
                continue
            total_activations += sum(s['total'] for s in stats_list)
            total_zero += sum(s['zero'] for s in stats_list)
            total_near_zero_01 += sum(s['near_zero_01'] for s in stats_list)
            total_small += sum(s['small_01'] for s in stats_list)

        if total_activations > 0:
            print(f"Total MLP activations: {total_activations:,}")
            print(f"  Exactly zero: {total_zero:,} ({100*total_zero/total_activations:.2f}%)")
            print(f"  Near zero (|x|<=0.01): {total_near_zero_01:,} ({100*total_near_zero_01/total_activations:.2f}%)")
            print(f"  Small (|x|<=0.1): {total_small:,} ({100*total_small/total_activations:.2f}%)")
            print(f"  ACTIVE (|x|>0.01): {total_activations - total_near_zero_01:,} ({100*(total_activations - total_near_zero_01)/total_activations:.2f}%)")

        # Compute gate/SiLU statistics
        print("\n=== Gate Activation Sparsity (SiLU) ===")

        gate_total = 0
        gate_negative = 0
        gate_post_near_zero = 0
        gate_post_small = 0

        for layer_name, stats_list in gate_activations.items():
            if not stats_list:
                continue
            gate_total += sum(s['total'] for s in stats_list)
            gate_negative += sum(s['pre_silu_negative'] for s in stats_list)
            gate_post_near_zero += sum(s['post_silu_near_zero_01'] for s in stats_list)
            gate_post_small += sum(s['post_silu_small_01'] for s in stats_list)

        if gate_total > 0:
            print(f"Total gate activations: {gate_total:,}")
            print(f"  Pre-SiLU negative (suppressed): {gate_negative:,} ({100*gate_negative/gate_total:.2f}%)")
            print(f"  Post-SiLU near zero (|x|<=0.01): {gate_post_near_zero:,} ({100*gate_post_near_zero/gate_total:.2f}%)")
            print(f"  Post-SiLU small (|x|<=0.1): {gate_post_small:,} ({100*gate_post_small/gate_total:.2f}%)")
            print(f"  Post-SiLU ACTIVE (|x|>0.01): {gate_total - gate_post_near_zero:,} ({100*(gate_total - gate_post_near_zero)/gate_total:.2f}%)")

        # For MoE models, report router statistics
        if is_moe and router_stats:
            print("\n=== MoE Router Statistics ===")
            for layer_name, stats_list in list(router_stats.items())[:3]:
                if stats_list:
                    print(f"  {layer_name}: shape={stats_list[0]['routing_shape']}, num_experts={stats_list[0]['total_experts']}")
            # MoE models use top-k routing, typically k=2 out of 64 experts = 3.125% active
            # This is the TRUE sparsity metric for MoE
            if router_stats:
                first_stats = list(router_stats.values())[0]
                if first_stats:
                    num_experts = first_stats[0]['total_experts']
                    # Qwen3-30B-A3B uses top-2 routing
                    k = 2  # typical for Qwen MoE
                    expert_sparsity = 100 * (1 - k / num_experts)
                    print(f"  Expert-level sparsity: {expert_sparsity:.1f}% (top-{k} of {num_experts} experts)")

        result = {
            'prompt': prompt,
            'generated': generated,
            'mlp_total': total_activations,
            'mlp_active_pct': 100 * (total_activations - total_near_zero_01) / total_activations if total_activations > 0 else 0,
            'gate_total': gate_total,
            'gate_active_pct': 100 * (gate_total - gate_post_near_zero) / gate_total if gate_total > 0 else 0,
            'gate_negative_pct': 100 * gate_negative / gate_total if gate_total > 0 else 0,
            'is_moe': is_moe,
        }
        results.append(result)

        # Per-layer breakdown for first prompt
        if i == 0:
            print("\n=== Per-Layer Gate Stats (first prompt) ===")
            for layer_name in sorted(gate_activations.keys())[:5]:  # First 5 layers
                stats_list = gate_activations[layer_name]
                if not stats_list:
                    continue
                avg_neg_pct = np.mean([100 * s['pre_silu_negative'] / s['total'] for s in stats_list])
                avg_active_pct = np.mean([100 * (s['total'] - s['post_silu_near_zero_01']) / s['total'] for s in stats_list])
                print(f"  {layer_name}: negative={avg_neg_pct:.1f}%, active={avg_active_pct:.1f}%")
            print("  ...")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY ACROSS ALL PROMPTS")
    print("=" * 70)

    avg_mlp_active = np.mean([r['mlp_active_pct'] for r in results])
    avg_gate_active = np.mean([r['gate_active_pct'] for r in results])
    avg_gate_negative = np.mean([r['gate_negative_pct'] for r in results])

    print(f"Average MLP output active neurons: {avg_mlp_active:.2f}%")
    print(f"Average Gate active neurons (post-SiLU): {avg_gate_active:.2f}%")
    print(f"Average Gate negative (suppressed by SiLU): {avg_gate_negative:.2f}%")
    print(f"\n=> Approximately {100-avg_gate_active:.1f}% of gate neurons are effectively inactive per token")

    return results


@app.local_entrypoint()
def main():
    models_to_test = [
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-30B-A3B",  # MoE
    ]

    all_results = {}
    for model in models_to_test:
        print(f"\n\n{'#'*70}")
        print(f"# Testing: {model}")
        print(f"{'#'*70}")
        results = measure_sparsity.remote(model_name=model)
        all_results[model] = results

    # Final comparison
    print("\n\n" + "="*70)
    print("FINAL COMPARISON ACROSS MODELS")
    print("="*70)
    for model, results in all_results.items():
        avg_mlp = sum(r['mlp_active_pct'] for r in results) / len(results)
        avg_gate = sum(r['gate_active_pct'] for r in results) / len(results)
        print(f"\n{model}:")
        print(f"  MLP active: {avg_mlp:.1f}%")
        print(f"  Gate active: {avg_gate:.1f}%")

