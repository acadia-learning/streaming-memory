"""
Measure activation sparsity in Qwen3-8B during generation.
Captures MLP activations and computes statistics on how many neurons are "firing".
"""

from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def measure_activation_sparsity(
    model_name: str = "Qwen/Qwen3-8B",
    prompts: list[str] = None,
    max_new_tokens: int = 50,
    sparsity_threshold: float = 0.0,
    device: str = "auto",
):
    """
    Measure how sparse the MLP activations are during generation.

    Args:
        model_name: HuggingFace model name
        prompts: List of prompts to test
        max_new_tokens: Number of tokens to generate
        sparsity_threshold: Consider neurons "inactive" if |activation| <= threshold
        device: Device to use ("auto", "cuda", "mps", "cpu")
    """
    if prompts is None:
        prompts = [
            "The capital of France is",
            "In machine learning, gradient descent is used to",
            "The theory of relativity states that",
            "To make a good cup of coffee, you should",
            "The main difference between Python and JavaScript is",
        ]

    print(f"Loading model: {model_name}")
    print(f"Device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    # Storage for activations
    activation_stats = defaultdict(list)

    # Find MLP layers and register hooks
    hooks = []

    def make_hook(layer_name):
        def hook(module, input, output):
            # For Qwen, the MLP output is what we want to measure
            # The gate activation (after SiLU) determines sparsity
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output

            # Detach and move to CPU for analysis
            act = act.detach().float().cpu()

            # Compute sparsity metrics
            total_neurons = act.numel()

            # Various sparsity measures
            zero_count = (act.abs() <= sparsity_threshold).sum().item()
            near_zero_count = (act.abs() <= 0.01).sum().item()
            small_count = (act.abs() <= 0.1).sum().item()

            activation_stats[layer_name].append({
                'total': total_neurons,
                'zero': zero_count,
                'near_zero_001': (act.abs() <= 0.001).sum().item(),
                'near_zero_01': near_zero_count,
                'small_01': small_count,
                'mean_abs': act.abs().mean().item(),
                'max_abs': act.abs().max().item(),
                'std': act.std().item(),
            })
        return hook

    # Register hooks on MLP layers
    # Qwen uses a gated MLP with gate_proj, up_proj, down_proj
    for name, module in model.named_modules():
        # Hook into the MLP module or the gate activation
        if 'mlp' in name.lower() and name.endswith('mlp'):
            hook = module.register_forward_hook(make_hook(name))
            hooks.append(hook)
            print(f"Registered hook on: {name}")

    print(f"\nRegistered {len(hooks)} hooks")
    print(f"\nRunning {len(prompts)} generations with {max_new_tokens} tokens each...")
    print("=" * 60)

    # Run generations
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Prompt: {prompt[:50]}...")

        # Clear previous stats for this generation
        for key in activation_stats:
            activation_stats[key].clear()

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated[:100]}...")

        # Compute aggregate statistics
        print(f"\nSparsity Statistics (threshold={sparsity_threshold}):")
        print("-" * 60)

        total_activations = 0
        total_zero = 0
        total_near_zero = 0
        total_small = 0

        # Aggregate across all layers
        for layer_name, stats_list in activation_stats.items():
            if not stats_list:
                continue

            layer_total = sum(s['total'] for s in stats_list)
            layer_zero = sum(s['zero'] for s in stats_list)
            layer_near_zero = sum(s['near_zero_01'] for s in stats_list)
            layer_small = sum(s['small_01'] for s in stats_list)

            total_activations += layer_total
            total_zero += layer_zero
            total_near_zero += layer_near_zero
            total_small += layer_small

        if total_activations > 0:
            print(f"Total activations measured: {total_activations:,}")
            print(f"Exactly zero (|x| <= {sparsity_threshold}): {total_zero:,} ({100*total_zero/total_activations:.2f}%)")
            print(f"Near zero (|x| <= 0.01): {total_near_zero:,} ({100*total_near_zero/total_activations:.2f}%)")
            print(f"Small (|x| <= 0.1): {total_small:,} ({100*total_small/total_activations:.2f}%)")
            print(f"Active neurons (|x| > 0.01): {total_activations - total_near_zero:,} ({100*(total_activations - total_near_zero)/total_activations:.2f}%)")

        # Per-layer breakdown for first prompt
        if i == 0:
            print("\nPer-Layer Breakdown (first prompt):")
            print("-" * 60)
            for layer_name, stats_list in sorted(activation_stats.items()):
                if not stats_list:
                    continue
                avg_mean = np.mean([s['mean_abs'] for s in stats_list])
                avg_zero_pct = np.mean([100 * s['zero'] / s['total'] for s in stats_list])
                avg_near_zero_pct = np.mean([100 * s['near_zero_01'] / s['total'] for s in stats_list])
                print(f"{layer_name}: mean_abs={avg_mean:.4f}, zero%={avg_zero_pct:.1f}%, near_zero%={avg_near_zero_pct:.1f}%")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    print("\n" + "=" * 60)
    print("Measurement complete!")

    return activation_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Measure activation sparsity in Qwen3-8B")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model name")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max new tokens to generate")
    parser.add_argument("--threshold", type=float, default=0.0, help="Sparsity threshold")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/mps/cpu)")

    args = parser.parse_args()

    measure_activation_sparsity(
        model_name=args.model,
        max_new_tokens=args.max_tokens,
        sparsity_threshold=args.threshold,
        device=args.device,
    )


