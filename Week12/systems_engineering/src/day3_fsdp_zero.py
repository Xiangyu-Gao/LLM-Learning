"""
day3_fsdp_zero.py — FSDP and ZeRO: Distributed Training Memory
================================================================

CORE INSIGHT
------------
Training a 7B model needs ~112 GB of GPU memory for parameters, gradients,
and optimizer states alone (see Day 1).  Single GPUs max at 80 GB (A100).
The solution is to SHARD memory across multiple GPUs.

TWO APPROACHES
--------------

1. ZeRO (Zero Redundancy Optimizer) — DeepSpeed
   ZeRO eliminates redundancy: each GPU holds a SHARD of the state,
   not a full copy.  Three progressive stages:

   Stage 1 — Optimizer state sharding
     Each GPU holds 1/N of the Adam m, v tensors.
     Memory: params(full) + grads(full) + opt_states(1/N)
     Savings: reduces optimizer from 12B to (2 + 2 + 12/N) bytes/param

   Stage 2 — Gradient sharding (+ Stage 1)
     Each GPU holds 1/N of the gradients too.
     Memory: params(full) + grads(1/N) + opt_states(1/N)
     Savings: useful when gradients are large

   Stage 3 — Parameter sharding (+ Stage 1+2)
     Full model is sharded: each GPU holds only 1/N of parameters.
     Memory: params(1/N) + grads(1/N) + opt_states(1/N)
     Total static bytes/param: (2 + 2 + 12) / N = 16/N
     Savings: maximum — enables training 70B+ models on standard clusters

2. FSDP (Fully Sharded Data Parallel) — PyTorch native
   PyTorch's built-in ZeRO-3 equivalent.
   Shards parameters, gradients, and optimizer states across data-parallel ranks.
   Each rank fetches ("all-gathers") the full parameter tensor only when needed
   for a forward or backward pass, then immediately discards ("reduce-scatter").

   FSDP wrapping strategies:
   - NO_SHARD         → standard data parallel (no memory savings)
   - SHARD_GRAD_OP    → ZeRO-2 equivalent
   - FULL_SHARD       → ZeRO-3 equivalent (maximum savings)
   - HYBRID_SHARD     → shard within a node, replicate across nodes

MEMORY SAVINGS FORMULA (ZeRO-3 / FSDP FULL_SHARD)
--------------------------------------------------
   Per-GPU memory = Total_static / N  (roughly)
   For 7B model on 8 GPUs: 112 GB / 8 = 14 GB per GPU ✓

COMMUNICATION OVERHEAD
----------------------
   ZeRO-3 needs two collective operations per layer per step:
   - All-gather on forward pass  (to reassemble params)
   - Reduce-scatter on backward  (to shard gradients)
   Total extra data moved ≈ 3× parameter size per step.
   This is why ZeRO-3 becomes communication-bound on slow networks (e.g., PCIe).
   On NVLink (600 GB/s) it's negligible; on 10 GbE it can dominate.

WHEN ZeRO-3 HELPS
-----------------
   ✓ Model is too large to fit on 1 GPU even for inference
   ✓ You have fast intra-node interconnects (NVLink)
   ✓ Batch size per GPU is large (amortizes comm cost)
   ✗ Model fits on 1 GPU (data parallel is simpler)
   ✗ Slow network (inter-node Ethernet)

EXPERIMENT
----------
   1. ZeRO memory calculator: show per-GPU memory for stages 0-3
      across N=1,2,4,8 GPUs for GPT-2 and 7B configs.
   2. FSDP single-GPU demo:
      - Wrap GPT-2 with FSDP (FULL_SHARD on 1 GPU is a no-op for savings
        but demonstrates the API and wrapping pattern).
   3. ASCII diagrams of parameter sharding.

OUTPUT
------
    results/day3/day3_zero_memory.png
    results/day3/day3_summary.txt

NOTE
----
   FSDP requires torch.distributed. On a single GPU, we init a process
   group with world_size=1. No memory savings (nothing to shard across),
   but the API usage is identical to multi-GPU.
   For a real multi-GPU experiment: torchrun --nproc_per_node=N day3_fsdp_zero.py
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.gpt2.modeling_gpt2 import GPT2Block
    import functools
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "day3")


# ════════════════════════════════════════════════════════════════════════════
# 1.  ZeRO MEMORY CALCULATOR
# ════════════════════════════════════════════════════════════════════════════

def zero_memory_per_gpu(num_params: int, world_size: int, zero_stage: int,
                         mixed_precision: bool = True) -> dict:
    """
    Compute per-GPU static memory (bytes) for ZeRO stages 0-3.

    Assumptions:
        - FP16 weights if mixed_precision, else FP32
        - Adam optimizer (12 bytes/param for full FP32 states)
        - FP16 gradients during backward
    """
    w = 2 if mixed_precision else 4   # bytes per weight param

    N = world_size

    if zero_stage == 0:
        # Standard data parallel: full copy on every GPU
        params  = num_params * w
        grads   = num_params * w
        opt     = num_params * 12      # FP32 Adam states
    elif zero_stage == 1:
        # Shard optimizer states only
        params  = num_params * w
        grads   = num_params * w
        opt     = num_params * 12 / N
    elif zero_stage == 2:
        # Shard optimizer + gradients
        params  = num_params * w
        grads   = num_params * w / N
        opt     = num_params * 12 / N
    else:  # stage 3
        # Shard everything
        params  = num_params * w / N
        grads   = num_params * w / N
        opt     = num_params * 12 / N

    total = params + grads + opt
    return {
        "params_gb":  params  / 1024**3,
        "grads_gb":   grads   / 1024**3,
        "opt_gb":     opt     / 1024**3,
        "total_gb":   total   / 1024**3,
    }


def run_zero_calculator(smoke: bool = False) -> None:
    configs = {
        "GPT-2 (124M)":     124_000_000,
        "7B (LLaMA)":     7_000_000_000,
        "70B (LLaMA)": 70_000_000_000,
    }
    gpu_counts = [1, 2, 4, 8] if smoke else [1, 2, 4, 8, 16, 64]

    print(f"\n{'═'*70}")
    print("  ZeRO MEMORY CALCULATOR (per-GPU, mixed precision FP16)")
    print(f"{'═'*70}")

    all_results = {}

    for model_name, num_params in configs.items():
        print(f"\n  ── {model_name} ({num_params/1e9:.1f}B params) ──")
        print(f"  {'Stage':<10} {'GPUs':<6} {'Params':>9} {'Grads':>9} {'Opt':>9} {'Total':>9}")
        print(f"  {'─'*58}")

        model_results = {}
        for stage in range(4):
            for ngpu in gpu_counts:
                m = zero_memory_per_gpu(num_params, ngpu, stage)
                row = (
                    f"  ZeRO-{stage:<6} {ngpu:<6} "
                    f"{m['params_gb']:>8.1f}G {m['grads_gb']:>8.1f}G "
                    f"{m['opt_gb']:>8.1f}G {m['total_gb']:>8.1f}G"
                )
                print(row)
                model_results[(stage, ngpu)] = m
        all_results[model_name] = model_results

    return all_results


def plot_zero_savings(all_results: dict, gpu_counts: list) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    models = list(all_results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=False)
    if n_models == 1:
        axes = [axes]

    colors = {0: "salmon", 1: "gold", 2: "lightblue", 3: "mediumseagreen"}

    for ax, model_name in zip(axes, models):
        data = all_results[model_name]
        for stage in range(4):
            mem = [data[(stage, ng)]["total_gb"] for ng in gpu_counts if (stage, ng) in data]
            gn  = [ng for ng in gpu_counts if (stage, ng) in data]
            ax.plot(gn, mem, "o-", color=colors[stage], label=f"ZeRO-{stage}", linewidth=2)

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Number of GPUs")
        ax.set_ylabel("Per-GPU Memory (GB)")
        ax.set_title(model_name)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.axhline(80, color="gray", linestyle="--", alpha=0.5, label="A100 limit (80 GB)")

    plt.suptitle("ZeRO Per-GPU Memory vs Number of GPUs\n(mixed precision FP16, Adam optimizer)")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "day3_zero_memory.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\n  ZeRO memory plot → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# 2.  ASCII DIAGRAMS
# ════════════════════════════════════════════════════════════════════════════

DIAGRAMS = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STANDARD DATA PARALLEL (ZeRO-0)
Each GPU holds a FULL copy of everything.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  GPU 0: [P₁ P₂ P₃] [G₁ G₂ G₃] [O₁ O₂ O₃]
  GPU 1: [P₁ P₂ P₃] [G₁ G₂ G₃] [O₁ O₂ O₃]  ← identical!
  GPU 2: [P₁ P₂ P₃] [G₁ G₂ G₃] [O₁ O₂ O₃]  ← identical!
  GPU 3: [P₁ P₂ P₃] [G₁ G₂ G₃] [O₁ O₂ O₃]  ← identical!

  Params = P, Grads = G, Optimizer states = O
  Total memory per GPU: 16× param_bytes (FP16, Adam)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ZeRO STAGE 1 — Optimizer State Sharding
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  GPU 0: [P₁ P₂ P₃] [G₁ G₂ G₃] [O₁    ]   ← only its shard of O
  GPU 1: [P₁ P₂ P₃] [G₁ G₂ G₃] [   O₂ ]
  GPU 2: [P₁ P₂ P₃] [G₁ G₂ G₃] [      O₃]
  GPU 3: [P₁ P₂ P₃] [G₁ G₂ G₃] [         ]  (no O needed if N>3)

  Saves: optimizer states ÷ N (e.g., 12B bytes → 3B per GPU for N=4)
  Communication: all-reduce on gradients (same as DDP)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ZeRO STAGE 2 — + Gradient Sharding
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  GPU 0: [P₁ P₂ P₃] [G₁    ] [O₁    ]
  GPU 1: [P₁ P₂ P₃] [   G₂ ] [   O₂ ]
  GPU 2: [P₁ P₂ P₃] [      G₃] [      O₃]

  Saves: grads/N + optimizer/N
  Communication: reduce-scatter on gradients

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ZeRO STAGE 3 / FSDP FULL_SHARD — Everything Sharded
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  GPU 0: [P₁    ] [G₁    ] [O₁    ]
  GPU 1: [   P₂ ] [   G₂ ] [   O₂ ]
  GPU 2: [      P₃] [      G₃] [      O₃]

  Per-GPU memory: 16×param_bytes / N  (minimum!)
  For 7B on 8 GPUs:  112 GB / 8 = 14 GB per GPU ✓

  Forward pass:   ALL-GATHER params just before each layer
  Backward pass:  REDUCE-SCATTER gradients after each layer
  After backward: DISCARD params (only keep shard)

  Cost: 3× model size in extra communication per training step
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# ════════════════════════════════════════════════════════════════════════════
# 3.  FSDP SINGLE-GPU DEMO
# ════════════════════════════════════════════════════════════════════════════

def run_fsdp_demo(smoke: bool = False) -> list[str]:
    """
    Demonstrate FSDP wrapping on a single GPU.
    World_size=1 means no real sharding, but the API is identical
    to a multi-GPU run (just run with torchrun --nproc_per_node=N for real sharding).
    """
    if not HAS_TORCH:
        return ["[SKIP] torch not available"]
    if not torch.cuda.is_available():
        return ["[SKIP] CUDA not available — FSDP requires CUDA"]

    lines = []
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print(f"\n{'─'*56}")
    print("  FSDP DEMO (single GPU — API demonstration)")
    print(f"  For real sharding: torchrun --nproc_per_node=N day3_fsdp_zero.py")
    print(f"{'─'*56}")

    # Init process group (required for FSDP even on 1 GPU)
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29501")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"  Distributed: rank={rank}, world_size={world_size}")
    lines.append(f"rank={rank}, world_size={world_size}")

    # Load model without FSDP first and measure
    def get_mem():
        return torch.cuda.memory_allocated() / 1024**2

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model_name = "gpt2"
    model_dtype = torch.float16
    tokenizer  = AutoTokenizer.from_pretrained(model_name, use_safetensors=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Baseline: regular model
    print(f"\n  Loading {model_name} WITHOUT FSDP...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, use_safetensors=True, dtype=model_dtype,
    ).to(device)
    mem_base = get_mem()
    param_count = sum(p.numel() for p in base_model.parameters())
    print(f"  Parameters:    {param_count:,}")
    print(f"  Memory (base): {mem_base:.1f} MB")
    lines.append(f"Without FSDP: {mem_base:.1f} MB, {param_count:,} params")

    del base_model
    torch.cuda.empty_cache()

    # FSDP wrapped model
    print(f"\n  Loading {model_name} WITH FSDP (FULL_SHARD)...")
    raw_model = AutoModelForCausalLM.from_pretrained(
        model_name, use_safetensors=True, dtype=model_dtype,
    )  # keep dtype aligned with baseline so the memory comparison is fair

    # Define which module type to wrap (one FSDP unit per transformer block)
    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={GPT2Block},
    )

    mp_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

    fsdp_model = FSDP(
        raw_model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp_policy,
        device_id=device,
    )

    mem_fsdp = get_mem()
    print(f"  Memory (FSDP, world_size=1): {mem_fsdp:.1f} MB")
    print(f"  NOTE: world_size=1 forces NO_SHARD, so this is API overhead only.")
    print(f"  Real per-GPU savings require launching with torchrun on multiple GPUs.")
    lines.append(f"With FSDP (ws=1): {mem_fsdp:.1f} MB")
    lines.append("world_size=1 uses NO_SHARD; run multi-GPU to measure real savings")

    # Quick forward pass to verify FSDP works
    print(f"\n  Running forward pass through FSDP model...")
    prompt = "The transformer is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = fsdp_model(**inputs)
    print(f"  Forward pass OK — logits shape: {out.logits.shape}")
    lines.append(f"Forward pass OK, logits: {out.logits.shape}")

    # Print FSDP structure
    print(f"\n  FSDP module structure (first 200 chars):")
    fsdp_str = str(fsdp_model)[:200].replace("\n", "\n  ")
    print(f"  {fsdp_str}...")

    dist.destroy_process_group()
    return lines


# ════════════════════════════════════════════════════════════════════════════
# 4.  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-fsdp", action="store_true",
                        help="Skip FSDP demo (useful if process group fails)")
    args = parser.parse_args()
    smoke = args.smoke

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("═"*70)
    print("  Day 3 — FSDP and ZeRO: Distributed Training Memory")
    print("═"*70)

    # Print ASCII diagrams
    print(DIAGRAMS)

    # ZeRO calculator
    gpu_counts = [1, 2, 4, 8] if smoke else [1, 2, 4, 8, 16, 64]
    all_results = run_zero_calculator(smoke)
    plot_zero_savings(all_results, gpu_counts)

    # FSDP demo
    fsdp_lines = []
    if not args.skip_fsdp:
        try:
            fsdp_lines = run_fsdp_demo(smoke)
        except Exception as e:
            print(f"\n  [WARN] FSDP demo failed: {e}")
            print("  Re-run with --skip-fsdp to bypass this section.")
            fsdp_lines = [f"FSDP demo error: {e}"]

    # Write summary
    summary_path = os.path.join(RESULTS_DIR, "day3_summary.txt")
    with open(summary_path, "w") as f:
        f.write("FSDP AND ZeRO — DISTRIBUTED TRAINING MEMORY\n")
        f.write("=" * 60 + "\n\n")
        f.write(DIAGRAMS)
        f.write("\nZeRO STAGE MEMORY FORMULA\n")
        f.write("-" * 40 + "\n")
        f.write("  Per-GPU bytes/param (FP16 weights, Adam):\n")
        f.write("  Stage 0: 16     (full copy: 2+2+12)\n")
        f.write("  Stage 1: (4 + 12/N)\n")
        f.write("  Stage 2: (2 + 2/N + 12/N) = (2 + 14/N)\n")
        f.write("  Stage 3: 16/N   (everything sharded)\n\n")
        f.write("  7B model on 8 GPUs, Stage 3: 16×7B/8 = 14 GB per GPU\n\n")
        f.write("INTERVIEW ANSWERS\n")
        f.write("-" * 40 + "\n")
        f.write("Q: Difference between FSDP and ZeRO?\n")
        f.write("A: ZeRO is DeepSpeed's strategy; FSDP is PyTorch's native equivalent.\n")
        f.write("   Both implement ZeRO-3-style sharding. FSDP integrates natively with\n")
        f.write("   PyTorch's autograd and composites better with other torch features.\n\n")
        f.write("Q: When does ZeRO-3 help?\n")
        f.write("A: When the model doesn't fit on a single GPU. For a 7B model at FP16\n")
        f.write("   training, static memory alone is 112 GB. ZeRO-3 on 8 A100s brings\n")
        f.write("   that to 14 GB per GPU — within budget.\n\n")
        f.write("Q: Why does communication become the bottleneck?\n")
        f.write("A: ZeRO-3 does an all-gather (assemble full params) before each layer's\n")
        f.write("   forward pass and a reduce-scatter (shard gradients) after each layer's\n")
        f.write("   backward. The total communication volume is ≈3× model size per step.\n")
        f.write("   On NVLink (600 GB/s intra-node) this is fast. On 10 GbE Ethernet\n")
        f.write("   inter-node it stalls the compute pipeline.\n\n")
        if fsdp_lines:
            f.write("FSDP DEMO RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write("\n".join(fsdp_lines))

    print(f"\n  Summary → {summary_path}")
    print("\n  Day 3 complete.")
    print("  Key takeaway: ZeRO-3 / FSDP FULL_SHARD scales memory as O(1/N).")
    print("  Communication cost is 3× model size per step — fast links essential.")


if __name__ == "__main__":
    main()
