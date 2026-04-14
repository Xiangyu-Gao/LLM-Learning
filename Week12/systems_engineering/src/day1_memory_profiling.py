"""
day1_memory_profiling.py — Where Memory Actually Goes in Transformers
=====================================================================

CORE INSIGHT
------------
Memory in a transformer training run comes from four distinct buckets:

    1. Parameters          — the model weights themselves
    2. Gradients           — same shape as parameters (one per param)
    3. Optimizer states    — Adam keeps TWO FP32 copies per param (m, v)
                            plus an FP32 master-weight copy → 12 bytes/param
    4. Activations         — intermediate tensors saved for the backward pass
                            scale with batch_size × seq_len × hidden_dim
    5. KV cache            — attention keys/values accumulated during
                            autoregressive generation; grows with seq_len

FORMULA (training, Adam, FP16 mixed precision)
----------------------------------------------
    params_bytes      = N * 2                  (FP16 weights)
    grads_bytes       = N * 2                  (FP16 grads)
    optimizer_bytes   = N * (4 + 4 + 4)        (m, v, master in FP32)
    total_static      ≈ N * 16 bytes           ≈ 16x the weight size

    For N = 7B: 16 × 7e9 ≈ 112 GB — needs 2× 80 GB A100s just to hold state.

ACTIVATION MEMORY (per layer, approximate)
-------------------------------------------
    Each transformer layer saves roughly 8 intermediate tensors for backprop.
    The dominant ones:
        qkv_proj:   B × S × 3H          (query, key, value)
        attn_scores: B × heads × S × S   (the attention matrix — O(S²)!)
        attn_out:   B × S × H
        ffn_mid:    B × S × 4H
    Total per layer ≈ B × S × (8H + heads × S)  bytes × dtype_size

    With gradient checkpointing: only one checkpoint per N layers is stored;
    activations are RECOMPUTED on the backward pass at the cost of ~33% FLOPs.

KV CACHE (inference)
--------------------
    For each generated token we cache the K and V tensors for every layer:
        kv_cache = 2 × L × B × S × H × dtype_size
    This grows LINEARLY with sequence length.
    For a 7B model at seq=2048, batch=1, FP16:
        2 × 32 × 1 × 2048 × 4096 × 2 = 1.07 GB

EXPERIMENT
----------
    1. Derive and print the formula for a 7B model.
    2. Profile GPT-2 on a single forward+backward pass:
       - peak memory before/after gradient checkpointing
    3. Simulate KV cache growth across sequence lengths 128→4096.

OUTPUT
------
    results/day1/day1_summary.txt
    results/day1/day1_kv_cache.png

INTERVIEW TAKEAWAY
------------------
Q: "Why does memory scale with sequence length?"
A: The self-attention matrix is O(B × heads × S²).  At S=2048 with
   typical heads this adds hundreds of MB per layer.  At training time
   ALL these tensors are kept alive for the backward pass unless you
   use gradient checkpointing or FlashAttention (which fuses the kernel
   and never materializes the full S×S matrix).
"""

import argparse
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── optional torch import ────────────────────────────────────────────────────
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "day1")


# ════════════════════════════════════════════════════════════════════════════
# 1.  THEORETICAL MEMORY FORMULA
# ════════════════════════════════════════════════════════════════════════════

def bytes_to_gb(b: float) -> float:
    return b / 1024**3


def theoretical_memory(
    num_params: int,
    batch_size: int,
    seq_len: int,
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    ffn_multiplier: int = 4,
    training: bool = True,
    mixed_precision: bool = True,
    gradient_checkpointing: bool = False,
) -> dict:
    """
    Compute approximate memory breakdown for a transformer model.

    Parameters are in bytes; return values are in GB.
    """
    # ── dtype sizes ──────────────────────────────────────────────────────────
    w_bytes = 2 if mixed_precision else 4   # FP16 or FP32 weights
    is_training = training

    # ── 1. Parameters ────────────────────────────────────────────────────────
    param_mem = num_params * w_bytes

    # ── 2. Gradients ─────────────────────────────────────────────────────────
    grad_mem = num_params * w_bytes if is_training else 0

    # ── 3. Optimizer states (Adam) ────────────────────────────────────────────
    # FP32 master weights + FP32 momentum (m) + FP32 variance (v)
    opt_mem = num_params * 3 * 4 if is_training else 0   # 12 bytes/param

    # ── 4. Activations ────────────────────────────────────────────────────────
    # Per layer:  qkv (3H) + attn_scores (heads × S) + attn_out (H) + ffn (4H)
    #             ≈ (8H + num_heads * seq_len) per (B × S) element
    head_dim = hidden_dim // num_heads
    acts_per_layer = batch_size * seq_len * (
        3 * hidden_dim          # Q, K, V projections
        + num_heads * seq_len   # attention score matrix (S×S per head, over batch)
        + hidden_dim            # attention output
        + ffn_multiplier * hidden_dim  # FFN intermediate
        + hidden_dim            # layer norm inputs (2 per layer ≈ 1H each)
    )
    act_bytes = 2  # FP16 activations
    if is_training:
        total_act_mem = num_layers * acts_per_layer * act_bytes

        if gradient_checkpointing:
            # Checkpointing stores 1/sqrt(L) of activations; recomputes the rest
            ckpt_layers = max(1, int(math.sqrt(num_layers)))
            total_act_mem = ckpt_layers * acts_per_layer * act_bytes
            act_note = f"(checkpointed: {ckpt_layers}/{num_layers} layers stored)"
        else:
            act_note = "(full: all layers stored)"
    else:
        # In inference/validation, activations are transient and are not kept
        # alive for a backward pass, so the persistent activation footprint is
        # negligible compared with parameters and KV cache.
        total_act_mem = 0
        act_note = "(inference: transient, not stored for backward)"

    # ── 5. KV cache (inference) ───────────────────────────────────────────────
    # 2 tensors (K, V) × L layers × B × S × H
    kv_mem = 2 * num_layers * batch_size * seq_len * hidden_dim * w_bytes if not is_training else 0

    static_mem = param_mem + grad_mem + opt_mem
    total_mem = static_mem + total_act_mem + kv_mem

    return {
        "parameters_gb":   bytes_to_gb(param_mem),
        "gradients_gb":    bytes_to_gb(grad_mem),
        "optimizer_gb":    bytes_to_gb(opt_mem),
        "activations_gb":  bytes_to_gb(total_act_mem),
        "kv_cache_gb":     bytes_to_gb(kv_mem),
        "static_total_gb": bytes_to_gb(static_mem),
        "grand_total_gb":  bytes_to_gb(total_mem),
        "act_note":        act_note,
    }


def print_memory_table(label: str, mem: dict) -> None:
    print(f"\n{'─'*56}")
    print(f"  {label}")
    print(f"{'─'*56}")
    print(f"  Parameters:      {mem['parameters_gb']:8.2f} GB")
    print(f"  Gradients:       {mem['gradients_gb']:8.2f} GB")
    print(f"  Optimizer (Adam):{mem['optimizer_gb']:8.2f} GB")
    print(f"  Activations:     {mem['activations_gb']:8.2f} GB  {mem['act_note']}")
    print(f"  KV Cache:        {mem['kv_cache_gb']:8.2f} GB")
    print(f"  {'─'*40}")
    print(f"  Static total:    {mem['static_total_gb']:8.2f} GB  (params+grads+opt)")
    print(f"  GRAND TOTAL:     {mem['grand_total_gb']:8.2f} GB")
    print(f"{'─'*56}")


def run_theoretical(smoke: bool = False) -> str:
    """Print memory breakdown for representative models."""
    lines = []

    configs = {
        "GPT-2 (124M)": dict(
            num_params=124_000_000,
            num_layers=12,
            hidden_dim=768,
            num_heads=12,
        ),
        "7B (LLaMA-style)": dict(
            num_params=7_000_000_000,
            num_layers=32,
            hidden_dim=4096,
            num_heads=32,
        ),
        "13B (LLaMA-style)": dict(
            num_params=13_000_000_000,
            num_layers=40,
            hidden_dim=5120,
            num_heads=40,
        ),
    }

    BATCH = 8
    SEQ   = 2048

    header = "\n" + "═"*56 + "\n  THEORETICAL MEMORY ANALYSIS\n" + "═"*56
    print(header)
    lines.append(header)

    for name, cfg in configs.items():
        # Training without gradient checkpointing
        m_full = theoretical_memory(
            **cfg, batch_size=BATCH, seq_len=SEQ,
            training=True, gradient_checkpointing=False,
        )
        # Training WITH gradient checkpointing
        m_ckpt = theoretical_memory(
            **cfg, batch_size=BATCH, seq_len=SEQ,
            training=True, gradient_checkpointing=True,
        )
        # Inference (no grads, with KV cache)
        infer_batch = 1
        m_infer = theoretical_memory(
            **cfg, batch_size=infer_batch, seq_len=SEQ,
            training=False,
        )

        label = f"{name}  |  batch={BATCH}  seq={SEQ}"
        infer_label = f"{name}  |  batch={infer_batch}  seq={SEQ}"
        print_memory_table(f"Training (no ckpt) — {label}", m_full)
        print_memory_table(f"Training (w/ ckpt) — {label}", m_ckpt)
        print_memory_table(f"Inference          — {infer_label}", m_infer)

        saving = m_full["activations_gb"] - m_ckpt["activations_gb"]
        note = (
            f"\n  Gradient checkpointing saves {saving:.1f} GB of activations "
            f"({saving/m_full['activations_gb']*100:.0f}% reduction)\n"
        )
        print(note)
        lines.append(note)

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# 2.  PRACTICAL GPU PROFILING WITH GPT-2
# ════════════════════════════════════════════════════════════════════════════

def mb(bytes_val: float) -> str:
    return f"{bytes_val / 1024**2:.1f} MB"


def profile_gpt2(smoke: bool = False) -> list[str]:
    """
    Profile actual GPU memory usage for GPT-2 with and without
    gradient checkpointing.
    """
    if not HAS_TORCH:
        return ["[SKIP] torch not available"]
    if not torch.cuda.is_available():
        return ["[SKIP] CUDA not available — run on a GPU machine"]

    lines = []
    device = torch.device("cuda")
    model_name = "gpt2"  # 124M params — loads fast

    requested_seq_lens = [128, 256, 512] if smoke else [128, 256, 512, 1024, 2048]
    batch = 2 if smoke else 4

    print(f"\n{'═'*70}")
    print("  GPU PROFILING: GPT-2 (124M)")
    print(f"  Device: {torch.cuda.get_device_name()}")
    print(f"  batch={batch}, requested_seq_lens={requested_seq_lens}")
    print(f"  Columns: fwd_acts = memory allocated after forward (before backward)")
    print(f"           peak     = max memory across entire forward+backward")
    print(f"{'═'*70}")
    lines.append(f"GPU: {torch.cuda.get_device_name()}")

    def reset():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def load_model(gradient_checkpointing: bool):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to(device)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        return model

    def get_model_limits(model) -> tuple[int, int]:
        max_positions = getattr(model.config, "max_position_embeddings", None)
        if max_positions is None:
            max_positions = getattr(model.config, "n_positions", None)
        if max_positions is None:
            max_positions = max(requested_seq_lens)
        vocab_size = getattr(model.config, "vocab_size", 50257)
        return max_positions, vocab_size

    # MEASUREMENT STRATEGY
    # --------------------
    # We profile the transformer TRUNK (model.transformer), not the full LM head.
    # Reason: gradient checkpointing applies to transformer blocks. The LM head
    # outputs a logit tensor of shape [B, S, vocab_size] ≈ 49 MB for GPT-2 at
    # seq=256. This tensor must be materialized regardless of checkpointing
    # (it's needed for loss computation), and its size MASKS the activation savings.
    #
    # By profiling the trunk only, we measure exactly what checkpointing reduces:
    # the intermediate Q, K, V, attention weight, FFN tensors within each block.
    #
    # We use a scalar proxy loss (h.float().mean()) to trigger grad computation.
    # This keeps the "output" to a single float — negligible — so only the
    # transformer block activations dominate the overhead measurement.

    import gc

    results = {}
    for use_ckpt in [False, True]:
        label = "w/ gradient_checkpointing" if use_ckpt else "no checkpointing"
        results[label] = []

        model_for_limits = load_model(gradient_checkpointing=use_ckpt)
        max_positions, vocab_size = get_model_limits(model_for_limits)
        del model_for_limits
        gc.collect()
        torch.cuda.empty_cache()

        seq_lens = [seq for seq in requested_seq_lens if seq <= max_positions]
        skipped_seq_lens = [seq for seq in requested_seq_lens if seq > max_positions]

        limit_msg = f"  {label}: max_position_embeddings={max_positions}, vocab_size={vocab_size}"
        print(limit_msg)
        lines.append(limit_msg)
        if skipped_seq_lens:
            skip_msg = (
                f"  {label}: skipping unsupported seq_lens {skipped_seq_lens} "
                f"for {model_name} (limit={max_positions})"
            )
            print(skip_msg)
            lines.append(skip_msg)

        for seq in seq_lens:
            # Full cleanup before each measurement to avoid cross-contamination
            gc.collect()
            reset()

            model = load_model(gradient_checkpointing=use_ckpt)
            model.train()  # must be in train mode for checkpointing to activate

            # Baseline: memory used by model weights alone
            mem_after_load = torch.cuda.memory_allocated()

            # Create dummy input
            input_ids = torch.randint(0, vocab_size, (batch, seq), device=device)

            # ── Forward pass through transformer trunk only ────────────────
            # This avoids the logit tensor (~49 MB) which would mask savings.
            # Gradient checkpointing wraps each GPT2Block; without it, all
            # block-internal tensors (QKV, attn weights, FFN) are saved here.
            t0 = time.perf_counter()
            h = model.transformer(input_ids=input_ids).last_hidden_state
            # Scalar proxy loss — keeps output overhead negligible
            loss = h.float().mean()
            torch.cuda.synchronize()

            # Key measurement: live activation set BEFORE backward.
            # With checkpointing: only block input boundaries stored → low.
            # Without checkpointing: all intermediate tensors stored → high.
            mem_after_fwd = torch.cuda.memory_allocated()

            # ── Backward pass ─────────────────────────────────────────────
            loss.backward()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            # Peak across forward+backward (during backward, checkpointing
            # recomputes one block at a time — peak can approach no-ckpt level
            # temporarily, but steady-state is much lower).
            peak = torch.cuda.max_memory_allocated()

            fwd_overhead  = mem_after_fwd - mem_after_load
            peak_overhead = peak          - mem_after_load

            results[label].append({
                "seq":              seq,
                "after_load_mb":    mem_after_load / 1024**2,
                "fwd_overhead_mb":  fwd_overhead   / 1024**2,
                "peak_overhead_mb": peak_overhead  / 1024**2,
                "elapsed_s":        elapsed,
            })

            row = (
                f"  seq={seq:4d} | load={mb(mem_after_load):>10s} | "
                f"fwd_acts={mb(fwd_overhead):>10s} | "
                f"peak={mb(peak):>10s} | "
                f"t={elapsed:.2f}s"
            )
            print(row)
            lines.append(row)

            del model, input_ids, h, loss
            gc.collect()
            torch.cuda.empty_cache()

        print()

    # ── Summary: checkpointing savings in FORWARD activation memory ──────────
    print(f"\n{'─'*70}")
    print("  Gradient checkpointing: forward activation memory savings")
    print(f"  (measured at the key moment: after forward, before backward)")
    print(f"{'─'*70}")
    print(f"  {'seq':>6} | {'no_ckpt fwd':>12} | {'ckpt fwd':>12} | {'saved':>10} | {'pct':>6}")
    print(f"  {'─'*60}")
    for r_full, r_ckpt in zip(
        results["no checkpointing"],
        results["w/ gradient_checkpointing"],
    ):
        saved = r_full["fwd_overhead_mb"] - r_ckpt["fwd_overhead_mb"]
        pct   = saved / r_full["fwd_overhead_mb"] * 100 if r_full["fwd_overhead_mb"] > 0 else 0
        row = (
            f"  seq={r_full['seq']:4d} | "
            f"{r_full['fwd_overhead_mb']:>11.1f}M | "
            f"{r_ckpt['fwd_overhead_mb']:>11.1f}M | "
            f"{saved:>9.1f}M | "
            f"{pct:>5.0f}%"
        )
        print(row)
        lines.append(row)

    print(f"\n  NOTE: Peak memory (across fwd+bwd) shows smaller savings because")
    print(f"  checkpointing recomputes activations during backward, temporarily")
    print(f"  reaching the same peak. Forward-only activation memory is the right")
    print(f"  metric — this is what checkpointing actually reduces.")
    lines.append("  Forward activation memory savings demonstrate true checkpointing benefit.")

    return lines, results


# ════════════════════════════════════════════════════════════════════════════
# 3.  KV CACHE GROWTH SIMULATION
# ════════════════════════════════════════════════════════════════════════════

def simulate_kv_cache(smoke: bool = False) -> None:
    """
    Show how KV cache grows with sequence length for several model sizes.
    Saves a plot.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    configs = {
        "GPT-2 (124M)\nL=12, H=768":   (12, 768),
        "7B (LLaMA)\nL=32, H=4096":    (32, 4096),
        "13B (LLaMA)\nL=40, H=5120":   (40, 5120),
        "70B (LLaMA)\nL=80, H=8192":   (80, 8192),
    }
    seq_lens = np.arange(128, 4097, 128)
    batch    = 1
    dtype_bytes = 2  # FP16

    fig, ax = plt.subplots(figsize=(9, 5))

    for name, (L, H) in configs.items():
        # KV cache = 2 (K+V) × L × batch × seq × H × bytes
        kv_gb = 2 * L * batch * seq_lens * H * dtype_bytes / 1024**3
        ax.plot(seq_lens, kv_gb, marker=".", markersize=3, label=name)

    ax.axvline(2048, color="gray", linestyle="--", alpha=0.6, label="seq=2048")
    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_ylabel("KV Cache Size (GB)")
    ax.set_title("KV Cache Growth vs Sequence Length\n(batch=1, FP16, inference)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "day1_kv_cache.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\n  KV cache plot → {out_path}")

    # Print table at key seq lengths
    print(f"\n{'─'*60}")
    print("  KV Cache (GB) at key sequence lengths (batch=1, FP16)")
    print(f"{'─'*60}")
    header = f"  {'Model':<20} {'seq=512':>10} {'seq=2048':>10} {'seq=4096':>10}"
    print(header)
    for name, (L, H) in configs.items():
        short = name.split("\n")[0]
        vals = [
            2 * L * 1 * s * H * 2 / 1024**3
            for s in [512, 2048, 4096]
        ]
        print(f"  {short:<20} {vals[0]:>9.2f}G {vals[1]:>9.2f}G {vals[2]:>9.2f}G")


# ════════════════════════════════════════════════════════════════════════════
# 4.  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Quick run with reduced sizes")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Section 1: Theory ────────────────────────────────────────────────────
    theory_text = run_theoretical(smoke=args.smoke)

    # ── Section 2: Practical profiling ───────────────────────────────────────
    gpu_lines = []
    gpu_results = {}
    if HAS_TORCH and torch.cuda.is_available():
        gpu_lines, gpu_results = profile_gpt2(smoke=args.smoke)
    else:
        print("\n[INFO] No CUDA GPU found — skipping live profiling section.")

    # ── Section 3: KV cache ──────────────────────────────────────────────────
    simulate_kv_cache(smoke=args.smoke)

    # ── Write summary ────────────────────────────────────────────────────────
    summary_path = os.path.join(RESULTS_DIR, "day1_summary.txt")
    with open(summary_path, "w") as f:
        f.write("WHERE MEMORY ACTUALLY GOES IN TRANSFORMERS\n")
        f.write("=" * 56 + "\n\n")
        f.write("FORMULA SUMMARY\n")
        f.write("---------------\n")
        f.write("  Component         Bytes/param   Notes\n")
        f.write("  Parameters        2             FP16 weights\n")
        f.write("  Gradients         2             FP16 grads\n")
        f.write("  Adam optimizer    12            FP32: master + m + v\n")
        f.write("  ─────────────────────────────────────────\n")
        f.write("  Static total      16            ~16× weight size\n\n")
        f.write("  Activations       varies        O(B × S × H) per layer\n")
        f.write("  Attention matrix  varies        O(B × heads × S²) — the killer\n")
        f.write("  KV cache          varies        2 × L × B × S × H (inference)\n\n")
        f.write("KEY INSIGHT\n")
        f.write("-----------\n")
        f.write("  • Static memory (params+grads+optimizer) ≈ 16× weight size\n")
        f.write("  • For 7B model: 16 × 14 GB = ~112 GB minimum\n")
        f.write("  • Activations add another 10–100 GB depending on batch/seq\n")
        f.write("  • The O(S²) attention matrix dominates at long sequences\n")
        f.write("  • Gradient checkpointing trades 33% FLOPs for ~70% activation memory\n")
        f.write("  • KV cache grows linearly; at seq=4096 a 70B model needs ~16 GB\n\n")
        f.write("INTERVIEW ANSWERS\n")
        f.write("-----------------\n")
        f.write("Q: Why does memory scale with sequence length?\n")
        f.write("A: Self-attention materializes an S×S score matrix per head per layer.\n")
        f.write("   That's O(B × heads × S²) per layer, which becomes gigabytes at S=2048.\n\n")
        f.write("Q: What dominates memory at inference?\n")
        f.write("A: The KV cache. Each new token appends to K and V for every layer.\n")
        f.write("   With long context it easily rivals the model weights in size.\n\n")
        f.write("Q: How does KV cache work?\n")
        f.write("A: After the prefill (processing the prompt), each layer stores its\n")
        f.write("   K and V tensors. On each decode step we only compute Q for the new\n")
        f.write("   token and attend over the cached K/V — avoiding O(S) recomputation.\n")
        f.write("   Cost: memory grows as O(B × S × L × H).\n\n")
        if gpu_lines:
            f.write("GPU PROFILING RESULTS (GPT-2)\n")
            f.write("-" * 40 + "\n")
            f.write("\n".join(gpu_lines))
    print(f"\n  Summary → {summary_path}")
    print("\n  Day 1 complete. Key takeaway:")
    print("  Static memory = 16× weight size. Activations = O(S²) per layer.")
    print("  Gradient checkpointing is your lever for activation memory.")


if __name__ == "__main__":
    main()
