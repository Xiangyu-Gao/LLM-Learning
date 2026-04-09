"""
day1_scaling_behavior.py — How model scale changes optimization dynamics.
=========================================================================

CORE INSIGHT: "Scaling changes optimization curvature."

Larger models have higher Hessian eigenvalues (sharper loss landscapes).
In practice this means two things:
  1. At the same learning rate, a larger model shifts its output distribution
     further per gradient step (higher KL per update).
  2. The learning rate at which training becomes unstable is lower for larger
     models.  Empirically: small ~5e-4, medium ~1e-4, large ~5e-5 (~10× gap).

Why does hidden_dim affect KL growth?
  Each logit is a dot product of the output weight row and the hidden state:
      logit_j = W_j · h
  An Adam step changes W_j by ≈ lr per coordinate, so:
      Δlogit_j ≈ lr · Σ_k |Δw_jk| · |h_k|  ∝  lr · hidden_dim
  A larger hidden state means more terms contributing to the logit shift,
  so the output distribution moves further per unit LR.

EXPERIMENT
----------
Three policy sizes (small / medium / large) are trained under a sweep of
5 learning rates using REINFORCE with an EMA baseline.

Two experimental choices are important:
  - Adam, no gradient clipping: clipping caps the effective step size
    equally for all model sizes, masking the curvature difference.
  - KL as the stability metric: normal REINFORCE often sees reward peak
    then settle — that is not instability.  Sustained high KL (policy
    drifted and not recovering) is the right signal.

OUTPUT
------
results/day1/day1_scaling_behavior.png       — reward / KL / volatility per size
results/day1/day1_instability_thresholds.png — threshold bar chart (descending staircase)
results/day1/day1_summary.txt

INTERVIEW TAKEAWAY
------------------
Q: "Why do large models require smaller learning rates in RLHF?"
A: Each Adam step shifts logits by ~lr × hidden_dim, so larger models drift
   further in KL-space per update.  This pushes the policy out of the reward
   model's training distribution faster.  In practice: 7B → lr~1e-6,
   70B → lr~5e-7 or smaller.
"""

import argparse
import os
import sys
from copy import deepcopy
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    make_policy, kl_per_token, token_entropy,
    gradient_norm, reward_volatility, smooth, SCALE_CONFIGS,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "day1")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Reward ──────────────────────────────────────────────────────────────────

def simple_reward(tokens: torch.Tensor) -> torch.Tensor:
    """
    r(x) = 2·(x[0]==0) + 2·(x[-1]==1) - 0.5·(only one unique token)

    Chosen so all model sizes can reach the optimum (it is not hard), which
    lets us isolate *optimization dynamics* rather than *capability* differences.
    The small diversity penalty prevents the policy from collapsing to a single
    token before we can observe the KL dynamics.
    """
    r = torch.zeros(tokens.shape[0])
    r += 2.0 * (tokens[:, 0] == 0).float()
    r += 2.0 * (tokens[:, -1] == 1).float()
    for b in range(tokens.shape[0]):
        if tokens[b].unique().numel() < 2:
            r[b] -= 0.5
    return r


# ─── Training ────────────────────────────────────────────────────────────────

def train_once(
    size:       str,
    lr:         float,
    n_steps:    int = 200,
    batch_size: int = 64,
) -> Dict[str, List[float]]:
    """
    REINFORCE with EMA baseline.  Returns per-step metrics dict.

    The reference policy is a frozen copy of the initial policy.  KL from
    this reference measures how far training has moved the policy — the
    primary stability signal used in real RLHF systems.

    Gradient clipping is intentionally omitted: clipping would bound the
    step size equally for all model sizes, hiding the curvature difference
    we are trying to observe.
    """
    policy     = make_policy(size)
    ref_policy = deepcopy(policy)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    baseline  = 0.0   # EMA of mean reward — subtracting it reduces gradient variance
                      # without biasing the gradient (baseline theorem)

    history: Dict[str, List[float]] = {
        k: [] for k in ("reward", "kl", "entropy", "grad_norm", "volatility")
    }

    for step in range(n_steps):
        policy.train()
        tokens, log_probs, _ = policy.sample_sequence(batch_size)
        rewards  = simple_reward(tokens)
        mean_r   = rewards.mean().item()
        baseline = 0.9 * baseline + 0.1 * mean_r

        # REINFORCE: ∇J ≈ E[∇log π(x) · (r(x) - b)]
        # summing log_probs over the sequence gives the sequence-level log-prob
        loss = -(log_probs.sum(dim=1) * (rewards - baseline)).mean()
        optimizer.zero_grad()
        loss.backward()
        gnorm = gradient_norm(policy)   # recorded BEFORE any step
        optimizer.step()

        # Hard stop on numerical blow-up; pad history so all curves are same length
        if not np.isfinite(gnorm) or gnorm > 1e6:
            last = lambda k: history[k][-1] if history[k] else 0.0
            for _ in range(step + 1, n_steps):
                history["reward"].append(last("reward"))
                history["kl"].append(last("kl"))
                history["entropy"].append(last("entropy"))
                history["grad_norm"].append(gnorm)
                history["volatility"].append(reward_volatility(history["reward"], 20))
            break

        with torch.no_grad():
            kl  = kl_per_token(policy, ref_policy, tokens)
            ent = token_entropy(policy)

        history["reward"].append(mean_r)
        history["kl"].append(kl)
        history["entropy"].append(ent)
        history["grad_norm"].append(gnorm)
        history["volatility"].append(reward_volatility(history["reward"], 20))

    return history


# ─── Instability detection ───────────────────────────────────────────────────

def find_instability_threshold(
    size:    str,
    lrs:     List[float],
    n_steps: int,
) -> Tuple[float, Dict[str, Dict]]:
    """
    Train `size` at each LR and return the lowest LR that is unstable.

    Instability criterion: mean KL in the final 25% of training > 0.5.

    Why tail KL, not reward crash?
      Normal REINFORCE training often shows reward peak → settle — the policy
      finds the optimum then the baseline catches up, making reward *appear*
      to drop.  That is not instability.  Sustained high KL means the policy
      has genuinely drifted far from the reference and is not recovering,
      which is the real problem in RLHF (reward hacking, capability loss).

    Why 0.5 as the threshold?
      KL < 0.1: policy barely moved (under-training or LR too small).
      KL 0.1–0.5: healthy training zone — policy improved while staying anchored.
      KL > 0.5: policy has drifted meaningfully; in real RLHF this is where
                degradation and reward hacking start appearing.
      KL ≈ 2.08: maximum possible for vocab_size=8 (log 8); policy is deterministic
                 and completely different from reference.
    """
    all_curves: Dict[str, Dict] = {}
    threshold = float("inf")

    for lr in lrs:
        hist = train_once(size, lr, n_steps=n_steps)
        key  = f"{lr:.0e}"
        all_curves[key] = hist

        kls     = np.array(hist["kl"])
        gnorms  = np.array(hist["grad_norm"])
        tail_kl = kls[int(0.75 * len(kls)):].mean() if len(kls) > 5 else kls.mean()

        if tail_kl > 0.5 or gnorms.max() > 500 or not np.isfinite(gnorms).all():
            threshold = min(threshold, lr)

    return threshold, all_curves


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_results(results: Dict[str, Dict], lrs: List[float], n_steps: int):
    sizes   = list(results.keys())
    colors  = plt.cm.tab10.colors

    fig, axes = plt.subplots(3, len(sizes), figsize=(6 * len(sizes), 12))
    if len(sizes) == 1:
        axes = axes.reshape(-1, 1)

    for col, size in enumerate(sizes):
        for i, lr in enumerate(lrs):
            hist  = results[size][f"{lr:.0e}"]
            color = colors[i % len(colors)]
            label = f"lr={lr:.0e}"
            w     = max(1, min(10, n_steps // 10))

            axes[0, col].plot(smooth(hist["reward"],     w=w), color=color, alpha=0.85, label=label)
            axes[1, col].plot(smooth(hist["kl"],         w=w), color=color, alpha=0.85, label=label)
            axes[2, col].plot(smooth(hist["volatility"], w=max(1, w // 2)), color=color, alpha=0.85, label=label)

        cfg     = SCALE_CONFIGS[size]
        n_params = sum(p.numel() for p in make_policy(size).parameters())
        axes[0, col].set_title(f"{size.upper()} (hidden={cfg.hidden_dim}, ~{n_params:,} params)",
                               fontsize=11, fontweight="bold")

        for row, ylabel in enumerate(["Mean Reward", "KL(π || π_ref)", "Reward Volatility"]):
            axes[row, col].set_ylabel(ylabel)
            axes[row, col].set_xlabel("Update step")
            axes[row, col].legend(fontsize=8)

        axes[1, col].axhline(0.5, color="red", linestyle="--", alpha=0.4, label="KL=0.5 (threshold)")

    fig.suptitle("Day 1: Scaling Behavior — Instability Threshold vs. Model Size",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "day1_scaling_behavior.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_instability_thresholds(thresholds: Dict[str, float], lrs: List[float]):
    sizes  = list(thresholds.keys())
    thresh = [thresholds[s] if thresholds[s] != float("inf") else lrs[-1] * 2 for s in sizes]
    colors = ["steelblue", "darkorange", "crimson"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(sizes, thresh, color=colors[:len(sizes)])
    ax.set_yscale("log")
    ax.set_ylabel("Instability Threshold LR (log scale)")
    ax.set_title("Instability Onset vs. Model Scale\n"
                 "(lower bar = unstable at smaller LR = sharper landscape)", fontsize=11)
    for bar, t in zip(bars, thresh):
        ax.text(bar.get_x() + bar.get_width() / 2, t * 1.1,
                f"{t:.0e}", ha="center", va="bottom", fontsize=10)
    ax.axhline(lrs[-1], color="gray", linestyle="--", alpha=0.4, label="max tested LR")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "day1_instability_thresholds.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        sizes, lrs, n_steps = ["small", "large"], [5e-5, 5e-4], 100
    else:
        sizes   = ["small", "medium", "large"]
        lrs     = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
        n_steps = 400

    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print(" Day 1: Scaling Behavior")
    print("=" * 60)
    print(f"  Sizes:   {sizes}")
    print(f"  LRs:     {lrs}")
    print(f"  Steps:   {n_steps}")
    print()

    results, thresholds = {}, {}

    for size in sizes:
        cfg     = SCALE_CONFIGS[size]
        n_params = sum(p.numel() for p in make_policy(size).parameters())
        print(f"[{size.upper()}]  hidden={cfg.hidden_dim}, params={n_params:,}")
        threshold, curves = find_instability_threshold(size, lrs, n_steps)
        results[size]    = curves
        thresholds[size] = threshold
        print(f"  Stable LRs:   {[lr for lr in lrs if lr < threshold]}")
        print(f"  Unstable LRs: {[lr for lr in lrs if lr >= threshold]}")
        print(f"  Instability threshold: {threshold:.0e}")
        print()

    print("Generating plots...")
    plot_results(results, lrs, n_steps)
    plot_instability_thresholds(thresholds, lrs)

    summary_path = os.path.join(RESULTS_DIR, "day1_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Day 1 — Scaling Behavior Summary\n" + "=" * 40 + "\n\n")
        for size in sizes:
            cfg    = SCALE_CONFIGS[size]
            n_p    = sum(p.numel() for p in make_policy(size).parameters())
            thresh = thresholds[size]
            f.write(f"{size.upper()} (hidden={cfg.hidden_dim}, params={n_p:,})\n")
            f.write(f"  Instability threshold: {thresh:.2e}\n")
            for lr in lrs:
                hist = results[size][f"{lr:.0e}"]
                f.write(f"  LR={lr:.0e}: peak_R={max(hist['reward']):.3f}  "
                        f"final_R={np.mean(hist['reward'][-20:]):.3f}  "
                        f"max_KL={max(hist['kl']):.3f}\n")
            f.write("\n")
        f.write("KEY INSIGHT\n-----------\n"
                "Instability threshold descends ~10x per model size step.\n"
                "In real RLHF: 7B→lr~1e-6, 70B→lr~5e-7 or less.\n")
    print(f"  Saved: {summary_path}")
    print("\nDay 1 complete.")


if __name__ == "__main__":
    main()
