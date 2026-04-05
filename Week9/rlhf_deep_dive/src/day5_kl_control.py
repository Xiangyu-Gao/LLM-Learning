"""
Day 5 — KL Control & Instability
===================================
We run GRPO under four KL coefficient regimes and observe the failure modes:

  β = 0.00  — No KL penalty  → policy drifts freely, reward hacking risk
  β = 0.01  — Weak penalty   → some drift allowed, potential instability
  β = 0.10  — Balanced       → typical production value
  β = 1.00  — Strong penalty → stays near reference, slow reward improvement

For each β we track:
  - KL divergence from reference
  - Reward over time
  - Policy entropy (entropy collapse = mode collapse)
  - Token distribution shift (how far each token's probability moves)

The same task as Day 4: generate sequences that start with token 0.
Optimal policy puts P(x_0 = 0) = 1.0.  Reference policy starts uniform.

Key concepts
------------
  KL ≠ trust region: the clipped PPO objective is a *hard* trust region;
  the KL term is a *soft* penalty.  Both are needed for different failure modes.

  Entropy collapse: once π(a) → 1 for some a, exploration stops entirely.
  The policy can get "stuck" on a locally rewarded but globally suboptimal output.

Run
---
  python day5_kl_control.py
  python day5_kl_control.py --smoke
"""

import argparse, os, sys, copy
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import TinyRNNPolicy, kl_per_token, token_entropy, smooth

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "day5")
os.makedirs(RESULTS_DIR, exist_ok=True)

VOCAB_SIZE = 8
SEQ_LEN    = 4


# ─── Reward ───────────────────────────────────────────────────────────────────

def reward_fn(tokens: torch.Tensor) -> torch.Tensor:
    """+1 if sequence starts with token 0, else 0."""
    return (tokens[:, 0] == 0).float()


# ─── Token probability drift ──────────────────────────────────────────────────

@torch.no_grad()
def token0_prob_at_step0(policy: TinyRNNPolicy) -> float:
    """P(first token == 0) under current policy."""
    B = 512
    tokens, _, _ = policy.sample_sequence(B)
    return (tokens[:, 0] == 0).float().mean().item()


@torch.no_grad()
def max_token_prob(policy: TinyRNNPolicy) -> float:
    """Max probability assigned to any single token at step 0 (mode collapse indicator)."""
    B = 512
    tokens, _, _ = policy.sample_sequence(B)
    # Check first position
    counts = torch.zeros(VOCAB_SIZE)
    for t in range(VOCAB_SIZE):
        counts[t] = (tokens[:, 0] == t).float().sum()
    return (counts / B).max().item()


# ─── GRPO with configurable KL ────────────────────────────────────────────────

def run_experiment(kl_coef: float, n_steps: int, G: int, n_prompts: int) -> dict:
    """Run GRPO with a specific KL coefficient. Returns metric history."""
    policy     = TinyRNNPolicy(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN)
    ref_policy = copy.deepcopy(policy)
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

    history = {
        "reward":    [],
        "kl":        [],
        "entropy":   [],
        "p_correct": [],   # P(first token == 0)
        "mode_prob": [],   # max single-token prob (mode collapse indicator)
    }

    B   = n_prompts * G
    eps = 0.2

    for step in range(n_steps):
        # ── Rollout ───────────────────────────────────────────
        with torch.no_grad():
            tokens, lp_old, _ = policy.sample_sequence(B)
            rewards = reward_fn(tokens)

        # ── Group-normalised advantage ────────────────────────
        r_grouped   = rewards.view(n_prompts, G)
        r_mean      = r_grouped.mean(dim=1, keepdim=True)
        r_std       = r_grouped.std(dim=1,  keepdim=True) + 1e-8
        adv         = ((r_grouped - r_mean) / r_std).view(B).detach()

        # ── PPO clipped loss ───────────────────────────────────
        lp_new, _ = policy.log_probs_of(tokens)
        log_ratio  = (lp_new - lp_old.detach()).sum(dim=1)
        ratio      = log_ratio.exp()
        surr1      = ratio * adv
        surr2      = ratio.clamp(1 - eps, 1 + eps) * adv
        ppo_loss   = -torch.min(surr1, surr2).mean()

        # ── KL penalty ────────────────────────────────────────
        kl = kl_per_token(policy, ref_policy, tokens)
        loss = ppo_loss + kl_coef * kl

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        # ── Metrics ───────────────────────────────────────────
        history["reward"].append(rewards.mean().item())
        history["kl"].append(kl_per_token(policy, ref_policy, tokens))
        history["entropy"].append(token_entropy(policy))
        history["p_correct"].append(token0_prob_at_step0(policy))
        history["mode_prob"].append(max_token_prob(policy))

    return history


# ─── Visualise ────────────────────────────────────────────────────────────────

def plot_results(results: dict, n_steps: int):
    """
    results: {kl_coef_str: history_dict}
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Day 5 — KL Control & Instability", fontsize=13, fontweight="bold")

    kl_coefs = list(results.keys())
    colors   = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db"]
    w        = max(1, n_steps // 30)

    def _plot(ax, metric, title, ylabel, ylim=None, hline=None, hline_label=None):
        for i, (label, h) in enumerate(results.items()):
            ax.plot(smooth(h[metric], w), label=f"β={label}", color=colors[i], lw=2)
        if hline is not None:
            ax.axhline(hline, ls="--", c="gray", lw=1, label=hline_label)
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8)

    _plot(axes[0, 0], "reward",    "Reward over Training",
          "Mean Reward", ylim=(-0.05, 1.1), hline=1.0, hline_label="Optimal")
    _plot(axes[0, 1], "kl",        "KL(π_ref || π_θ)",
          "KL Divergence (nats)")
    _plot(axes[0, 2], "entropy",   "Policy Entropy",
          "Entropy (nats)", ylim=(0, None))
    _plot(axes[1, 0], "p_correct", "P(first token = 0)  [reward proxy]",
          "Probability", ylim=(-0.05, 1.1), hline=1.0, hline_label="Perfect")
    _plot(axes[1, 1], "mode_prob", "Max Single-Token Prob at Step 0\n(mode collapse indicator)",
          "Probability", ylim=(-0.05, 1.1),
          hline=1/VOCAB_SIZE, hline_label=f"Uniform = {1/VOCAB_SIZE:.2f}")

    # ── Final-state bar chart ─────────────────────────────────
    ax = axes[1, 2]
    bar_labels = list(results.keys())
    final_rew  = [np.mean(h["reward"][-20:])    for h in results.values()]
    final_kl   = [np.mean(h["kl"][-20:])        for h in results.values()]
    final_H    = [np.mean(h["entropy"][-20:])   for h in results.values()]

    x  = np.arange(len(bar_labels))
    w2 = 0.25
    ax.bar(x - w2, final_rew, w2, label="Reward",  color="#2ecc71", alpha=0.85)
    ax.bar(x,      final_kl,  w2, label="KL",      color="#e74c3c", alpha=0.85)
    ax.bar(x + w2, final_H,   w2, label="Entropy", color="#3498db", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"β={l}" for l in bar_labels], fontsize=9)
    ax.set_title("Final-State Summary (last 20 steps)")
    ax.set_ylabel("Value")
    ax.legend(fontsize=8)
    ax.set_ylim(0)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "day5_kl_control.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def print_failure_modes(results: dict):
    print("\n  Failure mode summary:")
    print(f"  {'β':>6}  {'reward':>8}  {'KL':>8}  {'entropy':>8}  {'diagnosis'}")
    print(f"  {'─'*65}")
    for label, h in results.items():
        r = np.mean(h["reward"][-20:])
        k = np.mean(h["kl"][-20:])
        e = np.mean(h["entropy"][-20:])
        if float(label) == 0.0:
            dx = "Unconstrained drift — reward hacking risk"
        elif k > 0.5:
            dx = "High KL — drifted far from reference"
        elif e < 0.5:
            dx = "Low entropy — mode collapse"
        elif r < 0.4:
            dx = "Reward too low — over-regularised"
        else:
            dx = "Stable ✓"
        print(f"  {label:>6}  {r:>8.3f}  {k:>8.4f}  {e:>8.4f}  {dx}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    n_steps   = 150  if args.smoke else 500
    G         = 4    if args.smoke else 8
    n_prompts = 8    if args.smoke else 16

    kl_values = [0.0, 0.01, 0.10, 1.00]

    print("=" * 55)
    print("  Day 5 — KL Control & Instability")
    print(f"  mode={'smoke' if args.smoke else 'full'}  n_steps={n_steps}")
    print(f"  KL coefficients: {kl_values}")
    print("=" * 55)

    results = {}
    for beta in kl_values:
        label = f"{beta:.2f}"
        print(f"\n  Running β={label} ...")
        h = run_experiment(beta, n_steps, G, n_prompts)
        results[label] = h
        print(f"    final reward={np.mean(h['reward'][-10:]):.3f}  "
              f"kl={np.mean(h['kl'][-10:]):.4f}  "
              f"H={np.mean(h['entropy'][-10:]):.3f}")

    print("\n  Plotting ...")
    plot_results(results, n_steps)
    print_failure_modes(results)
    print("\n  Day 5 complete.\n")


if __name__ == "__main__":
    main()
