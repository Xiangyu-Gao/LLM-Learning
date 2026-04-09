"""
day3_kl_vs_entropy.py — KL Penalty vs Entropy Bonus: What Each Does.
=====================================================================

CORE INSIGHT
------------
KL and entropy serve different but complementary purposes in RLHF:

  KL penalty   → keeps you *near the reference model* (anchor)
  Entropy bonus → keeps the policy *diverse* (exploration)

They solve different failure modes:
  Without KL    → policy drifts far from reference → reward hacking,
                  capability degradation, gibberish output
  Without entropy → policy collapses to the single highest-reward
                    sequence (mode collapse / entropy death)

THE RLHF OBJECTIVE (with both):
  J(θ) = E[r(x)] - β·KL(π_θ||π_ref) + α·H(π_θ)

  where KL(π_θ||π_ref) = E_{x~π_θ}[log π_θ(x) - log π_ref(x)]
  and   H(π_θ)         = -E_{x~π_θ}[log π_θ(x)]

Note: the entropy bonus -α·E[log π] is actually equivalent to reducing
the KL coefficient by α when α ≤ β.  The formula simplifies to:
  J(θ) = E[r(x)] - (β-α)·E[log π_θ/π_ref] - β·E[log π_ref]
The last term is constant w.r.t. θ, so entropy bonus ≡ reduced KL.
But they're conceptually distinct and controlled separately in practice.

COEFFICIENT DESIGN
------------------
The key constraint: β and α must NOT be equal.
  H(π) = -E[log π]  and  KL = E[log π] - E[log π_ref]
So entropy bonus = -α·E[log π] is equivalent to a KL penalty of coefficient α
(the E[log π_ref] term is constant w.r.t. θ).  If β ≈ α, ENT_ONLY ≈ KL_ONLY.
We use β=0.3 and α=0.2 (different enough to see distinct behaviour).

EXPERIMENT
----------
Four conditions trained for N steps:
  NONE    : no regularization (reward signal only)
  KL_ONLY : only KL penalty (β=0.3, α=0)
  ENT_ONLY: only entropy bonus (β=0, α=0.05)
  BOTH    : KL + entropy (β=0.1, α=0.05)

Tracked metrics: reward, KL from reference, token entropy.

PREDICTED OUTCOMES
------------------
  NONE    : reward high, entropy → 0 (entropy death), KL → 2.0 (explodes)
  KL_ONLY : reward moderate (β slows learning), KL suppressed, entropy
            indirectly protected via proximity to high-entropy reference
  ENT_ONLY: reward high (α=0.2 allows learning), entropy maintained (>1.0),
            KL drifts MORE than KL_ONLY (no explicit anchor to reference)
  BOTH    : reward slightly lower (two constraints), KL controlled,
            entropy maintained — best balance

OUTPUT
------
results/day3/day3_kl_vs_entropy.png  — 3-metric × 4-condition comparison
results/day3/day3_tradeoffs.png       — Phase portrait: KL vs entropy space

INTERVIEW TAKEAWAYS
-------------------
Q: "Difference between KL regularization and entropy bonus?"
A: KL measures distance from the reference model — it's about *where* the
   policy is in parameter space.  Entropy measures *how spread out* the
   policy distribution is — it's about diversity.  KL=0 means you're the
   reference model; high entropy means you explore widely.  They're related
   (entropy is the diagonal of KL) but answer different questions.

Q: "Why not just use entropy?"
A: Entropy alone doesn't anchor the policy.  A high-entropy policy near the
   reference is good.  A high-entropy policy far from the reference is bad
   (it's exploring, but in the wrong part of space).  KL tells you *where*
   you are; entropy tells you *how wide* you are.  You need both.
"""

import argparse
import os
import sys
from copy import deepcopy
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    TinyRNNPolicy, make_policy, kl_per_token, token_entropy,
    gradient_norm, smooth,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "day3")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Reward ──────────────────────────────────────────────────────────────────

def shaped_reward(tokens: torch.Tensor) -> torch.Tensor:
    r = torch.zeros(tokens.shape[0])
    r += 2.0 * (tokens[:, 0] == 0).float()
    r += 2.0 * (tokens[:, -1] == 1).float()
    return r


# ─── Training loop ───────────────────────────────────────────────────────────

def train(
    kl_beta:   float,
    ent_alpha: float,
    n_steps:   int,
    batch_size: int,
    lr:        float,
    label:     str,
) -> Dict[str, List[float]]:
    """
    Policy gradient with optional KL penalty and entropy bonus.

    Objective: J = E[r(x)] - β·KL(π||π_ref) + α·H(π)

    The gradient is:
        ∇J = E[∇log π · r(x)]   (reward signal)
           - β · ∇KL              (KL penalty gradient)
           + α · ∇H               (entropy bonus gradient)

    We implement this by augmenting the per-token reward:
        r_eff(x_t) = r_t - β·(log π_θ(x_t) - log π_ref(x_t)) - α·log π_θ(x_t)

    For sequence-level reward (r only at end):
        r_eff = [0, 0, ..., r_final] - β·kl_per_token - α·log_probs_per_token
    """
    policy     = make_policy("small")
    ref_policy = deepcopy(policy)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    history: Dict[str, List[float]] = {
        k: [] for k in ("reward", "kl", "entropy", "eff_reward")
    }

    for step in range(n_steps):
        policy.train()
        tokens, log_probs, _ = policy.sample_sequence(batch_size)
        rewards = shaped_reward(tokens)                 # (B,) — sequence reward

        # Broadcast reward to per-token (last token gets the reward)
        step_rewards = torch.zeros_like(log_probs)      # (B, T)
        step_rewards[:, -1] = rewards

        # KL penalty per token
        with torch.no_grad():
            ref_lp, _ = ref_policy.log_probs_of(tokens)

        kl_per_tok = log_probs - ref_lp                  # (B, T)  log π - log π_ref

        # Entropy bonus per token: -log π_θ (because H = -E[log π])
        # Gradient of H = -∇log π → treated as +α·(-log π) bonus
        ent_bonus = -log_probs                           # (B, T)

        # Effective per-token reward
        r_eff = step_rewards - kl_beta * kl_per_tok + ent_alpha * ent_bonus

        # Sequence-level advantage (use mean of effective rewards as baseline)
        seq_r_eff   = r_eff.sum(dim=1)                  # (B,)
        baseline    = seq_r_eff.mean().detach()
        advantages  = seq_r_eff - baseline

        # REINFORCE loss
        seq_lp = log_probs.sum(dim=1)
        loss   = -(seq_lp * advantages).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        # Diagnostics
        with torch.no_grad():
            kl_val  = kl_per_token(policy, ref_policy, tokens)
            ent_val = token_entropy(policy)
            r_val   = rewards.mean().item()
            reff_val = r_eff.sum(dim=1).mean().item()

        history["reward"].append(r_val)
        history["kl"].append(kl_val)
        history["entropy"].append(ent_val)
        history["eff_reward"].append(reff_val)

    print(f"  [{label}] final_r={np.mean(history['reward'][-20:]):.3f}  "
          f"final_kl={np.mean(history['kl'][-20:]):.3f}  "
          f"final_ent={np.mean(history['entropy'][-20:]):.3f}")
    return history


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_comparison(
    all_histories: Dict[str, Dict[str, List]],
    n_steps:       int,
):
    conditions = list(all_histories.keys())
    colors     = {
        "NONE":     "crimson",
        "KL_ONLY":  "steelblue",
        "ENT_ONLY": "darkorange",
        "BOTH":     "seagreen",
    }
    labels = {
        "NONE":     "No regularization",
        "KL_ONLY":  "KL penalty only (β=0.1)",
        "ENT_ONLY": "Entropy bonus only (α=0.05)",
        "BOTH":     "KL + Entropy (β=0.1, α=0.05)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    w = min(15, n_steps // 10)

    for cond in conditions:
        hist  = all_histories[cond]
        color = colors[cond]
        label = labels[cond]
        axes[0].plot(smooth(hist["reward"],  w=w), color=color, label=label, alpha=0.85)
        axes[1].plot(smooth(hist["kl"],      w=w), color=color, label=label, alpha=0.85)
        axes[2].plot(smooth(hist["entropy"], w=w), color=color, label=label, alpha=0.85)

    axes[0].set_title("Reward")
    axes[0].set_ylabel("Mean Reward"); axes[0].set_xlabel("Step")
    axes[0].legend(fontsize=8)

    axes[1].set_title("KL Divergence from Reference")
    axes[1].set_ylabel("KL(π_θ || π_ref)"); axes[1].set_xlabel("Step")
    axes[1].axhline(1.0, color="red", linestyle="--", alpha=0.4, label="KL=1")
    axes[1].legend(fontsize=8)

    axes[2].set_title("Token Entropy")
    axes[2].set_ylabel("H(π_θ) (nats)"); axes[2].set_xlabel("Step")
    axes[2].legend(fontsize=8)

    fig.suptitle(
        "Day 3: KL Penalty vs Entropy Bonus — What Each Term Does",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "day3_kl_vs_entropy.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_phase_portrait(all_histories: Dict[str, Dict[str, List]]):
    """
    2D phase portrait: KL (x-axis) vs Entropy (y-axis).

    Shows the trajectory each condition takes through (KL, H) space.
    Good region: low KL (close to reference), high entropy (diverse).
    Bad region: high KL (far from reference) or low entropy (collapsed).
    """
    colors = {
        "NONE":     "crimson",
        "KL_ONLY":  "steelblue",
        "ENT_ONLY": "darkorange",
        "BOTH":     "seagreen",
    }
    labels = {
        "NONE":     "No reg",
        "KL_ONLY":  "KL only",
        "ENT_ONLY": "Entropy only",
        "BOTH":     "KL+Entropy",
    }

    fig, ax = plt.subplots(figsize=(7, 6))

    for cond, hist in all_histories.items():
        kl  = np.array(hist["kl"])
        ent = np.array(hist["entropy"])
        # Plot trajectory (thin line) + endpoint (large dot)
        ax.plot(kl, ent, color=colors[cond], alpha=0.35, linewidth=1)
        ax.scatter(kl[0],  ent[0],  color=colors[cond], marker="o", s=80,
                   zorder=5, alpha=0.5)
        ax.scatter(kl[-1], ent[-1], color=colors[cond], marker="*", s=200,
                   zorder=6, label=labels[cond])

    ax.axvline(1.0, color="red",  linestyle="--", alpha=0.3, label="KL=1 (caution)")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3, label="H=0.5 (low diversity)")

    ax.set_xlabel("KL Divergence from Reference →  (higher = further from reference)")
    ax.set_ylabel("Token Entropy →  (lower = more collapsed)")
    ax.set_title("Phase Portrait: KL vs Entropy Trajectory\n"
                 "★ = final state,  ○ = start")
    ax.legend(fontsize=9)

    # Shade the "good" region
    ax.fill_between([0, 1.0], [0.5, 0.5], [3.0, 3.0], alpha=0.05, color="seagreen",
                    label="Target region (low KL, high entropy)")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "day3_phase_portrait.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        n_steps    = 80
        batch_size = 32
    else:
        n_steps    = 600
        batch_size = 64

    torch.manual_seed(42)
    np.random.seed(42)
    lr = 1e-3

    print("=" * 60)
    print(" Day 3: KL vs Entropy Regularization")
    print("=" * 60)
    print()

    # β and α must differ meaningfully.
    # β=α means entropy bonus ≡ KL penalty (they cancel to the same gradient).
    # β=0.3 suppresses KL without full mode collapse; α=0.2 clearly distinguishes
    # ENT_ONLY from KL_ONLY in the KL trajectory.
    configs = {
        "NONE":     {"kl_beta": 0.0,  "ent_alpha": 0.0},
        "KL_ONLY":  {"kl_beta": 0.3,  "ent_alpha": 0.0},
        "ENT_ONLY": {"kl_beta": 0.0,  "ent_alpha": 0.2},
        "BOTH":     {"kl_beta": 0.3,  "ent_alpha": 0.1},
    }

    all_histories = {}
    for label, cfg in configs.items():
        print(f"Training [{label}]  β={cfg['kl_beta']}  α={cfg['ent_alpha']}...")
        hist = train(
            kl_beta   = cfg["kl_beta"],
            ent_alpha = cfg["ent_alpha"],
            n_steps   = n_steps,
            batch_size = batch_size,
            lr        = lr,
            label     = label,
        )
        all_histories[label] = hist

    print()
    print("Generating plots...")
    plot_comparison(all_histories, n_steps)
    plot_phase_portrait(all_histories)

    print()
    print("INTERPRETATION:")
    print(f"  {'Condition':12s}  {'reward':>7}  {'KL':>6}  {'entropy':>8}  Status")
    print(f"  {'-'*12}  {'-'*7}  {'-'*6}  {'-'*8}  ------")
    for label, hist in all_histories.items():
        final_r   = np.mean(hist["reward"][-30:])
        final_kl  = np.mean(hist["kl"][-30:])
        final_ent = np.mean(hist["entropy"][-30:])

        if final_ent < 0.3:
            status = "ENTROPY DEATH (no entropy reg)"
        elif final_kl > 1.5:
            status = "KL EXPLOSION (no KL penalty)"
        elif final_r < 1.5 and final_kl < 0.3:
            status = "MODE COLLAPSE (beta too high)"
        elif final_kl > 0.7 and final_ent > 0.8:
            status = "KL drifts, entropy maintained"
        elif final_kl < 0.5 and final_ent > 0.8:
            status = "KL anchored, entropy maintained"
        else:
            status = "OK"
        print(f"  {label:12s}  {final_r:7.3f}  {final_kl:6.3f}  {final_ent:8.3f}  {status}")

    print()
    print("Day 3 complete.")


if __name__ == "__main__":
    main()
