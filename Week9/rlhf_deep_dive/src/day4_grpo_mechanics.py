"""
Day 4 — GRPO Mechanics
========================
GRPO (Group Relative Policy Optimisation) replaces the PPO value function
with a *group-normalised* advantage, computed from G parallel rollouts
of the same prompt.

Algorithm (one outer step)
--------------------------
  For each prompt p in batch:
    1. Sample G sequences: x_1, ..., x_G  ~ π_θ(· | p)
    2. Score each with reward function r(x_i | p)
    3. Normalise within group:
         A_i = (r_i − mean_{g}(r_g)) / (std_{g}(r_g) + ε)
    4. PPO clipped update using A_i as advantage (no critic needed)
    5. KL penalty to reference policy (same as PPO-RL)

Why it works
------------
  • The group mean serves as the *baseline* (reduces variance without bias).
  • No learned value function → simpler, fewer hyperparameters.
  • Off-policy stable because we use importance sampling (ratio r_t).

Experiment
----------
  Task: generate a 4-token sequence that starts with token 0.
  Reward: +1 if tokens[0] == 0, else 0.
  Group size G = 8.

We visualise:
  - Advantage distribution (should be bimodal: good/bad within group)
  - KL drift from reference
  - Reward improvement
  - Comparison to PPO (same task, different algorithm)

Run
---
  python day4_grpo_mechanics.py
  python day4_grpo_mechanics.py --smoke
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "day4")
os.makedirs(RESULTS_DIR, exist_ok=True)

VOCAB_SIZE = 8
SEQ_LEN    = 4


# ─── Reward ───────────────────────────────────────────────────────────────────

def starts_with_zero(tokens: torch.Tensor) -> torch.Tensor:
    """tokens: (B, T).  Reward: +1 if tokens[:,0] == 0 else 0."""
    return (tokens[:, 0] == 0).float()


# ─── GRPO ─────────────────────────────────────────────────────────────────────

def grpo_step(
    policy:     TinyRNNPolicy,
    ref_policy: TinyRNNPolicy,
    reward_fn,
    opt:        torch.optim.Optimizer,
    n_prompts:  int,
    G:          int,
    eps:        float = 0.2,
    kl_coef:    float = 0.04,
):
    """
    One GRPO outer step.

    n_prompts: number of (virtual) prompts in the batch.
    G:         group size — how many sequences to sample per prompt.
    Total batch = n_prompts * G sequences.
    """
    B = n_prompts * G

    # ── 1. Collect group rollouts ─────────────────────────────
    with torch.no_grad():
        tokens_all, lp_old_all, _ = policy.sample_sequence(B)
        rewards = reward_fn(tokens_all)  # (B,)

    # ── 2. Group-normalise advantages ─────────────────────────
    # Reshape to (n_prompts, G) for within-group statistics
    r_grouped = rewards.view(n_prompts, G)
    r_mean    = r_grouped.mean(dim=1, keepdim=True)   # (n_prompts, 1)
    r_std     = r_grouped.std(dim=1,  keepdim=True) + 1e-8
    adv_grouped = (r_grouped - r_mean) / r_std       # (n_prompts, G)
    advantages  = adv_grouped.view(B)                # (B,) — flat

    # ── 3. Policy gradient loss ───────────────────────────────
    lp_new, _ = policy.log_probs_of(tokens_all)    # (B, T)
    lp_old_all = lp_old_all.detach()

    # Sequence-level log-ratio (sum over T).
    # NOTE: with a single update per rollout (no epoch loop), lp_new == lp_old_all
    # at this point, so ratio == 1 always and the clip below never activates.
    # The clipped form is kept intentionally to (a) show the PPO connection and
    # (b) make it trivial to add multi-epoch updates later by wrapping lines
    # 102-111 in a for-loop, at which point the ratio will diverge from 1.
    log_ratio = (lp_new - lp_old_all).sum(dim=1)   # (B,)
    ratio     = log_ratio.exp()

    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - eps, 1 + eps) * advantages
    ppo_loss = -torch.min(surr1, surr2).mean()

    # ── 4. KL penalty ─────────────────────────────────────────
    kl = kl_per_token(policy, ref_policy, tokens_all)
    total_loss = ppo_loss + kl_coef * kl

    opt.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()

    return {
        "reward":      rewards.mean().item(),
        "advantage":   advantages.detach().tolist(),  # for distribution plot
        "kl":          kl,
        "entropy":     token_entropy(policy),
        "clip_frac":   ((ratio < 1 - eps) | (ratio > 1 + eps)).float().mean().item(),
    }


# ─── PPO (for comparison) ─────────────────────────────────────────────────────

def ppo_step_simple(
    policy:     TinyRNNPolicy,
    ref_policy: TinyRNNPolicy,
    reward_fn,
    opt,
    batch_size: int,
    eps:        float = 0.2,
    kl_coef:    float = 0.04,
    n_epochs:   int   = 2,
):
    """Simple PPO without a value function (uses mean-reward baseline instead)."""
    with torch.no_grad():
        tokens, lp_old, _ = policy.sample_sequence(batch_size)
        rewards = reward_fn(tokens)

    # Baseline = batch mean
    advantages = rewards - rewards.mean()
    if advantages.std() > 1e-6:
        advantages = advantages / (advantages.std() + 1e-8)
    advantages = advantages.detach()

    for _ in range(n_epochs):
        lp_new, _ = policy.log_probs_of(tokens)
        log_ratio  = (lp_new - lp_old.detach()).sum(dim=1)
        ratio      = log_ratio.exp()
        surr1      = ratio * advantages
        surr2      = ratio.clamp(1 - eps, 1 + eps) * advantages
        kl         = kl_per_token(policy, ref_policy, tokens)
        loss       = -torch.min(surr1, surr2).mean() + kl_coef * kl
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

    return {
        "reward":  rewards.mean().item(),
        "kl":      kl_per_token(policy, ref_policy, tokens),
        "entropy": token_entropy(policy),
    }


# ─── Training loops ───────────────────────────────────────────────────────────

def run_grpo(n_steps: int, G: int, n_prompts: int, kl_coef: float):
    policy     = TinyRNNPolicy(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN)
    ref_policy = copy.deepcopy(policy)
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

    history = {"reward": [], "kl": [], "entropy": [], "adv_snapshots": [], "clip_frac": []}
    snap_steps = set(range(0, n_steps, max(1, n_steps // 5)))

    for step in range(n_steps):
        m = grpo_step(policy, ref_policy, starts_with_zero, opt, n_prompts, G, kl_coef=kl_coef)
        history["reward"].append(m["reward"])
        history["kl"].append(m["kl"])
        history["entropy"].append(m["entropy"])
        history["clip_frac"].append(m["clip_frac"])
        if step in snap_steps:
            history["adv_snapshots"].append((step, m["advantage"]))

        if (step + 1) % max(1, n_steps // 4) == 0:
            print(f"    GRPO step {step+1:3d}/{n_steps}  "
                  f"reward={m['reward']:.3f}  kl={m['kl']:.4f}  H={m['entropy']:.3f}")

    return history, policy


def run_ppo_compare(n_steps: int, batch_size: int, kl_coef: float):
    policy     = TinyRNNPolicy(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN)
    ref_policy = copy.deepcopy(policy)
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

    history = {"reward": [], "kl": [], "entropy": []}
    for step in range(n_steps):
        m = ppo_step_simple(policy, ref_policy, starts_with_zero, opt, batch_size, kl_coef=kl_coef)
        history["reward"].append(m["reward"])
        history["kl"].append(m["kl"])
        history["entropy"].append(m["entropy"])

    return history


# ─── Visualise ────────────────────────────────────────────────────────────────

def plot_results(grpo_hist, ppo_hist, n_steps: int, G: int):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Day 4 — GRPO Mechanics (Group size G={G})", fontsize=13, fontweight="bold")

    w = max(1, n_steps // 30)

    # ── (0,0) Reward: GRPO vs PPO ─────────────────────────────
    ax = axes[0, 0]
    ax.plot(smooth(grpo_hist["reward"], w), label=f"GRPO (G={G})", color="#2ecc71", lw=2)
    ax.plot(smooth(ppo_hist["reward"],  w), label="PPO (no critic)", color="#3498db", lw=2)
    ax.axhline(1.0, ls="--", c="gray", lw=1, label="Optimal reward=1")
    ax.set_title("Reward: GRPO vs PPO")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Reward")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)

    # ── (0,1) KL divergence ───────────────────────────────────
    ax = axes[0, 1]
    ax.plot(smooth(grpo_hist["kl"], w), label="GRPO", color="#2ecc71", lw=2)
    ax.plot(smooth(ppo_hist["kl"],  w), label="PPO",  color="#3498db", lw=2)
    ax.set_title("KL(π_ref || π_θ)")
    ax.set_xlabel("Step")
    ax.set_ylabel("KL Divergence (nats)")
    ax.legend(fontsize=8)

    # ── (0,2) Entropy ─────────────────────────────────────────
    ax = axes[0, 2]
    ax.plot(smooth(grpo_hist["entropy"], w), label="GRPO", color="#2ecc71", lw=2)
    ax.plot(smooth(ppo_hist["entropy"],  w), label="PPO",  color="#3498db", lw=2)
    ax.set_title("Policy Entropy")
    ax.set_xlabel("Step")
    ax.set_ylabel("Entropy (nats)")
    ax.legend(fontsize=8)

    # ── (1,0) Advantage distributions at snapshots ─────────────
    ax = axes[1, 0]
    cmap = plt.cm.viridis
    snaps = grpo_hist["adv_snapshots"]
    for i, (step_idx, advs) in enumerate(snaps):
        color = cmap(i / max(1, len(snaps) - 1))
        ax.hist(advs, bins=20, alpha=0.5, color=color,
                label=f"step {step_idx}", density=True)
    ax.axvline(0, c="black", lw=0.8)
    ax.set_title("Group-Normalised Advantage Distribution\n(should be zero-mean by construction)")
    ax.set_xlabel("Advantage A_i")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)

    # ── (1,1) Clipping frequency ─────────────────────────────
    ax = axes[1, 1]
    ax.plot(smooth(grpo_hist["clip_frac"], w), color="darkorange")
    ax.set_title("GRPO — Clipping Frequency")
    ax.set_xlabel("Step")
    ax.set_ylabel("Fraction Clipped")
    ax.set_ylim(0, 1)
    ax.axhline(0.0, ls="--", c="gray", lw=1, label="Converged")
    ax.legend(fontsize=8)

    # ── (1,2) Key equations ──────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    txt = (
        "GRPO Algorithm\n"
        "────────────────────────────────────\n"
        f"Group size G = {G}\n\n"
        "For each prompt p:\n"
        "  1. Sample x_1,...,x_G ~ π_θ(·|p)\n"
        "  2. Score: r_i = R(x_i, p)\n"
        "  3. Normalise within group:\n"
        "       A_i = (r_i − μ_g) / (σ_g + ε)\n"
        "  4. PPO clipped update on A_i\n"
        "  5. KL penalty: β·KL(π_ref||π_θ)\n\n"
        "Key difference from PPO:\n"
        "  • No value function V_φ needed\n"
        "  • Group mean serves as baseline\n"
        "  • Works with sparse rewards\n"
        "  • Off-policy via importance ratio"
    )
    ax.text(0.05, 0.97, txt, transform=ax.transAxes,
            fontsize=9.5, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fffbe6", alpha=0.9))

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "day4_grpo_mechanics.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    n_steps   = 150  if args.smoke else 500
    G         = 4    if args.smoke else 8
    n_prompts = 8    if args.smoke else 16
    kl_coef   = 0.04

    print("=" * 55)
    print("  Day 4 — GRPO Mechanics")
    print(f"  mode={'smoke' if args.smoke else 'full'}  G={G}  n_prompts={n_prompts}")
    print("=" * 55)

    print(f"\n[1/2] GRPO (G={G}) ...")
    grpo_hist, grpo_policy = run_grpo(n_steps, G, n_prompts, kl_coef)

    print(f"\n[2/2] PPO (for comparison) ...")
    ppo_hist = run_ppo_compare(n_steps, batch_size=G * n_prompts, kl_coef=kl_coef)

    print("\n  Plotting ...")
    plot_results(grpo_hist, ppo_hist, n_steps, G)

    print(f"\n  GRPO final reward:  {np.mean(grpo_hist['reward'][-20:]):.4f}")
    print(f"  PPO  final reward:  {np.mean(ppo_hist['reward'][-20:]):.4f}")
    print(f"  GRPO final KL:      {np.mean(grpo_hist['kl'][-20:]):.4f}")
    print("\n  Day 4 complete.\n")


if __name__ == "__main__":
    main()
