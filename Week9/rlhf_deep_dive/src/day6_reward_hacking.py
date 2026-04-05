"""
Day 6 — Reward Hacking Simulation
====================================
We deliberately design degenerate reward functions and watch the policy
exploit them.  The goal is to *see* reward hacking in action.

Three reward functions
----------------------
  1. Repetition reward:
       R = count(token_0) / T
       Hack: policy outputs all token_0s → gets max reward without "doing" anything useful.

  2. Keyword reward:
       R = 1 if any(tokens == TARGET_TOKEN) else 0
       Hack: policy puts TARGET_TOKEN first, then fills rest randomly.

  3. Format reward:
       R = 1 if tokens[0] == OPEN and tokens[-1] == CLOSE else 0
       Hack: policy learns to bracket every sequence, ignoring middle content.

For each reward we train GRPO without a KL penalty (β=0) to maximise hacking,
and WITH a KL penalty to show how KL mitigates but doesn't eliminate it.

Metrics we track
----------------
  - Mean reward (expected to rise quickly — that's the hack)
  - Token entropy (expected to collapse as policy specialises)
  - Repetition score  = max(count(token_x)) / T   (how repetitive)
  - True quality proxy: P(sequence has all distinct tokens) — never optimised for

Run
---
  python day6_reward_hacking.py
  python day6_reward_hacking.py --smoke
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "day6")
os.makedirs(RESULTS_DIR, exist_ok=True)

VOCAB_SIZE    = 8
SEQ_LEN       = 4
TARGET_TOKEN  = 3   # for keyword reward
OPEN_TOKEN    = 1   # for format reward
CLOSE_TOKEN   = 6   # for format reward


# ─── Reward functions ─────────────────────────────────────────────────────────

def repetition_reward(tokens: torch.Tensor) -> torch.Tensor:
    """R = fraction of tokens that are token_0."""
    return (tokens == 0).float().mean(dim=1)


def keyword_reward(tokens: torch.Tensor) -> torch.Tensor:
    """R = 1 if TARGET_TOKEN appears anywhere in the sequence."""
    return (tokens == TARGET_TOKEN).any(dim=1).float()


def format_reward(tokens: torch.Tensor) -> torch.Tensor:
    """R = 1 if sequence starts with OPEN and ends with CLOSE."""
    starts = (tokens[:, 0]  == OPEN_TOKEN).float()
    ends   = (tokens[:, -1] == CLOSE_TOKEN).float()
    return starts * ends


REWARD_CONFIGS = {
    "repetition": {
        "fn":    repetition_reward,
        "title": "Repetition Reward  R = count(token_0) / T",
        "hack":  "Policy collapses to all-zeros output",
    },
    "keyword": {
        "fn":    keyword_reward,
        "title": f"Keyword Reward  R = 1 if token_{TARGET_TOKEN} present",
        "hack":  "Policy puts target token first, rest random",
    },
    "format": {
        "fn":    format_reward,
        "title": f"Format Reward  R = 1 if [tok_{OPEN_TOKEN}...tok_{CLOSE_TOKEN}]",
        "hack":  "Policy brackets every output, middle content ignored",
    },
}


# ─── Diagnostic metrics ───────────────────────────────────────────────────────

@torch.no_grad()
def repetition_score(policy: TinyRNNPolicy, n: int = 256) -> float:
    """Max fraction of any single token in generated sequences."""
    tokens, _, _ = policy.sample_sequence(n)
    counts = torch.zeros(VOCAB_SIZE)
    for t in range(VOCAB_SIZE):
        counts[t] = (tokens == t).float().mean()
    return counts.max().item()


@torch.no_grad()
def diversity_score(policy: TinyRNNPolicy, n: int = 256) -> float:
    """Fraction of sequences that have all distinct tokens (true quality proxy)."""
    tokens, _, _ = policy.sample_sequence(n)
    distinct = torch.zeros(n, dtype=torch.bool)
    for i in range(n):
        distinct[i] = (tokens[i].unique().numel() == SEQ_LEN)
    return distinct.float().mean().item()


# ─── GRPO trainer ─────────────────────────────────────────────────────────────

def run_grpo(reward_fn, n_steps: int, G: int, n_prompts: int, kl_coef: float) -> dict:
    policy     = TinyRNNPolicy(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN)
    ref_policy = copy.deepcopy(policy)
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

    B   = n_prompts * G
    eps = 0.2

    history = {
        "reward":      [],
        "entropy":     [],
        "rep_score":   [],
        "diversity":   [],
        "kl":          [],
    }

    for step in range(n_steps):
        with torch.no_grad():
            tokens, lp_old, _ = policy.sample_sequence(B)
            rewards = reward_fn(tokens)

        # Group-normalised advantage
        r_g   = rewards.view(n_prompts, G)
        r_m   = r_g.mean(dim=1, keepdim=True)
        r_s   = r_g.std(dim=1,  keepdim=True) + 1e-8
        adv   = ((r_g - r_m) / r_s).view(B).detach()

        lp_new, _ = policy.log_probs_of(tokens)
        log_r      = (lp_new - lp_old.detach()).sum(dim=1)
        ratio      = log_r.exp()
        surr1      = ratio * adv
        surr2      = ratio.clamp(1 - eps, 1 + eps) * adv
        kl         = kl_per_token(policy, ref_policy, tokens)
        loss       = -torch.min(surr1, surr2).mean() + kl_coef * kl

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        history["reward"].append(rewards.mean().item())
        history["entropy"].append(token_entropy(policy))
        history["rep_score"].append(repetition_score(policy))
        history["diversity"].append(diversity_score(policy))
        history["kl"].append(kl_per_token(policy, ref_policy, tokens))

    return history


# ─── Visualise ────────────────────────────────────────────────────────────────

def plot_reward_hacking(all_results: dict, n_steps: int):
    """
    all_results: {reward_name: {"no_kl": history, "with_kl": history}}
    """
    n_rewards = len(all_results)
    fig, axes = plt.subplots(n_rewards, 4, figsize=(18, n_rewards * 4))
    fig.suptitle("Day 6 — Reward Hacking Simulation", fontsize=13, fontweight="bold")

    if n_rewards == 1:
        axes = axes[np.newaxis, :]

    w = max(1, n_steps // 30)

    for row, (rname, runs) in enumerate(all_results.items()):
        cfg   = REWARD_CONFIGS[rname]
        no_kl = runs["no_kl"]
        kl    = runs["with_kl"]

        # Reward
        ax = axes[row, 0]
        ax.plot(smooth(no_kl["reward"],  w), label="β=0 (no KL)",   color="#e74c3c", lw=2)
        ax.plot(smooth(kl["reward"],     w), label="β=0.1 (KL)",    color="#2ecc71", lw=2)
        ax.set_title(f"{cfg['title']}\nReward", fontsize=9)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel("Step")
        ax.legend(fontsize=7)
        if row == 0:
            ax.set_ylabel("Mean Reward")

        # Entropy
        ax = axes[row, 1]
        ax.plot(smooth(no_kl["entropy"], w), color="#e74c3c", lw=2, label="β=0")
        ax.plot(smooth(kl["entropy"],    w), color="#2ecc71", lw=2, label="β=0.1")
        ax.set_title("Token Entropy (↓ = collapse)", fontsize=9)
        ax.set_xlabel("Step")
        ax.set_ylim(0)
        ax.legend(fontsize=7)
        if row == 0:
            ax.set_ylabel("Entropy (nats)")

        # Repetition score
        ax = axes[row, 2]
        ax.plot(smooth(no_kl["rep_score"], w), color="#e74c3c", lw=2, label="β=0")
        ax.plot(smooth(kl["rep_score"],    w), color="#2ecc71", lw=2, label="β=0.1")
        ax.axhline(1/VOCAB_SIZE, ls="--", c="gray", lw=1,
                   label=f"Uniform={1/VOCAB_SIZE:.2f}")
        ax.set_title("Repetition Score\nmax single-token fraction", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Step")
        ax.legend(fontsize=7)
        if row == 0:
            ax.set_ylabel("Fraction")

        # Diversity (true quality proxy — never optimised for)
        ax = axes[row, 3]
        ax.plot(smooth(no_kl["diversity"], w), color="#e74c3c", lw=2, label="β=0")
        ax.plot(smooth(kl["diversity"],    w), color="#2ecc71", lw=2, label="β=0.1")
        ax.axhline(0, ls="--", c="gray", lw=1)
        ax.set_title("Diversity Score\n(P(all distinct) — NOT optimised)", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Step")
        ax.legend(fontsize=7)
        if row == 0:
            ax.set_ylabel("Fraction")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "day6_reward_hacking.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_hacking_summary(all_results: dict):
    """Side-by-side: reward goes up while quality goes down."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Day 6 — Reward ↑ while Quality ↓  (the hacking signature)",
                 fontsize=12, fontweight="bold")

    colors = {"repetition": "#e74c3c", "keyword": "#e67e22", "format": "#9b59b6"}
    w = 10

    ax_r = axes[0]
    ax_d = axes[1]

    for rname, runs in all_results.items():
        h = runs["no_kl"]
        ax_r.plot(smooth(h["reward"],    w), color=colors[rname], label=rname, lw=2)
        ax_d.plot(smooth(h["diversity"], w), color=colors[rname], label=rname, lw=2)

    ax_r.set_title("Reward (designed metric) ↑")
    ax_r.set_xlabel("Step")
    ax_r.set_ylabel("Reward")
    ax_r.set_ylim(-0.05, 1.1)
    ax_r.legend(fontsize=9)

    ax_d.set_title("Diversity (true quality) ↓\n← NEVER optimised for")
    ax_d.set_xlabel("Step")
    ax_d.set_ylabel("Fraction with all-distinct tokens")
    ax_d.set_ylim(-0.05, 1.05)
    ax_d.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "day6_hacking_summary.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    n_steps   = 150  if args.smoke else 600
    G         = 4    if args.smoke else 8
    n_prompts = 8    if args.smoke else 16

    print("=" * 55)
    print("  Day 6 — Reward Hacking Simulation")
    print(f"  mode={'smoke' if args.smoke else 'full'}  n_steps={n_steps}")
    print("=" * 55)

    all_results = {}
    for rname, cfg in REWARD_CONFIGS.items():
        print(f"\n  Reward: {rname}")
        print(f"  Hack:   {cfg['hack']}")

        print("    Running β=0 (no KL) ...")
        h_no_kl = run_grpo(cfg["fn"], n_steps, G, n_prompts, kl_coef=0.0)

        print("    Running β=0.1 (with KL) ...")
        h_kl = run_grpo(cfg["fn"], n_steps, G, n_prompts, kl_coef=0.1)

        all_results[rname] = {"no_kl": h_no_kl, "with_kl": h_kl}

        # Print final state
        r0   = np.mean(h_no_kl["reward"][-20:])
        div0 = np.mean(h_no_kl["diversity"][-20:])
        r1   = np.mean(h_kl["reward"][-20:])
        div1 = np.mean(h_kl["diversity"][-20:])
        print(f"    β=0:   reward={r0:.3f}  diversity={div0:.3f}  "
              f"(reward↑ {'+' if r0>0.125 else ''}  quality↓ {'-' if div0<0.2 else ''})")
        print(f"    β=0.1: reward={r1:.3f}  diversity={div1:.3f}")

    print("\n  Plotting ...")
    plot_reward_hacking(all_results, n_steps)
    plot_hacking_summary(all_results)

    print("\n  Reward hacking signatures to watch for:")
    print("    1. Reward increases rapidly while diversity drops")
    print("    2. Entropy collapses (policy becomes deterministic)")
    print("    3. Repetition score approaches 1.0")
    print("    4. KL grows unboundedly (without β penalty)")
    print("\n  Day 6 complete.\n")


if __name__ == "__main__":
    main()
