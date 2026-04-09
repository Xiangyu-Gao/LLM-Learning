"""
day4_instability_taxonomy.py — A Taxonomy of RLHF Failure Modes.
================================================================

CORE CONCEPT
------------
RLHF training can fail in qualitatively distinct ways.  Each failure mode
has a distinctive signature in the metrics (reward, KL, entropy, gradient norm).
Knowing which mode you're in tells you exactly what to fix.

THE FIVE FAILURE MODES
----------------------

1. REWARD COLLAPSE
   Cause:    Non-stationary reward (reward model overfit / distribution shift) →
             policy perpetually chases a shifting target → reward oscillates.
             In real RLHF: reward model gets "hacked", effectively changing what
             the reward model values over time.
   Signature: Reward rises then crashes repeatedly.  Gradient norm spikes at
              each phase transition as the policy must reverse its preferences.
   Fix:       Monitor reward model calibration, use reward model ensembles,
              periodic reward model re-training.

2. MODE COLLAPSE
   Cause:    KL coefficient β too high → policy can't move away from reference.
   Signature: KL stays near 0, entropy decays monotonically, reward plateaus below optimum.
   Fix:       Reduce β, add entropy bonus, use temperature-scaled sampling.

3. KL EXPLOSION
   Cause:    Learning rate too high + no KL penalty → policy drifts far from
             reference in very few steps.
   Signature: KL grows steeply in the first few dozen steps, reaches maximum
              (log V for vocab size V).  Policy may find reward but on a
              distribution the reward model was never trained on.
   Fix:       Add KL penalty, use PPO clipping, reduce LR.

4. ENTROPY DEATH
   Cause:    Reward strongly incentivises a specific token at every position +
             no entropy regularization → all token distributions collapse.
   Signature: Entropy collapses from log(V) → ~0 monotonically.  Reward appears
              high but diversity is gone.  Model outputs repetitive text.
   Fix:       Entropy bonus, temperature scaling, reward reshaping.

5. GRADIENT SPIKES
   Cause:    Batch size too small + high LR + no clipping → gradient variance
             too high → occasional large steps that destabilize training.
   Signature: Gradient norm has sharp spikes.  Reward and KL are noisy
              with sudden jumps.
   Fix:       Larger batch, gradient clipping, reduced LR.

EXPERIMENT DESIGN NOTES
-----------------------
  reward_collapse  : Oscillating reward (phase flips every phase_len steps).
                     Phase 0 rewards [x[0]=0, x[-1]=1]; phase 1 rewards
                     [x[0]=1, x[-1]=0] — opposite target.
  mode_collapse    : β=5.0 (huge KL penalty) prevents exploration.  ✓ same as before.
  kl_explosion     : LR=5e-2 (50× larger), no KL penalty — KL grows steeply fast.
  entropy_death    : Reward = number of zeros in sequence (partial credit at
                     every position).  Policy collapses all positions to 0 → H→0.
  gradient_spikes  : batch=4, LR=5e-2, no clipping — high per-step variance
                     plus large unconstrained steps.

RELATIONSHIP TO REAL RLHF
--------------------------
  Reward hacking      ≈ KL explosion (policy escapes to out-of-distribution region)
  Alignment tax       ≈ mode collapse (KL β too high, restricts useful behavior)
  Training instability ≈ reward collapse or gradient spikes
  Repetitive output   ≈ entropy death

OUTPUT
------
results/day4/day4_failure_library.png   — 5-panel failure signatures
results/day4/day4_comparison.png        — Side-by-side in same axes (reward + KL + entropy)

INTERVIEW GOLD
--------------
Q: "What does reward collapse look like in your metrics?"
A: Reward rises then crashes repeatedly.  Gradient norm spikes at each crash
   (policy attempts a large reversal step).  Monitor reward trend over a rolling
   window — sustained decline after initial rise signals collapse.

Q: "How do you detect mode collapse?"
A: Watch entropy.  If entropy is monotonically decreasing and reward has plateaued
   below the theoretical maximum, you're in mode collapse.  Check also if the
   model outputs the same sequence repeatedly.

Q: "What is entropy death vs mode collapse?"
A: Mode collapse = policy concentrates probability on a small subset of sequences.
   Entropy death = the extreme end of mode collapse where one sequence dominates.
   In LLMs: entropy death manifests as degenerate repetition (the 'I am a helpful
   assistant' loop, or '......' outputs).
"""

import argparse
import os
import sys
from copy import deepcopy
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    TinyRNNPolicy, make_policy, kl_per_token, token_entropy,
    gradient_norm, smooth,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "day4")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Reward ──────────────────────────────────────────────────────────────────

def shaped_reward(tokens: torch.Tensor) -> torch.Tensor:
    r = torch.zeros(tokens.shape[0])
    r += 2.0 * (tokens[:, 0] == 0).float()
    r += 2.0 * (tokens[:, -1] == 1).float()
    return r


def all_zero_reward(tokens: torch.Tensor) -> torch.Tensor:
    """
    Reward = number of zeros in the sequence (0 to seq_len).

    Partial credit at every position → clear gradient at every step.
    Without entropy regularisation the policy quickly collapses all
    positions to token=0, driving entropy to near 0.
    """
    return tokens.eq(0).float().sum(dim=1)


class OscillatingReward:
    """
    Non-stationary reward that flips its target every `phase_len` steps.

    Phase 0: reward [x[0]=0, x[-1]=1]  (same as shaped_reward peak)
    Phase 1: reward [x[0]=1, x[-1]=0]  (opposite target)

    Simulates reward model overfitting / distribution shift in real RLHF:
    the reward model effectively changes what it values, making the policy
    perpetually chase a moving target → reward rises then crashes each cycle.
    """
    def __init__(self, phase_len: int = 60):
        self.phase_len = phase_len
        self._step     = 0

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        phase = (self._step // self.phase_len) % 2
        self._step += 1
        if phase == 0:
            r = 2.0 * (tokens[:, 0] == 0).float() + 2.0 * (tokens[:, -1] == 1).float()
        else:
            r = 2.0 * (tokens[:, 0] == 1).float() + 2.0 * (tokens[:, -1] == 0).float()
        return r


# ─── Generic training loop ───────────────────────────────────────────────────

def train_scenario(
    scenario:   str,
    n_steps:    int,
    batch_size: int,
) -> Dict[str, List[float]]:
    """
    Each scenario deliberately triggers a specific failure mode.

    Parameters for each scenario:
      reward_collapse  : oscillating reward (non-stationary), LR=1e-3, no KL
      mode_collapse    : LR=1e-3, kl_beta=5.0 (huge KL penalty)
      kl_explosion     : LR=5e-2 (high), no KL penalty → fast drift from reference
      entropy_death    : all_zero_reward (partial credit at all positions), no regularisation
      gradient_spikes  : batch=4, LR=5e-2, no clip → high-variance unconstrained steps
    """
    phase_len = max(10, n_steps // 6)   # 6 oscillation cycles over the full run
    configs = {
        "reward_collapse": dict(lr=1e-3,  batch=batch_size, kl_beta=0.0,
                                ent_alpha=0.0, clip=False,
                                reward_fn=OscillatingReward(phase_len=phase_len)),
        "mode_collapse":   dict(lr=1e-3,  batch=batch_size, kl_beta=5.0,
                                ent_alpha=0.0, clip=True,  reward_fn=shaped_reward),
        "kl_explosion":    dict(lr=5e-2,  batch=batch_size, kl_beta=0.0,
                                ent_alpha=0.0, clip=False, reward_fn=shaped_reward),
        "entropy_death":   dict(lr=1e-3,  batch=batch_size, kl_beta=0.0,
                                ent_alpha=0.0, clip=False, reward_fn=all_zero_reward),
        "gradient_spikes": dict(lr=5e-2,  batch=4,          kl_beta=0.0,
                                ent_alpha=0.0, clip=False, reward_fn=shaped_reward),
    }

    cfg        = configs[scenario]
    policy     = make_policy("small")
    ref_policy = deepcopy(policy)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])

    history: Dict[str, List[float]] = {
        k: [] for k in ("reward", "kl", "entropy", "grad_norm")
    }

    for step in range(n_steps):
        policy.train()
        B      = cfg["batch"]
        tokens, log_probs, _ = policy.sample_sequence(B)
        rewards = cfg["reward_fn"](tokens)

        # KL penalty
        with torch.no_grad():
            ref_lp, _ = ref_policy.log_probs_of(tokens)
        kl_per_tok = log_probs - ref_lp               # (B, T)

        # Effective reward
        step_rewards = torch.zeros_like(log_probs)
        step_rewards[:, -1] = rewards
        r_eff = step_rewards - cfg["kl_beta"] * kl_per_tok + cfg["ent_alpha"] * (-log_probs)

        # Advantage
        seq_reff   = r_eff.sum(dim=1)
        baseline   = seq_reff.mean().detach()
        advantages = seq_reff - baseline

        # Loss
        seq_lp = log_probs.sum(dim=1)
        loss   = -(seq_lp * advantages).mean()

        optimizer.zero_grad()
        loss.backward()
        gnorm = gradient_norm(policy)
        if cfg["clip"]:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        with torch.no_grad():
            kl_val  = kl_per_token(policy, ref_policy, tokens)
            ent_val = token_entropy(policy)

        history["reward"].append(rewards.mean().item())
        history["kl"].append(kl_val)
        history["entropy"].append(ent_val)
        history["grad_norm"].append(gnorm)

    return history


# ─── Plotting ────────────────────────────────────────────────────────────────

SCENARIO_META = {
    "reward_collapse": {
        "color":   "crimson",
        "title":   "1. Reward Collapse",
        "cause":   "Non-stationary reward → policy chases shifting target",
        "signal":  "Reward rises then crashes (repeating cycle)",
    },
    "mode_collapse": {
        "color":   "steelblue",
        "title":   "2. Mode Collapse",
        "cause":   "β too large → policy can't explore",
        "signal":  "KL ≈ 0, entropy decays, reward plateau",
    },
    "kl_explosion": {
        "color":   "darkorange",
        "title":   "3. KL Explosion",
        "cause":   "High LR + no KL penalty → fast drift from reference",
        "signal":  "KL grows steeply then plateaus at max (log V)",
    },
    "entropy_death": {
        "color":   "purple",
        "title":   "4. Entropy Death",
        "cause":   "All-position reward + no entropy reg → full collapse",
        "signal":  "Entropy → 0 monotonically, reward converges",
    },
    "gradient_spikes": {
        "color":   "darkgreen",
        "title":   "5. Gradient Spikes",
        "cause":   "Batch=4 + LR=5e-2 + no clip → high-variance steps",
        "signal":  "Spiky grad norm, noisy reward/KL",
    },
}


def plot_failure_library(
    all_histories: Dict[str, Dict[str, List]],
    n_steps:       int,
):
    """
    5-column figure, each column = one failure mode.
    4 rows: reward, KL, entropy, grad norm.
    """
    scenarios = list(SCENARIO_META.keys())
    n_cols    = len(scenarios)
    n_rows    = 4
    metrics   = ["reward", "kl", "entropy", "grad_norm"]
    ylabels   = ["Reward", "KL(π||π_ref)", "Token Entropy (nats)", "Gradient Norm"]
    w         = min(10, n_steps // 15)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))

    for col, scenario in enumerate(scenarios):
        hist  = all_histories[scenario]
        meta  = SCENARIO_META[scenario]
        color = meta["color"]

        for row, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
            ax = axes[row, col]
            ax.plot(smooth(hist[metric], w=w), color=color, alpha=0.85, linewidth=1.5)

            if row == 0:
                ax.set_title(
                    f"{meta['title']}\nCause: {meta['cause']}",
                    fontsize=9, fontweight="bold", color=meta["color"],
                )
                # Annotate the failure signature
                ax.text(0.98, 0.98, f"Signal: {meta['signal']}",
                        transform=ax.transAxes, fontsize=7,
                        ha="right", va="top",
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel("Step", fontsize=8)

    fig.suptitle(
        "Day 4: RLHF Instability Taxonomy — Five Failure Modes",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "day4_failure_library.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_overlay_comparison(
    all_histories: Dict[str, Dict[str, List]],
    n_steps:       int,
):
    """
    Overlay all scenarios on three axes (reward, KL, entropy) for comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    w = min(10, n_steps // 15)

    for scenario, hist in all_histories.items():
        meta  = SCENARIO_META[scenario]
        color = meta["color"]
        label = meta["title"].split(".")[1].strip()

        axes[0].plot(smooth(hist["reward"],  w=w), color=color, label=label, alpha=0.85)
        axes[1].plot(smooth(hist["kl"],      w=w), color=color, label=label, alpha=0.85)
        axes[2].plot(smooth(hist["entropy"], w=w), color=color, label=label, alpha=0.85)

    axes[0].set_title("Reward"); axes[0].set_ylabel("Mean Reward")
    axes[0].set_xlabel("Step");  axes[0].legend(fontsize=8)

    axes[1].set_title("KL Divergence"); axes[1].set_ylabel("KL(π||π_ref)")
    axes[1].set_xlabel("Step");         axes[1].legend(fontsize=8)
    axes[1].axhline(1.0, color="red", linestyle="--", alpha=0.3)

    axes[2].set_title("Token Entropy"); axes[2].set_ylabel("H(π) (nats)")
    axes[2].set_xlabel("Step");         axes[2].legend(fontsize=8)

    fig.suptitle(
        "Day 4: All Failure Modes Overlaid",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "day4_comparison.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        n_steps    = 50
        batch_size = 32
    else:
        n_steps    = 400
        batch_size = 64

    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print(" Day 4: Instability Taxonomy")
    print("=" * 60)
    print()

    all_histories = {}
    for scenario in SCENARIO_META:
        meta = SCENARIO_META[scenario]
        print(f"[{meta['title']}]  ({meta['cause']})...")
        hist = train_scenario(scenario, n_steps, batch_size)
        all_histories[scenario] = hist

        final_r   = np.mean(hist["reward"][-20:])
        min_r     = min(hist["reward"])
        max_r     = max(hist["reward"])
        final_kl  = np.mean(hist["kl"][-20:])
        final_ent = np.mean(hist["entropy"][-20:])
        max_gnorm = max(hist["grad_norm"])
        print(f"  → final_r={final_r:.2f}  (min={min_r:.2f}, max={max_r:.2f})  "
              f"kl={final_kl:.2f}  ent={final_ent:.2f}  max_gnorm={max_gnorm:.1f}")
        print()

    print("Generating plots...")
    plot_failure_library(all_histories, n_steps)
    plot_overlay_comparison(all_histories, n_steps)

    # Diagnosis guide
    print()
    print("DIAGNOSIS GUIDE:")
    print("-" * 60)
    diag = [
        ("Reward rise then crash",         "gradient_spikes or reward_collapse"),
        ("Gradient norm spike before crash","reward_collapse (LR too high)"),
        ("KL near 0, entropy falling",      "mode_collapse (β too high)"),
        ("KL growing without bound",        "kl_explosion (no KL penalty)"),
        ("Entropy → 0, reward looks fine",  "entropy_death (peaked reward)"),
        ("Spiky noisy reward/KL curves",    "gradient_spikes (batch too small)"),
    ]
    for symptom, diagnosis in diag:
        print(f"  Symptom: {symptom}")
        print(f"  Likely:  {diagnosis}")
        print()

    print("Day 4 complete.")


if __name__ == "__main__":
    main()
