"""
Day 1 — PPO from First Principles
===================================
Environment : 2-arm bandit
              action 0 → E[R] = 0.3
              action 1 → E[R] = 0.7   ← optimal

We implement and compare three algorithms:
  1. REINFORCE   — vanilla policy gradient
  2. PPO         — proximal policy optimisation (clipped surrogate)

Key equations
-------------
Policy gradient theorem:
  ∇J(θ) = E_π [ ∇ log π_θ(a|s) · Q(s,a) ]

REINFORCE (Monte-Carlo policy gradient):
  θ ← θ + α · ∇ log π_θ(a) · R

Importance sampling ratio:
  r_t = π_θ(a|s) / π_θ_old(a|s)

PPO clipped objective (maximised):
  L^CLIP = E [ min( r_t · A_t, clip(r_t, 1−ε, 1+ε) · A_t ) ]

KL divergence between consecutive policies:
  KL(p_old || p_new) = Σ_a p_old(a) · log[ p_old(a) / p_new(a) ]

Run
---
  python day1_ppo_fundamentals.py            # full run  (~5 s)
  python day1_ppo_fundamentals.py --smoke    # quick run (~1 s)
"""

import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "day1")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Environment ──────────────────────────────────────────────────────────────

class TwoArmBandit:
    """
    Stateless bandit.  Two actions with Gaussian rewards.
    Optimal policy: always choose action 1.
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def step(self, action: int) -> float:
        mean = 0.3 if action == 0 else 0.7
        return float(self.rng.normal(mean, 0.1))


# ─── Policy ───────────────────────────────────────────────────────────────────

class BanditPolicy:
    """
    Softmax policy over 2 actions.  Parameterised by raw logits [w_0, w_1].

    π(a=k) = exp(w_k) / (exp(w_0) + exp(w_1))
    """
    def __init__(self):
        self.logits = torch.zeros(2, requires_grad=True)

    def probs(self) -> torch.Tensor:
        return F.softmax(self.logits, dim=0)

    def sample(self):
        probs = self.probs()
        dist  = torch.distributions.Categorical(probs)
        a     = dist.sample()
        lp    = dist.log_prob(a)
        return a.item(), lp, probs.detach().clone()

    def log_prob(self, action: int) -> torch.Tensor:
        return F.log_softmax(self.logits, dim=0)[action]

    def parameters(self):
        return [self.logits]


def kl_divergence(p_old: torch.Tensor, p_new: torch.Tensor) -> float:
    """KL(p_old || p_new)"""
    return (p_old * (p_old.log() - p_new.log())).sum().item()


# ─── REINFORCE ────────────────────────────────────────────────────────────────

def run_reinforce(n_steps: int, lr: float, seed: int = 0):
    """
    Vanilla REINFORCE on the 2-arm bandit.

    Each step: sample one action, observe reward, update policy.
    Update rule:  θ ← θ + α · ∇ log π(a) · R
    """
    env    = TwoArmBandit(seed)
    policy = BanditPolicy()
    opt    = torch.optim.Adam(policy.parameters(), lr=lr)

    rewards, p1_probs = [], []

    for _ in range(n_steps):
        action, log_prob, probs = policy.sample()
        reward = env.step(action)

        # Maximise E[log π(a) · R]  ↔  minimise −log π(a) · R
        loss = -log_prob * reward
        opt.zero_grad()
        loss.backward()
        opt.step()

        rewards.append(reward)
        p1_probs.append(probs[1].item())

    return rewards, p1_probs


# ─── PPO ──────────────────────────────────────────────────────────────────────

def run_ppo(
    n_steps: int,
    lr: float,
    eps: float = 0.2,
    n_epochs: int = 4,
    batch_size: int = 32,
    seed: int = 0,
):
    """
    PPO on the 2-arm bandit.

    Each outer step:
      1. Collect `batch_size` (action, reward) pairs with current policy π_old.
      2. Compute advantage  A = R − baseline  (running mean baseline).
      3. Run `n_epochs` PPO epochs over the collected batch.

    The clipped objective prevents the policy from moving too far from π_old
    in a single update — this is the key stability mechanism of PPO.
    """
    env    = TwoArmBandit(seed)
    policy = BanditPolicy()
    opt    = torch.optim.Adam(policy.parameters(), lr=lr)

    rewards_history = []
    p1_probs_history = []
    ratio_history   = []  # mean ratio per outer step
    clip_freq_history = []
    kl_history      = []

    baseline = 0.0  # exponential moving average of rewards

    for step in range(n_steps):
        # ── 1. Collect batch with old policy ──────────────────────
        probs_old = policy.probs().detach()  # snapshot for KL

        actions_list, lp_old_list, rewards_list = [], [], []
        for _ in range(batch_size):
            action, lp, _ = policy.sample()
            r = env.step(action)
            actions_list.append(action)
            lp_old_list.append(lp.detach())
            rewards_list.append(r)

        actions_t  = torch.tensor(actions_list, dtype=torch.long)
        lp_old_t   = torch.stack(lp_old_list)
        rewards_t  = torch.tensor(rewards_list, dtype=torch.float32)

        # ── 2. Advantage: subtract running baseline ────────────────
        baseline   = 0.9 * baseline + 0.1 * rewards_t.mean().item()
        advantages = rewards_t - baseline

        # Normalise advantages for stability
        if advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── 3. PPO update epochs ───────────────────────────────────
        step_ratios, step_clips = [], []
        for _ in range(n_epochs):
            lp_new = F.log_softmax(policy.logits, dim=0)[actions_t]
            ratios = (lp_new - lp_old_t).exp()  # r_t = π_new / π_old

            # Clipped surrogate objective
            surr1  = ratios * advantages
            surr2  = ratios.clamp(1.0 - eps, 1.0 + eps) * advantages
            loss   = -torch.min(surr1, surr2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            step_ratios.extend(ratios.detach().tolist())
            clipped = ((ratios < 1 - eps) | (ratios > 1 + eps)).float().mean().item()
            step_clips.append(clipped)

        probs_new = policy.probs().detach()
        kl        = kl_divergence(probs_old, probs_new)

        rewards_history.append(rewards_t.mean().item())
        p1_probs_history.append(policy.probs()[1].item())
        ratio_history.append(step_ratios)  # keep all for histogram
        clip_freq_history.append(np.mean(step_clips))
        kl_history.append(kl)

    return rewards_history, p1_probs_history, ratio_history, clip_freq_history, kl_history


# ─── Visualise ────────────────────────────────────────────────────────────────

def smooth(x, w: int = 20):
    arr = np.asarray(x, dtype=float)
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def plot_results(rf_rew, rf_p1,
                 ppo_rew, ppo_p1, ppo_ratios, ppo_clips, ppo_kls,
                 n_steps: int):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Day 1 — PPO from First Principles", fontsize=13, fontweight="bold")

    w = max(1, n_steps // 30)

    # ── (0,0) Reward curves ─────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(smooth(rf_rew,  w), label="REINFORCE", alpha=0.85)
    ax.plot(smooth(ppo_rew, w), label="PPO",       alpha=0.85)
    ax.axhline(0.7, ls="--", c="gray", lw=1, label="Optimal E[R]=0.7")
    ax.set_title("Mean Reward per Step")
    ax.set_xlabel("Outer step")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8)

    # ── (0,1) Policy convergence ────────────────────────────────
    ax = axes[0, 1]
    ax.plot(smooth(rf_p1,  w), label="REINFORCE", alpha=0.85)
    ax.plot(smooth(ppo_p1, w), label="PPO",       alpha=0.85)
    ax.axhline(1.0, ls="--", c="gray", lw=1, label="Optimal π(a=1)=1")
    ax.set_title("π(a=1) over Training")
    ax.set_xlabel("Outer step")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    # ── (0,2) Ratio distribution (last 30% of training) ─────────
    ax = axes[0, 2]
    # Flatten ratios from the last 30% of outer steps
    cutoff = int(0.7 * n_steps)
    all_ratios = [r for step_r in ppo_ratios[cutoff:] for r in step_r]
    ax.hist(all_ratios, bins=40, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(1.0,       c="red",    ls="--", lw=1.5, label="r=1  (no change)")
    ax.axvline(1.0 + 0.2, c="orange", ls="--", lw=1.2, label="clip bounds ±ε")
    ax.axvline(1.0 - 0.2, c="orange", ls="--", lw=1.2)
    ax.set_title("Ratio Distribution  r_t = π_new/π_old\n(late training)")
    ax.set_xlabel("Ratio r_t")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # ── (1,0) Clipping frequency ────────────────────────────────
    ax = axes[1, 0]
    ax.plot(smooth(ppo_clips, w), color="darkorange")
    ax.set_title("Clipping Frequency")
    ax.set_xlabel("Outer step")
    ax.set_ylabel("Fraction of Samples Clipped")
    ax.set_ylim(0, 1)
    ax.axhline(0.0, ls="--", c="gray", lw=1, label="Converged (no clipping)")
    ax.legend(fontsize=8)

    # ── (1,1) KL divergence ────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(smooth(ppo_kls, w), color="crimson")
    ax.set_title("KL(π_old || π_new) per Outer Step")
    ax.set_xlabel("Outer step")
    ax.set_ylabel("KL Divergence (nats)")

    # ── (1,2) Key equations annotation ─────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    txt = (
        "Key Equations\n"
        "─────────────────────────────────\n"
        "REINFORCE:\n"
        "  ∇J = E[ ∇ log π(a) · R ]\n\n"
        "Importance sampling ratio:\n"
        "  r_t = π_θ(a) / π_θ_old(a)\n\n"
        "PPO clipped objective:\n"
        "  L = E[ min(\n"
        "    r_t · A_t,\n"
        "    clip(r_t, 1−ε, 1+ε) · A_t\n"
        "  ) ]\n\n"
        "KL penalty (alternative):\n"
        "  L = E[r_t · A_t] − β · KL\n\n"
        f"  ε = 0.2  (this experiment)"
    )
    ax.text(0.05, 0.97, txt, transform=ax.transAxes,
            fontsize=9.5, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fffbe6", alpha=0.9))

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "day1_ppo_fundamentals.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test")
    args = parser.parse_args()

    n_steps    = 200  if args.smoke else 800
    batch_size = 16   if args.smoke else 64
    n_epochs   = 2    if args.smoke else 4
    lr         = 0.05

    print("=" * 55)
    print("  Day 1 — PPO from First Principles")
    print(f"  mode={'smoke' if args.smoke else 'full'}  steps={n_steps}  batch={batch_size}")
    print("=" * 55)

    print("\n[1/2] REINFORCE ...")
    rf_rew, rf_p1 = run_reinforce(n_steps, lr)

    print("[2/2] PPO ...")
    ppo_rew, ppo_p1, ppo_ratios, ppo_clips, ppo_kls = run_ppo(
        n_steps, lr, eps=0.2, n_epochs=n_epochs, batch_size=batch_size
    )

    print("\nPlotting ...")
    plot_results(rf_rew, rf_p1,
                 ppo_rew, ppo_p1, ppo_ratios, ppo_clips, ppo_kls,
                 n_steps)

    print(f"\n  Final π(a=1):  REINFORCE = {rf_p1[-1]:.3f}   PPO = {ppo_p1[-1]:.3f}")
    print(f"  Final KL:      {ppo_kls[-1]:.6f}")
    print("\n  Day 1 complete.\n")


if __name__ == "__main__":
    main()
