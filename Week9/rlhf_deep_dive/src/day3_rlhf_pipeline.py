"""
Day 3 — RLHF Pipeline Internals
=================================
We implement the complete RLHF pipeline on a toy language model:

  Stage 1 — SFT    : Pre-train TinyRNNPolicy on supervised sequences.
  Stage 2 — RM     : Train TinyRewardModel on synthetic preferences
                     using the Bradley-Terry (pairwise ranking) loss.
  Stage 3 — PPO-RL : Fine-tune the policy with PPO against the RM,
                     with a KL penalty to the frozen SFT reference model.

Token alphabet : {0, 1, 2, 3, 4, 5, 6, 7}
Sequence length: 4

Preference definition (synthetic):
  "better" sequence — sorted / non-decreasing (e.g. [0,1,2,3])
  "worse"  sequence — unsorted or starting with high tokens (e.g. [7,3,0,5])

KL penalty in PPO:
  L = E[clip(r_t,1−ε,1+ε)·A_t] − β · KL(π_ref || π_θ)

  Without the KL term the policy drifts to exploit the imperfect RM,
  causing reward hacking (shown explicitly in Day 6).

Run
---
  python day3_rlhf_pipeline.py
  python day3_rlhf_pipeline.py --smoke
"""

import argparse, os, sys, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import TinyRNNPolicy, TinyRewardModel, rollout, kl_per_token, token_entropy, smooth

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "day3")
os.makedirs(RESULTS_DIR, exist_ok=True)

VOCAB_SIZE = 8
SEQ_LEN    = 4


# ─── Stage 1: SFT ─────────────────────────────────────────────────────────────

def is_sorted(tokens: torch.Tensor) -> torch.Tensor:
    """tokens: (B,T). Returns bool (B,) — True if non-decreasing."""
    return (tokens[:, 1:] >= tokens[:, :-1]).all(dim=1)


def generate_sft_data(n: int, seed: int = 0) -> torch.Tensor:
    """
    Generate n "preferred" sequences: sorted token sequences sampled from
    a distribution that favours smaller tokens at the start.
    """
    rng = np.random.default_rng(seed)
    seqs = []
    while len(seqs) < n:
        s = sorted(rng.integers(0, VOCAB_SIZE, size=SEQ_LEN).tolist())
        seqs.append(s)
    return torch.tensor(seqs, dtype=torch.long)


def run_sft(policy: TinyRNNPolicy, n_steps: int, batch_size: int, lr: float = 3e-3):
    """Supervised fine-tuning: maximise log p(x) on the SFT dataset."""
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    dataset = generate_sft_data(n=5000)

    losses = []
    for step in range(n_steps):
        idx   = torch.randint(0, len(dataset), (batch_size,))
        batch = dataset[idx]           # (B, T)
        lp, _ = policy.log_probs_of(batch)   # (B, T)
        loss  = -lp.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    return losses


# ─── Stage 2: Reward Model ────────────────────────────────────────────────────

def make_preference_pair(rng: np.random.Generator):
    """
    Return (preferred, rejected) token arrays.
    preferred = sorted sequence (non-decreasing)
    rejected  = different sequence (unsorted or at least non-identical)
    """
    base    = sorted(rng.integers(0, VOCAB_SIZE, size=SEQ_LEN).tolist())
    shuffle = base.copy()
    for _ in range(20):
        rng.shuffle(shuffle)
        if shuffle != base:
            return base, shuffle
    # Edge case: all tokens identical → flip one token in the rejected copy
    shuffle    = base.copy()
    shuffle[0] = (shuffle[0] + 1) % VOCAB_SIZE
    return base, shuffle


def generate_preferences(n: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    preferred, rejected = [], []
    for _ in range(n):
        p, r = make_preference_pair(rng)
        preferred.append(p)
        rejected.append(r)
    return (torch.tensor(preferred, dtype=torch.long),
            torch.tensor(rejected, dtype=torch.long))


def bradley_terry_loss(rm: TinyRewardModel,
                       preferred: torch.Tensor,
                       rejected:  torch.Tensor) -> torch.Tensor:
    """
    Bradley-Terry pairwise ranking loss:
      P(preferred > rejected) = sigmoid(r(preferred) − r(rejected))
      Loss = −E[ log σ(r(preferred) − r(rejected)) ]
    """
    r_pref = rm(preferred)   # (B,)
    r_rej  = rm(rejected)    # (B,)
    return -F.logsigmoid(r_pref - r_rej).mean()


def run_reward_model_training(rm: TinyRewardModel,
                              n_steps: int, batch_size: int, lr: float = 1e-3):
    preferred_all, rejected_all = generate_preferences(n=2000)
    opt = torch.optim.Adam(rm.parameters(), lr=lr)
    losses, accs = [], []

    for step in range(n_steps):
        idx      = torch.randint(0, len(preferred_all), (batch_size,))
        pref     = preferred_all[idx]
        rej      = rejected_all[idx]
        loss     = bradley_terry_loss(rm, pref, rej)
        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            acc = (rm(pref) > rm(rej)).float().mean().item()
        losses.append(loss.item())
        accs.append(acc)

    return losses, accs


# ─── Stage 3: PPO with KL penalty ─────────────────────────────────────────────

def ppo_step(
    policy:     TinyRNNPolicy,
    ref_policy: TinyRNNPolicy,
    rm:         TinyRewardModel,
    opt:        torch.optim.Optimizer,
    batch_size: int,
    eps:        float = 0.2,
    kl_coef:    float = 0.1,
    n_epochs:   int   = 2,
):
    """One outer PPO step: collect rollout, compute advantages, update policy."""
    # ── Collect rollout ────────────────────────────────────────
    with torch.no_grad():
        tokens, lp_old, _ = policy.sample_sequence(batch_size)
        rm_reward = rm(tokens)                         # (B,)
        kl        = kl_per_token(policy, ref_policy, tokens)

    advantages = rm_reward - kl_coef * kl   # scalar KL subtracted from each sample
    advantages = advantages.detach()
    if advantages.std() > 1e-6:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ── PPO epochs ────────────────────────────────────────────
    for _ in range(n_epochs):
        lp_new, _ = policy.log_probs_of(tokens)       # (B, T)
        # Sequence-level ratio: sum of per-token log-prob differences
        log_ratio = (lp_new - lp_old.detach()).sum(dim=1)  # (B,)
        ratio     = log_ratio.exp()

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - eps, 1 + eps) * advantages
        loss  = -torch.min(surr1, surr2).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

    metrics = {
        "rm_reward": rm_reward.mean().item(),
        "kl":        kl,
        "p_sorted":  is_sorted(tokens).float().mean().item(),
        "entropy":   token_entropy(policy),
    }
    return metrics


def run_ppo_rl(policy: TinyRNNPolicy, ref_policy: TinyRNNPolicy,
               rm: TinyRewardModel,
               n_steps: int, batch_size: int, kl_coef: float = 0.1):
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    history = {"rm_reward": [], "kl": [], "p_sorted": [], "entropy": []}

    for step in range(n_steps):
        m = ppo_step(policy, ref_policy, rm, opt, batch_size, kl_coef=kl_coef)
        for k, v in m.items():
            history[k].append(v)

        if (step + 1) % max(1, n_steps // 5) == 0:
            print(f"    step {step+1:4d}/{n_steps}  "
                  f"rm_reward={m['rm_reward']:.3f}  "
                  f"kl={m['kl']:.4f}  "
                  f"p_sorted={m['p_sorted']:.2f}  "
                  f"H={m['entropy']:.3f}")

    return history


# ─── Visualise ────────────────────────────────────────────────────────────────

def plot_results(sft_loss, rm_loss, rm_acc, ppo_history, n_ppo):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Day 3 — RLHF Pipeline Internals", fontsize=13, fontweight="bold")

    w = max(1, n_ppo // 30)

    # ── (0,0) SFT loss ────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(smooth(sft_loss, 10), color="#3498db")
    ax.set_title("Stage 1: SFT Training Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("−log p(x)")

    # ── (0,1) RM loss + accuracy ──────────────────────────────
    ax = axes[0, 1]
    ax2 = ax.twinx()
    ax.plot(smooth(rm_loss, 10), color="#e74c3c", label="BT loss", alpha=0.9)
    ax2.plot(smooth(rm_acc, 10), color="#2ecc71", label="Accuracy", alpha=0.9)
    ax.set_title("Stage 2: Reward Model Training\n(Bradley-Terry loss)")
    ax.set_xlabel("Step")
    ax.set_ylabel("BT Loss", color="#e74c3c")
    ax2.set_ylabel("Pairwise Accuracy", color="#2ecc71")
    ax2.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    # ── (0,2) PPO reward ──────────────────────────────────────
    ax = axes[0, 2]
    ax.plot(smooth(ppo_history["rm_reward"], w), color="#9b59b6")
    ax.set_title("Stage 3: PPO — RM Reward over Training")
    ax.set_xlabel("PPO step")
    ax.set_ylabel("Mean RM Score")

    # ── (1,0) KL divergence ───────────────────────────────────
    ax = axes[1, 0]
    ax.plot(smooth(ppo_history["kl"], w), color="crimson")
    ax.set_title("KL(π_ref || π_θ) during PPO")
    ax.set_xlabel("PPO step")
    ax.set_ylabel("KL Divergence (nats)")

    # ── (1,1) Policy entropy ─────────────────────────────────
    ax = axes[1, 1]
    ax.plot(smooth(ppo_history["entropy"], w), color="darkorange")
    ax.set_title("Policy Entropy (token-level)")
    ax.set_xlabel("PPO step")
    ax.set_ylabel("Entropy (nats)")
    ax.set_ylim(0)

    # ── (1,2) P(sorted) + equations ──────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    txt = (
        "RLHF Pipeline\n"
        "────────────────────────────────────\n"
        "Stage 1 — SFT:\n"
        "  L_SFT = -E[log π(x | prompt)]\n\n"
        "Stage 2 — Reward Model:\n"
        "  Bradley-Terry:\n"
        "  P(w>l) = σ(r(w) − r(l))\n"
        "  L_RM = -E[log P(preferred > rej)]\n\n"
        "Stage 3 — PPO + KL penalty:\n"
        "  r_eff = r_RM(x) − β·KL(π_ref||π_θ)\n"
        "  L_PPO = E[min(ratio·A, clip·A)]\n\n"
        "  β = KL coefficient\n"
        "  Without β: reward hacking!"
    )
    ax.text(0.05, 0.97, txt, transform=ax.transAxes,
            fontsize=9.5, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fffbe6", alpha=0.9))

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "day3_rlhf_pipeline.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    sft_steps = 100  if args.smoke else 500
    rm_steps  = 100  if args.smoke else 400
    ppo_steps = 100  if args.smoke else 400
    batch     = 32   if args.smoke else 64

    print("=" * 55)
    print("  Day 3 — RLHF Pipeline Internals")
    print(f"  mode={'smoke' if args.smoke else 'full'}")
    print("=" * 55)

    # ── Stage 1: SFT ────────────────────────────────────────
    print("\n[Stage 1] SFT ...")
    policy = TinyRNNPolicy(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN)
    sft_losses = run_sft(policy, sft_steps, batch_size=batch)
    ref_policy = copy.deepcopy(policy)   # freeze SFT weights as reference
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    print(f"  SFT done. Final loss = {sft_losses[-1]:.4f}")

    # ── Stage 2: Reward Model ────────────────────────────────
    print("\n[Stage 2] Reward Model (Bradley-Terry) ...")
    rm = TinyRewardModel(vocab_size=VOCAB_SIZE)
    rm_losses, rm_accs = run_reward_model_training(rm, rm_steps, batch_size=batch)
    print(f"  RM done. Final BT loss = {rm_losses[-1]:.4f}  acc = {rm_accs[-1]:.3f}")

    # ── Stage 3: PPO-RL ──────────────────────────────────────
    print("\n[Stage 3] PPO fine-tuning ...")
    ppo_history = run_ppo_rl(policy, ref_policy, rm, ppo_steps, batch, kl_coef=0.1)

    print("\n  Plotting ...")
    plot_results(sft_losses, rm_losses, rm_accs, ppo_history, ppo_steps)

    print(f"\n  Final RM reward:  {np.mean(ppo_history['rm_reward'][-20:]):.4f}")
    print(f"  Final KL:         {np.mean(ppo_history['kl'][-20:]):.4f}")
    print(f"  Final entropy:    {np.mean(ppo_history['entropy'][-20:]):.4f}")
    print("\n  Day 3 complete.\n")


if __name__ == "__main__":
    main()
