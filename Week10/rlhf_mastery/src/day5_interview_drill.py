"""
day5_interview_drill.py — Numerical Verification of Core RLHF Theorems.
========================================================================

OVERVIEW
--------
This day turns theoretical understanding into *verifiable code*.  For each
core interview theorem, we run a numerical experiment that demonstrates it
is true (or approximately true) on our toy model.

The five theorems demonstrated:
  1. E[∇log π] = 0  (score function has zero expectation)
  2. Baseline theorem: E[∇log π · b(s)] = 0  (baseline doesn't bias gradient)
  3. Policy gradient theorem: ∇J = E[∇log π · G]
  4. PPO vs GRPO comparison on the same task
  5. DPO loss derivation and comparison to PPO

WHY THIS IS INTERVIEW GOLD
--------------------------
Anyone can say "baseline doesn't bias the gradient."
Only someone who has *written it down and verified it* can say:
  "I verified this numerically on a 4-token MDP and the bias was 1e-5."

This is the difference between knowing a fact and understanding it.

THEOREM 1: E[∇log π] = 0
------------------------
For any normalized distribution π:
  E_{x~π}[∇_θ log π_θ(x)] = Σ_x π(x) · ∇log π(x)
                             = Σ_x π(x) · ∇π(x)/π(x)
                             = ∇ Σ_x π(x) = ∇ 1 = 0

This is the 'score function identity'.  It means constant rewards don't
produce a gradient — you must have *differential* rewards to learn.

THEOREM 2: Baseline theorem
---------------------------
E_{x~π}[∇log π(x) · b(s)] = b(s) · E_{x~π}[∇log π(x)]
                            = b(s) · 0
                            = 0

The baseline b(s) (any function of the state, not the action) factors out
of the expectation, and E[∇log π] = 0 kills it.  So subtracted the baseline
doesn't change the *expected* gradient — it only reduces variance.

THEOREM 3: Policy gradient theorem
-----------------------------------
∇J(θ) = ∇ E_{x~π_θ}[r(x)]
       = ∇ Σ_x π_θ(x) r(x)
       = Σ_x [∇π_θ(x)] r(x)
       = Σ_x π_θ(x) · [∇log π_θ(x)] r(x)
       = E_{x~π_θ}[∇log π_θ(x) · r(x)]

THEOREM 4: PPO vs GRPO
-----------------------
PPO: uses value network as baseline, clips ratio π_θ/π_old.
GRPO: uses group mean reward as baseline, clips ratio π_θ/π_old.
  GRPO advantage: A_i = (r_i - mean(r_group)) / std(r_group)
  This is the 'group normalization' that replaces the value network.

THEOREM 5: DPO
--------------
DPO rearranges the RLHF objective to avoid explicit RL:
  L_DPO(θ) = -E[log σ(β·log π_θ(y_w|x)/π_ref(y_w|x)
                  - β·log π_θ(y_l|x)/π_ref(y_l|x))]
where y_w = preferred, y_l = rejected.

OUTPUT
------
results/day5/day5_theorem_verification.png — 5-panel numerical proofs
results/day5/day5_algorithm_comparison.png — PPO vs GRPO vs DPO on same task
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
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    TinyRNNPolicy, make_policy, kl_per_token, token_entropy, smooth,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "day5")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Reward ──────────────────────────────────────────────────────────────────

def shaped_reward(tokens: torch.Tensor) -> torch.Tensor:
    r = torch.zeros(tokens.shape[0])
    r += 2.0 * (tokens[:, 0] == 0).float()
    r += 2.0 * (tokens[:, -1] == 1).float()
    return r


# ─── Exact helpers (key: vocab=8, seq_len=4 → only 4096 sequences) ───────────

def enumerate_sequences(vocab_size: int = 8, seq_len: int = 4) -> torch.Tensor:
    """Return all V^T sequences. Shape: (V^T, T)."""
    ranges = [torch.arange(vocab_size)] * seq_len
    grids  = torch.meshgrid(*ranges, indexing="ij")
    return torch.stack([g.flatten() for g in grids], dim=1)   # (4096, 4)


def exact_expected_reward(policy, all_seqs: torch.Tensor) -> float:
    """J(θ) = Σ_x π_θ(x)·r(x) — exact, no MC noise."""
    with torch.no_grad():
        lp, _ = policy.log_probs_of(all_seqs)
        pis    = lp.sum(1).exp()               # (N,)
        rew    = shaped_reward(all_seqs)       # (N,)
        return (pis * rew).sum().item()


def exact_policy_gradient(policy, all_seqs: torch.Tensor,
                           weights: torch.Tensor) -> torch.Tensor:
    """
    Compute Σ_x weights(x) · π(x).detach() · ∇log π(x)  (exact, no MC).

    With weights=1:   gives E[∇log π] = 0  (score-function identity).
    With weights=r:   gives E[∇log π · r]  = ∇J  (policy gradient).
    """
    lp, _ = policy.log_probs_of(all_seqs)
    pis    = lp.sum(1).exp().detach()          # π(x) — frozen weights
    # ∇(Σ_x π(x).detach() · w(x) · log π(x)) = Σ_x π(x)·w(x)·∇log π(x)
    policy.zero_grad()
    lp2, _ = policy.log_probs_of(all_seqs)
    (pis * weights * lp2.sum(1)).sum().backward()
    g = torch.cat([p.grad.flatten() for p in policy.parameters()
                   if p.grad is not None]).detach().clone()
    return g


# ─── Theorem 1: E[∇log π] = 0 ───────────────────────────────────────────────

def verify_score_function_identity(n_samples: int = 5000) -> Dict:
    """
    Verify: E_{x~π}[∇_θ log π_θ(x)] = 0  (score-function identity).

    APPROACH: Exact enumeration over all 8^4 = 4096 sequences.
      - E[∇log π] is computed as Σ_x π(x)·∇log π(x) with π(x) detached.
      - This collapses to ∇1 = 0 — verified to machine precision.
      - Compare against E[∇log π · r] = ∇J ≠ 0 (real learning signal).

    Why exact rather than MC?
      MC estimates have noise O(σ/√n); the exact approach gives the
      true zero directly, proving the identity rather than approximating it.
    """
    policy   = make_policy("small")
    all_seqs = enumerate_sequences(policy.vocab_size, policy.seq_len)  # (4096, 4)
    rewards  = shaped_reward(all_seqs)                                   # (4096,)

    # Exact E[∇log π] = Σ_x π(x)·∇log π(x)  — should be ≈ machine eps
    g_score     = exact_policy_gradient(policy, all_seqs,
                                        weights=torch.ones(len(all_seqs)))

    # Exact E[∇log π · r] = Σ_x π(x)·r(x)·∇log π(x) = ∇J  — should be ≠ 0
    g_reinforce = exact_policy_gradient(policy, all_seqs, weights=rewards)

    mean_score_norm     = g_score.norm().item()
    mean_reinforce_norm = g_reinforce.norm().item()

    return {
        "mean_score_norm":       mean_score_norm,
        "mean_reinforce_norm":   mean_reinforce_norm,
        "ratio_score_reinforce": mean_score_norm / (mean_reinforce_norm + 1e-8),
        "interpretation":   "‖E[∇log π]‖ / ‖E[∇log π · r]‖ (exact, no MC noise).  "
                            "Score gradient = ∇1 = 0; REINFORCE has real signal.",
    }


# ─── Theorem 2: Baseline does not bias gradient ──────────────────────────────

def verify_baseline_theorem(
    n_trials:  int = 100,
    n_samples: int = 256,
) -> Dict:
    """
    Compare gradient estimates with and without a baseline.

    CRITICAL: both estimates must use the SAME tokens for bias comparison.

    True gradient:   g1 = ∇log π · r(x)
    With baseline:   g2 = ∇log π · (r(x) - b)

    On the same batch: g1 - g2 = ∇log π · b = constant × ∇log π.
    Averaged over many batches: E[g1] - E[g2] = b · E[∇log π] = 0.

    KEY: baseline must be b ≈ E[r] to REDUCE variance.
    Using b >> E[r] (e.g., 2.0 when E[r] ≈ 0.5) INCREASES variance.
    We compute the exact E[r] via enumeration for an optimal baseline.
    """
    policy   = make_policy("small")
    all_seqs = enumerate_sequences(policy.vocab_size, policy.seq_len)

    # Exact E[r] under the initial policy — the optimal variance-reducing baseline
    baseline = exact_expected_reward(policy, all_seqs)

    grads_no_baseline   = []
    grads_with_baseline = []

    for _ in range(n_trials):
        # Sample ONCE, use for BOTH estimates
        tokens, log_probs, _ = policy.sample_sequence(n_samples)
        rewards = shaped_reward(tokens)
        seq_lp  = log_probs.sum(dim=1)

        # --- No baseline ---
        loss_nb = -(seq_lp * rewards).mean()
        policy.zero_grad()
        loss_nb.backward(retain_graph=True)
        g_nb = torch.cat([p.grad.flatten() for p in policy.parameters()
                          if p.grad is not None]).detach().clone()
        grads_no_baseline.append(g_nb)

        # --- With baseline (same tokens, same graph) ---
        adv     = rewards - baseline
        loss_wb = -(seq_lp * adv).mean()
        policy.zero_grad()
        loss_wb.backward()
        g_wb = torch.cat([p.grad.flatten() for p in policy.parameters()
                          if p.grad is not None]).detach().clone()
        grads_with_baseline.append(g_wb)

    nb_stack = torch.stack(grads_no_baseline)   # (n_trials, n_params)
    wb_stack = torch.stack(grads_with_baseline)

    mean_nb = nb_stack.mean(dim=0)
    mean_wb = wb_stack.mean(dim=0)
    # MC bias estimate (converges to 0 as n_trials → ∞)
    mc_bias     = (mean_nb - mean_wb).norm().item()
    mc_rel_bias = mc_bias / (mean_nb.norm().item() + 1e-8)

    # EXACT bias = ‖b · E[∇log π]‖ = b × ‖exact score grad‖ ≈ 0 to machine eps
    all_seqs    = enumerate_sequences(policy.vocab_size, policy.seq_len)
    exact_score = exact_policy_gradient(policy, all_seqs,
                                        weights=torch.ones(len(all_seqs)))
    exact_bias  = (baseline * exact_score.norm()).item()   # = 0 exactly

    var_nb  = nb_stack.var(dim=0).mean().item()
    var_wb  = wb_stack.var(dim=0).mean().item()
    var_red = 1 - var_wb / (var_nb + 1e-12)

    return {
        "baseline_value":         baseline,
        "exact_bias":             exact_bias,    # provably 0 by theorem
        "mc_relative_bias":       mc_rel_bias,   # MC noise, decreases with n_trials
        "variance_no_baseline":   var_nb,
        "variance_with_baseline": var_wb,
        "variance_reduction":     var_red,       # positive = baseline helps
        "interpretation":   f"Exact E[r]={baseline:.4f} used as baseline.  "
                            f"Exact bias={exact_bias:.2e} (theorem: = 0).  "
                            f"MC relative bias={mc_rel_bias:.4f} (noise, not true bias).  "
                            f"Variance {'reduced' if var_red > 0 else 'increased'} "
                            f"by {abs(var_red)*100:.1f}%.",
    }


# ─── Theorem 3: Policy gradient theorem ─────────────────────────────────────

def verify_policy_gradient_theorem(epsilon: float = 1e-4) -> Dict:
    """
    Compare analytic gradient (via autodiff) with numerical finite difference.

    Finite difference: ∂J/∂θ_i ≈ [J(θ + ε·e_i) - J(θ - ε·e_i)] / 2ε

    APPROACH: Use exact J(θ) = Σ_x π_θ(x)·r(x) via full enumeration.
    This eliminates MC noise entirely, making the comparison exact.
    Both analytic and numerical gradients should agree to high precision.
    """
    policy   = make_policy("small")
    all_seqs = enumerate_sequences(policy.vocab_size, policy.seq_len)  # (4096, 4)
    rewards  = shaped_reward(all_seqs)                                   # (4096,)

    # --- Exact analytic gradient: ∇J = Σ_x π(x)·r(x)·∇log π(x) ---
    analytic_grad = exact_policy_gradient(policy, all_seqs, weights=rewards)

    # --- Exact numerical gradient (finite difference on first 20 params) ---
    n_check    = 20
    num_grad   = torch.zeros(n_check)
    all_params = [p for p in policy.parameters() if p.requires_grad]

    idx = 0
    for p in all_params:
        if idx >= n_check:
            break
        n_this = min(p.numel(), n_check - idx)
        orig   = p.data.clone()
        p_flat = p.data.clone().flatten()

        for i in range(n_this):
            # J(θ + ε)
            p_flat_p = p_flat.clone(); p_flat_p[i] += epsilon
            p.data = p_flat_p.reshape(p.shape)
            j_plus = exact_expected_reward(policy, all_seqs)

            # J(θ - ε)
            p_flat_m = p_flat.clone(); p_flat_m[i] -= epsilon
            p.data = p_flat_m.reshape(p.shape)
            j_minus = exact_expected_reward(policy, all_seqs)

            num_grad[idx + i] = (j_plus - j_minus) / (2 * epsilon)
            p.data = orig   # restore

        idx += n_this
        if idx >= n_check:
            break

    analytic_subset = analytic_grad[:n_check]
    cos_sim = F.cosine_similarity(
        analytic_subset.unsqueeze(0),
        num_grad.unsqueeze(0),
    ).item()

    return {
        "analytic_grad_norm":  analytic_grad.norm().item(),
        "numerical_grad_norm": num_grad.norm().item(),
        "cosine_similarity":   cos_sim,
        "interpretation":      f"Cosine similarity={cos_sim:.3f} (>0.9 expected — exact J, no MC noise).  "
                               "Both gradients use full enumeration of all 4096 sequences.",
    }


# ─── Theorem 4: PPO vs GRPO ──────────────────────────────────────────────────

def run_ppo(n_steps: int, batch_size: int, lr: float, group_size: int = 8):
    """Standard PPO with value-network baseline."""
    from common import TinyValueNet

    policy     = make_policy("small")
    ref_policy = deepcopy(policy); ref_policy.eval()
    value_net  = TinyValueNet(hidden_dim=policy.hidden_dim)
    opt_policy = torch.optim.Adam(policy.parameters(), lr=lr)
    opt_value  = torch.optim.Adam(value_net.parameters(), lr=lr * 3)

    history = {"reward": [], "kl": [], "entropy": []}
    CLIP_EPS = 0.2

    for step in range(n_steps):
        policy.train()
        tokens, old_lp, hiddens = policy.sample_sequence(batch_size)
        rewards  = shaped_reward(tokens)

        # Value estimate from last hidden state
        vals = value_net(hiddens[:, -1, :]).detach()   # (B,)
        adv  = rewards - vals
        adv  = (adv - adv.mean()) / (adv.std() + 1e-6)

        # PPO update
        new_lp, new_h = policy.log_probs_of(tokens)
        ratio = torch.exp(new_lp.sum(1) - old_lp.sum(1).detach())
        clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
        policy_loss = -torch.min(ratio * adv, clipped * adv).mean()

        opt_policy.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt_policy.step()

        # Detach hidden states so value loss doesn't try to backward
        # through a graph that was already freed by policy_loss.backward()
        new_vals = value_net(new_h[:, -1, :].detach())
        value_loss = F.mse_loss(new_vals, rewards)
        opt_value.zero_grad()
        value_loss.backward()
        opt_value.step()

        with torch.no_grad():
            kl  = kl_per_token(policy, ref_policy, tokens)
            ent = token_entropy(policy)
        history["reward"].append(rewards.mean().item())
        history["kl"].append(kl)
        history["entropy"].append(ent)

    return history


def run_grpo(n_steps: int, batch_size: int, lr: float, group_size: int = 8):
    """
    GRPO: sample G sequences for the same 'prompt', normalise advantage within group.

    In our toy: 'prompt' = the BOS token (there's only one prompt).
    The group advantage is: A_i = (r_i - mean(group)) / (std(group) + ε)

    No value network needed.
    """
    policy     = make_policy("small")
    ref_policy = deepcopy(policy); ref_policy.eval()
    optimizer  = torch.optim.Adam(policy.parameters(), lr=lr)

    history = {"reward": [], "kl": [], "entropy": []}
    CLIP_EPS = 0.2
    n_groups = batch_size // group_size

    for step in range(n_steps):
        policy.train()
        all_tokens = []
        all_lp     = []
        all_adv    = []

        for g in range(n_groups):
            tokens, lp, _ = policy.sample_sequence(group_size)
            rewards = shaped_reward(tokens)
            # Group-normalised advantage
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
            all_tokens.append(tokens)
            all_lp.append(lp)
            all_adv.append(adv)

        tokens  = torch.cat(all_tokens)
        old_lp  = torch.cat(all_lp).detach()
        adv_cat = torch.cat(all_adv)

        # PPO-clip with group advantages
        new_lp, _ = policy.log_probs_of(tokens)
        ratio = torch.exp(new_lp.sum(1) - old_lp.sum(1))
        clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
        loss = -torch.min(ratio * adv_cat, clipped * adv_cat).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            kl  = kl_per_token(policy, ref_policy, tokens)
            ent = token_entropy(policy)
        history["reward"].append(shaped_reward(tokens).mean().item())
        history["kl"].append(kl)
        history["entropy"].append(ent)

    return history


def run_dpo(n_steps: int, batch_size: int, lr: float, beta: float = 0.1):
    """
    DPO: direct preference optimisation.

    We simulate preference pairs:
      y_w = sequence starting with token 0 (preferred)
      y_l = sequence starting with token 7 (rejected)

    DPO loss:
      L = -E[log σ(β·(log π_θ(y_w)/π_ref(y_w) - log π_θ(y_l)/π_ref(y_l)))]

    This is *offline* RL — no rollouts during training, just pair comparisons.
    """
    policy     = make_policy("small")
    ref_policy = deepcopy(policy); ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    history   = {"reward": [], "kl": [], "entropy": []}

    # Pre-generate preference pairs (offline)
    def make_pairs(n: int):
        # Preferred: token 0 at position 0 and token 1 at position -1
        yw = torch.zeros(n, 4, dtype=torch.long)
        yw[:, 0]  = 0
        yw[:, -1] = 1
        # Fill middle with random tokens
        yw[:, 1:-1] = torch.randint(0, 8, (n, 2))

        # Rejected: token 7 at position 0
        yl = torch.zeros(n, 4, dtype=torch.long)
        yl[:, 0]  = 7
        yl[:, 1:] = torch.randint(0, 8, (n, 3))
        return yw, yl

    for step in range(n_steps):
        policy.train()
        yw, yl = make_pairs(batch_size)

        lp_w, _ = policy.log_probs_of(yw)
        lp_l, _ = policy.log_probs_of(yl)
        with torch.no_grad():
            ref_lp_w, _ = ref_policy.log_probs_of(yw)
            ref_lp_l, _ = ref_policy.log_probs_of(yl)

        # Log-ratios
        log_ratio_w = (lp_w.sum(1) - ref_lp_w.sum(1))   # (B,)
        log_ratio_l = (lp_l.sum(1) - ref_lp_l.sum(1))   # (B,)

        # DPO loss
        logit = beta * (log_ratio_w - log_ratio_l)
        loss  = -F.logsigmoid(logit).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        # Evaluate policy quality via rollout
        with torch.no_grad():
            toks, _, _ = policy.sample_sequence(batch_size)
            r   = shaped_reward(toks).mean().item()
            kl  = kl_per_token(policy, ref_policy, toks)
            ent = token_entropy(policy)
        history["reward"].append(r)
        history["kl"].append(kl)
        history["entropy"].append(ent)

    return history


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_theorem_verification(results: Dict):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Theorem 1: Score function identity
    ax = axes[0, 0]
    r1 = results["theorem1"]
    ax.bar(["Score (const reward)", "REINFORCE (real reward)"],
           [r1["mean_score_norm"], r1["mean_reinforce_norm"]],
           color=["steelblue", "darkorange"])
    ax.set_title("Thm 1: E[∇log π] ≈ 0\n(Score Function Identity)")
    ax.set_ylabel("‖gradient‖")
    ax.text(0.5, 0.95,
            f"Score norm: {r1['mean_score_norm']:.4f}\n"
            f"REINFORCE norm: {r1['mean_reinforce_norm']:.4f}\n"
            f"Ratio: {r1['ratio_score_reinforce']:.3f} (→ 0 for large N)",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            bbox=dict(facecolor="lightyellow", alpha=0.8))

    # Theorem 2: Baseline theorem
    ax = axes[0, 1]
    r2 = results["theorem2"]
    ax.bar(["No baseline", "With baseline"],
           [r2["variance_no_baseline"], r2["variance_with_baseline"]],
           color=["crimson", "seagreen"])
    ax.set_title("Thm 2: Baseline Reduces Variance\n(No Bias Introduced)")
    ax.set_ylabel("Gradient Variance (per param)")
    ax.text(0.5, 0.95,
            f"Exact bias: {r2['exact_bias']:.2e} (= 0 by proof)\n"
            f"Variance change: {r2['variance_reduction']*100:.1f}%",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            bbox=dict(facecolor="lightyellow", alpha=0.8))

    # Theorem 3: Policy gradient
    ax = axes[0, 2]
    r3 = results["theorem3"]
    ax.bar(["Analytic (REINFORCE)", "Numerical (FD)"],
           [r3["analytic_grad_norm"], r3["numerical_grad_norm"]],
           color=["steelblue", "darkorange"])
    ax.set_title("Thm 3: Policy Gradient Theorem\nAnalytic vs Numerical")
    ax.set_ylabel("Gradient Norm (first 20 params)")
    ax.text(0.5, 0.95,
            f"Cosine similarity: {r3['cosine_similarity']:.3f}\n"
            "(>0.9 = exact match, no MC noise)",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            bbox=dict(facecolor="lightyellow", alpha=0.8))

    # Algorithm comparison
    for i, (key, label, color) in enumerate([
        ("ppo",  "PPO",  "steelblue"),
        ("grpo", "GRPO", "darkorange"),
        ("dpo",  "DPO",  "seagreen"),
    ]):
        hist = results[key]
        n    = len(hist["reward"])
        w    = max(1, n // 15)

        axes[1, 0].plot(smooth(hist["reward"],  w=w), label=label, color=color, alpha=0.85)
        axes[1, 1].plot(smooth(hist["kl"],      w=w), label=label, color=color, alpha=0.85)
        axes[1, 2].plot(smooth(hist["entropy"], w=w), label=label, color=color, alpha=0.85)

    axes[1, 0].set_title("Thm 4/5: PPO vs GRPO vs DPO — Reward")
    axes[1, 0].set_ylabel("Mean Reward"); axes[1, 0].legend(fontsize=9)
    axes[1, 1].set_title("KL Divergence"); axes[1, 1].set_ylabel("KL(π||π_ref)")
    axes[1, 1].legend(fontsize=9)
    axes[1, 2].set_title("Token Entropy"); axes[1, 2].set_ylabel("H(π) nats")
    axes[1, 2].legend(fontsize=9)

    for ax in axes.flatten():
        ax.set_xlabel("Step")

    fig.suptitle("Day 5: Numerical Verification of Core RLHF Theorems",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "day5_theorem_verification.png")
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
        batch_size = 64
        n_samples  = 500
        n_trials   = 20
    else:
        n_steps    = 400
        batch_size = 128
        n_samples  = 5000
        n_trials   = 100

    torch.manual_seed(42)
    np.random.seed(42)
    lr = 1e-3

    print("=" * 60)
    print(" Day 5: Interview Drill — Numerical Theorem Verification")
    print("=" * 60)
    print()

    results = {}

    print("[Theorem 1] Verifying E[∇log π] = 0 ...")
    results["theorem1"] = verify_score_function_identity(n_samples)
    r1 = results["theorem1"]
    print(f"  Score norm:     {r1['mean_score_norm']:.6f}  (should be small)")
    print(f"  REINFORCE norm: {r1['mean_reinforce_norm']:.6f}  (should be large)")
    print(f"  Ratio (lower=better): {r1['ratio_score_reinforce']:.4f}")
    print(f"  {r1['interpretation']}")
    print()

    print("[Theorem 2] Verifying baseline doesn't bias gradient ...")
    results["theorem2"] = verify_baseline_theorem(n_trials, n_samples // 10)
    r2 = results["theorem2"]
    print(f"  Baseline (exact E[r]):  {r2['baseline_value']:.4f}")
    print(f"  Exact bias (theorem):   {r2['exact_bias']:.2e}  (= 0 by proof)")
    print(f"  MC relative bias:       {r2['mc_relative_bias']:.6f}  (MC noise, not true bias)")
    print(f"  Variance change: {r2['variance_reduction']*100:.1f}%")
    print(f"  {r2['interpretation']}")
    print()

    print("[Theorem 3] Verifying policy gradient theorem ...")
    results["theorem3"] = verify_policy_gradient_theorem()
    r3 = results["theorem3"]
    print(f"  Cosine sim (analytic vs numeric): {r3['cosine_similarity']:.3f}")
    print(f"  {r3['interpretation']}")
    print()

    print("[Theorem 4] Running PPO ...")
    results["ppo"]  = run_ppo(n_steps, batch_size, lr)
    print(f"  PPO  final reward: {np.mean(results['ppo']['reward'][-20:]):.3f}")

    print("[Theorem 4] Running GRPO ...")
    results["grpo"] = run_grpo(n_steps, batch_size, lr, group_size=8)
    print(f"  GRPO final reward: {np.mean(results['grpo']['reward'][-20:]):.3f}")

    print("[Theorem 5] Running DPO ...")
    results["dpo"]  = run_dpo(n_steps, batch_size, lr)
    print(f"  DPO  final reward: {np.mean(results['dpo']['reward'][-20:]):.3f}")
    print()

    print("Generating plots...")
    plot_theorem_verification(results)

    # Final summary
    print()
    print("INTERVIEW ANSWERS (based on numerical verification):")
    print("-" * 60)
    print()
    print("Q: Why is E[∇log π] = 0?")
    r1 = results['theorem1']
    print(f"   A: Score norm / REINFORCE norm = {r1['ratio_score_reinforce']:.4f} (≈ 0 theoretically).")
    print("      Proof: Σ_x π(x)·∇log π(x) = Σ_x ∇π(x) = ∇Σ_x π(x) = ∇1 = 0")
    print()
    print("Q: Why doesn't subtracting the baseline bias the gradient?")
    r2 = results['theorem2']
    print(f"   A: Exact bias = {r2['exact_bias']:.2e} (provably 0: b·E[∇log π] = b·0 = 0).")
    print(f"      Baseline = exact E[r] = {r2['baseline_value']:.4f}.")
    print(f"      Variance changed by {r2['variance_reduction']*100:.1f}% — essentially free variance reduction.")
    print()
    print("Q: Compare PPO vs GRPO vs DPO:")
    for key in ["ppo", "grpo", "dpo"]:
        h = results[key]
        print(f"   {key.upper():4s}: reward={np.mean(h['reward'][-20:]):.2f} "
              f"kl={np.mean(h['kl'][-20:]):.2f} ent={np.mean(h['entropy'][-20:]):.2f}")
    print("      PPO:  online, value net baseline, clip")
    print("      GRPO: online, group baseline (no value net), clip")
    print("      DPO:  offline, no rollouts, preference pairs only")
    print()
    print("Day 5 complete.")


if __name__ == "__main__":
    main()
