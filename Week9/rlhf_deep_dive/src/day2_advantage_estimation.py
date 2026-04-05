"""
Day 2 — Advantage Estimation Deep Dive
========================================
Environment : "Spell" task — the policy generates a 4-token sequence.
              Reward at step t: +1 if token == t, else 0.
              (i.e. the optimal sequence is [0, 1, 2, 3].)

We compare three ways to estimate the gradient signal:
  1. Raw return    G_t = Σ_{k≥t} γ^{k-t} r_k
  2. Centred       G_t − mean_batch(G_t)
  3. GAE           A_t^GAE(λ) using a learned value function V_φ(s_t)

Key insight: subtracting any baseline b(s_t) does NOT change the expected
gradient (because E[∇ log π · b(s)] = 0), but it can dramatically reduce
variance, making learning faster and more stable.

Run
---
  python day2_advantage_estimation.py
  python day2_advantage_estimation.py --smoke
"""

import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Config ───────────────────────────────────────────────────────────────────
VOCAB_SIZE  = 8     # token IDs 0..7
SEQ_LEN     = 4     # episode length
GAMMA       = 0.99
LAM         = 0.95  # GAE λ  (0 = low-var/high-bias, 1 = high-var/low-bias)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "day2")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Environment ──────────────────────────────────────────────────────────────

def spell_reward(tokens: torch.Tensor) -> torch.Tensor:
    """
    Per-step reward: +1 if tokens[t] == t else 0.
    tokens : (B, T) int64
    Returns: (B, T) float32
    """
    B, T  = tokens.shape
    target = torch.arange(T).unsqueeze(0).expand(B, -1)  # (B, T)
    return (tokens == target).float()


# ─── Shared stateless policy for Day 2 ───────────────────────────────────────
# We use independent per-position logits for clarity; this makes the value
# function V(s_t) depend only on step t, which is easy to learn.

class PositionPolicy(nn.Module):
    """
    Independent softmax at each position.
    logits : (T, V)  — one distribution per step.
    """
    def __init__(self, vocab_size: int = VOCAB_SIZE, seq_len: int = SEQ_LEN):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(seq_len, vocab_size))
        self.seq_len = seq_len

    def sample_batch(self, batch_size: int):
        """Returns tokens (B,T), log_probs (B,T)."""
        probs = F.softmax(self.logits, dim=-1)      # (T, V)
        dist  = torch.distributions.Categorical(probs)
        tokens    = dist.sample((batch_size,))      # (B, T)
        log_probs = dist.log_prob(tokens)           # (B, T)
        return tokens, log_probs

    def log_probs_of(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B,T). Returns (B,T)."""
        lp = F.log_softmax(self.logits, dim=-1)     # (T, V)
        return lp[torch.arange(self.seq_len), tokens.T].T  # (B, T)


class StepValueNet(nn.Module):
    """V(s_t) — here s_t = step index, so this is just a learnable bias per step.
    We store seq_len+1 values: V(0) ... V(T), where V(T)=0 is the terminal state."""
    def __init__(self, seq_len: int = SEQ_LEN):
        super().__init__()
        # +1 for terminal state V(T); kept at 0 by not training index T directly
        self.values = nn.Parameter(torch.zeros(seq_len + 1))

    def forward(self, step: int) -> torch.Tensor:
        return self.values[step]  # scalar


# ─── Return / Advantage computations ─────────────────────────────────────────

def mc_returns(step_rewards: torch.Tensor, gamma: float = GAMMA) -> torch.Tensor:
    """
    Monte-Carlo returns.
    step_rewards: (B, T)
    Returns:      (B, T)  G_t = Σ_{k≥t} γ^{k-t} r_k
    """
    B, T = step_rewards.shape
    G = torch.zeros_like(step_rewards)
    g = torch.zeros(B)
    for t in reversed(range(T)):
        g    = step_rewards[:, t] + gamma * g
        G[:, t] = g
    return G


def gae_advantages(
    step_rewards: torch.Tensor,
    value_net: StepValueNet,
    gamma: float = GAMMA,
    lam: float   = LAM,
) -> torch.Tensor:
    """
    GAE advantages for a batch of episodes.

    step_rewards: (B, T)
    Returns:      (B, T)  A_t^GAE
    """
    B, T = step_rewards.shape
    # V(s_t) = value_net(t)  (same for all episodes, since state = step index)
    values = torch.stack([value_net(t) for t in range(T + 1)])  # (T+1,)  V(0)..V(T)
    # V(terminal) = 0 by convention
    values[-1] = 0.0

    A_batch = torch.zeros(B, T)
    for b in range(B):
        rewards = step_rewards[b]   # (T,)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae   = delta + gamma * lam * gae
            A_batch[b, t] = gae
    return A_batch


# ─── Training runs ────────────────────────────────────────────────────────────

def train(
    method: str,
    n_steps: int,
    batch_size: int,
    lr_policy: float,
    lr_value:  float = 0.05,
    n_grad_samples: int = 10,
) -> dict:
    """
    Train the PositionPolicy using one of:
      'raw'      — raw Monte-Carlo returns G_t
      'centred'  — G_t − mean(G_t)
      'gae'      — GAE with a learned value baseline
    Tracks:
      - mean episode reward
      - gradient variance (std of ∂L/∂logits across n_grad_samples mini-batches)
      - P(optimal token) at each position
    """
    policy    = PositionPolicy()
    opt_p     = torch.optim.Adam(policy.parameters(), lr=lr_policy)

    value_net = StepValueNet() if method == "gae" else None
    opt_v     = torch.optim.Adam(value_net.parameters(), lr=lr_value) if value_net else None

    history = {
        "reward": [],
        "grad_var": [],
        "p_optimal": [],   # P(token_t == t), averaged over positions
    }

    for step in range(n_steps):
        tokens, log_probs = policy.sample_batch(batch_size)
        rewards           = spell_reward(tokens)            # (B, T)
        ep_reward         = rewards.sum(dim=1).mean().item()

        # ── Compute advantage signal ──────────────────────────
        if method == "raw":
            G = mc_returns(rewards)          # (B, T)
            adv = G

        elif method == "centred":
            G   = mc_returns(rewards)        # (B, T)
            # Subtract per-step mean across batch
            adv = G - G.mean(dim=0, keepdim=True)

        elif method == "gae":
            adv = gae_advantages(rewards, value_net)  # (B, T)
            # Also train value net with MSE loss on returns
            G   = mc_returns(rewards)
            V_pred = torch.stack([value_net(t) for t in range(SEQ_LEN)]).unsqueeze(0).expand(batch_size, -1)
            v_loss = F.mse_loss(V_pred, G.detach())
            opt_v.zero_grad()
            v_loss.backward()
            opt_v.step()

        else:
            raise ValueError(f"Unknown method: {method}")

        # Normalise advantages
        if adv.std() > 1e-6:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ── Policy gradient loss ──────────────────────────────
        # L = -E[ log π(a) · A ]
        loss = -(log_probs * adv.detach()).mean()
        opt_p.zero_grad()
        loss.backward()

        # Capture gradient variance across multiple mini-batches
        if step % max(1, n_steps // n_grad_samples) == 0:
            grad_samples = []
            for _ in range(8):
                t2, lp2 = policy.sample_batch(batch_size)
                r2       = spell_reward(t2)
                if method == "raw":
                    a2 = mc_returns(r2)
                elif method == "centred":
                    G2 = mc_returns(r2)
                    a2 = G2 - G2.mean(dim=0, keepdim=True)
                else:
                    a2 = gae_advantages(r2, value_net)
                if a2.std() > 1e-6:
                    a2 = (a2 - a2.mean()) / (a2.std() + 1e-8)
                l2 = -(lp2 * a2.detach()).mean()
                policy.zero_grad()
                l2.backward()
                grad_samples.append(policy.logits.grad.clone().flatten())
            grad_mat = torch.stack(grad_samples)   # (8, T*V)
            gvar = grad_mat.var(dim=0).mean().item()
            history["grad_var"].append(gvar)

        opt_p.step()

        # P(optimal): for each position t, P(token == t)
        with torch.no_grad():
            probs     = F.softmax(policy.logits, dim=-1)  # (T, V)
            p_optimal = probs[torch.arange(SEQ_LEN), torch.arange(SEQ_LEN)].mean().item()

        history["reward"].append(ep_reward)
        history["p_optimal"].append(p_optimal)

    return history


# ─── Theory demonstration: baseline doesn't change E[∇ log π · b] ─────────────

def demonstrate_baseline_theorem(n_samples: int = 5000) -> dict:
    """
    Empirically verify:  E[ ∇_θ log π(a) · b ] = 0  for any b.

    The GRADIENT of log π(a_k) w.r.t. logit w_j is:
        ∂ log π(a_k) / ∂ w_j = 1_{j==k} - π(j)

    We compute this gradient vector for each sampled action and check that:
      - E[gradient · R]     ≈  E[gradient · (R − b)]   (same expected gradient)
      - Var[gradient · R]   >  Var[gradient · (R − b)]  (lower variance with good b)

    We use varying per-action rewards and b ≈ E[R] to make variance reduction visible.
    """
    rng    = np.random.default_rng(42)
    logits = torch.tensor([0.2, -0.5, 1.1, -0.3])
    probs  = F.softmax(logits, dim=0)

    # Per-action rewards: vary so that the baseline can actually help
    true_rewards = [0.1, 0.4, 0.8, 0.3]
    b            = 0.5   # approximate E_π[R]  (reduces variance when b ≈ E[R])

    sum_g_raw      = torch.zeros(4)
    sum_g_centered = torch.zeros(4)
    norms_raw, norms_centered = [], []

    for _ in range(n_samples):
        a = torch.distributions.Categorical(probs).sample().item()
        r = true_rewards[a] + rng.normal(0, 0.1)   # add noise

        # gradient direction: ∂ log π(a) / ∂ w_j = 1_{j=a} - π(j)
        e_a   = torch.zeros(4); e_a[a] = 1.0
        g_dir = e_a - probs                         # (4,)

        g_raw      = float(r)       * g_dir
        g_centered = float(r - b)   * g_dir

        sum_g_raw      += g_raw      / n_samples
        sum_g_centered += g_centered / n_samples
        norms_raw.append(g_raw.norm().item())
        norms_centered.append(g_centered.norm().item())

    return {
        "mean_raw":       sum_g_raw.norm().item(),       # ≈ same
        "mean_centered":  sum_g_centered.norm().item(),  # ≈ same
        "std_raw":        float(np.std(norms_raw)),
        "std_centered":   float(np.std(norms_centered)), # should be smaller
        "baseline_b":     b,
    }


# ─── Visualise ────────────────────────────────────────────────────────────────

def smooth(x, w: int = 15):
    arr = np.asarray(x, dtype=float)
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def plot_results(histories: dict, theorem: dict, n_steps: int):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Day 2 — Advantage Estimation Deep Dive", fontsize=13, fontweight="bold")

    colors = {"raw": "#e74c3c", "centred": "#3498db", "gae": "#2ecc71"}
    w = max(1, n_steps // 40)

    # ── (0,0) Reward curves ──────────────────────────────────
    ax = axes[0, 0]
    for name, h in histories.items():
        ax.plot(smooth(h["reward"], w), label=name, color=colors[name], alpha=0.9)
    ax.axhline(SEQ_LEN, ls="--", c="gray", lw=1, label=f"Optimal R={SEQ_LEN}")
    ax.set_title("Episode Return (max possible = 4)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Return")
    ax.set_ylim(0, SEQ_LEN + 0.2)
    ax.legend(fontsize=8)

    # ── (0,1) Gradient variance ───────────────────────────────
    ax = axes[0, 1]
    for name, h in histories.items():
        ax.plot(h["grad_var"], label=name, color=colors[name], alpha=0.9, marker="o", ms=4)
    ax.set_title("Gradient Variance over Training")
    ax.set_xlabel("Measurement point")
    ax.set_ylabel("Var[ ∂L/∂θ ]")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    # ── (0,2) P(optimal token) ────────────────────────────────
    ax = axes[0, 2]
    for name, h in histories.items():
        ax.plot(smooth(h["p_optimal"], w), label=name, color=colors[name], alpha=0.9)
    ax.axhline(1.0, ls="--", c="gray", lw=1)
    ax.set_title("P(Optimal Token) per Position")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean probability")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    # ── (1,0) Baseline theorem demo ───────────────────────────
    ax = axes[1, 0]
    means = [theorem["mean_raw"], theorem["mean_centered"]]
    stds  = [theorem["std_raw"],  theorem["std_centered"]]
    xs    = np.arange(2)
    bars  = ax.bar(xs, means, yerr=stds, width=0.4,
                   color=["#e74c3c", "#3498db"], capsize=6, alpha=0.8)
    ax.axhline(0, c="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Raw  E[∇logπ·R]", f"Centred  E[∇logπ·(R−{theorem['baseline_b']})]"])
    ax.set_title(f"E[∇logπ·b] = 0  (baseline theorem)\nn={5000} samples")
    ax.set_ylabel("Mean gradient estimate")
    ax.text(0, means[0] + stds[0]*0.1, f"{means[0]:.4f}", ha="center", fontsize=9)
    ax.text(1, means[1] + stds[1]*0.1, f"{means[1]:.4f}", ha="center", fontsize=9)

    # ── (1,1) GAE bias-variance tradeoff ─────────────────────
    ax = axes[1, 1]
    lambdas = [0.0, 0.3, 0.5, 0.8, 0.95, 1.0]
    labels  = ["λ=0\n(TD, high bias)\nlow var",
               "λ=0.3", "λ=0.5", "λ=0.8",
               "λ=0.95\n(GAE default)", "λ=1\n(MC, low bias)\nhigh var"]
    # Illustrative (not measured): higher λ = more MC, lower bias but higher variance
    bias_illus = [0.8, 0.55, 0.4, 0.25, 0.1, 0.0]
    var_illus  = [0.05, 0.15, 0.3, 0.5, 0.75, 1.0]
    xs = np.array(lambdas)
    ax.plot(xs, bias_illus, "o-", label="Bias (↓ good)",     color="#e74c3c")
    ax.plot(xs, var_illus,  "s--", label="Variance (↓ good)", color="#3498db")
    ax.axvline(0.95, c="green", ls=":", lw=1.5, label="Typical λ=0.95")
    ax.set_title("GAE Bias–Variance Tradeoff vs λ")
    ax.set_xlabel("GAE λ")
    ax.set_ylabel("Relative magnitude (normalised)")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8)

    # ── (1,2) Key equations ──────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    txt = (
        "Key Equations\n"
        "────────────────────────────────────\n"
        "Monte-Carlo return:\n"
        "  G_t = Σ_{k≥t} γ^{k-t} r_k\n\n"
        "Baseline theorem:\n"
        "  E[∇ log π(a|s) · b(s)] = 0\n"
        "  → subtracting b does NOT\n"
        "    change expected gradient\n\n"
        "TD error:\n"
        "  δ_t = r_t + γV(s_{t+1}) − V(s_t)\n\n"
        "GAE:\n"
        "  A_t^GAE = Σ_{l≥0} (γλ)^l · δ_{t+l}\n"
        "  λ→0: low var, high bias (TD)\n"
        "  λ→1: high var, low bias (MC)"
    )
    ax.text(0.05, 0.97, txt, transform=ax.transAxes,
            fontsize=9.5, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fffbe6", alpha=0.9))

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "day2_advantage_estimation.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    n_steps    = 200  if args.smoke else 1000
    batch_size = 32   if args.smoke else 128
    lr_policy  = 0.02

    print("=" * 55)
    print("  Day 2 — Advantage Estimation Deep Dive")
    print(f"  mode={'smoke' if args.smoke else 'full'}  steps={n_steps}  batch={batch_size}")
    print("=" * 55)

    print("\n  Demonstrating baseline theorem ...")
    theorem = demonstrate_baseline_theorem(5000)
    print(f"    ||E[∇logπ · R]||      = {theorem['mean_raw']:.5f}  (sample std={theorem['std_raw']:.4f})")
    print(f"    ||E[∇logπ · (R−b)]|| = {theorem['mean_centered']:.5f}  (sample std={theorem['std_centered']:.4f})")
    print(f"    → Means ≈ same; variance: {theorem['std_raw']:.4f} → {theorem['std_centered']:.4f}")

    methods = ["raw", "centred", "gae"]
    histories = {}
    for m in methods:
        print(f"\n  Training [{m}] ...")
        histories[m] = train(m, n_steps=n_steps, batch_size=batch_size, lr_policy=lr_policy)
        final_r = np.mean(histories[m]["reward"][-20:])
        final_v = histories[m]["grad_var"][-1] if histories[m]["grad_var"] else float("nan")
        print(f"    Final reward={final_r:.3f}/{SEQ_LEN}   grad_var={final_v:.5f}")

    print("\n  Plotting ...")
    plot_results(histories, theorem, n_steps)
    print("\n  Day 2 complete.\n")


if __name__ == "__main__":
    main()
