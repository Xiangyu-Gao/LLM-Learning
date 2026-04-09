"""
common.py — Shared infrastructure for Week 10 RLHF Mastery experiments.
=========================================================================
Extends Week 9 infrastructure with scaling support, replay buffer,
and diagnostic utilities.

New in Week 10
--------------
PolicyConfig       — Architecture hyperparameters (controls "model scale").
make_policy()      — Factory: 'small' (32-dim) | 'medium' (64-dim) | 'large' (128-dim).
ReplayBuffer       — Circular buffer with age tracking for off-policy experiments.
gradient_norm()    — L2 norm of all policy gradients (instability signal).
reward_volatility()— Rolling std of reward (instability signal).
sharpness()        — Gradient perturbation sharpness proxy (curvature estimate).

Inherited from Week 9
---------------------
TinyRNNPolicy      — GRU-based autoregressive policy (vocab_size=8, seq_len=4 default).
TinyValueNet       — Critic: hidden state → scalar V(s).
TinyRewardModel    — Reward model: token sequence → scalar.
rollout()          — Sample B sequences from policy.
compute_returns()  — Monte-Carlo returns G_t.
compute_gae()      — Generalised Advantage Estimation.
kl_per_token()     — Mean per-token KL(π_θ || π_ref).
token_entropy()    — Mean per-token Shannon entropy.
smooth()           — Moving-average smoothing for plots.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from dataclasses import dataclass, field
from collections import deque
from typing import List, Optional
import numpy as np


# ─── Architecture Config ──────────────────────────────────────────────────────

@dataclass
class PolicyConfig:
    """
    Controls "model scale" for scaling experiments.

    The key axes of scale in a toy RNN policy:
      hidden_dim  — width of the GRU hidden state
      embed_dim   — token embedding dimension
      n_layers    — number of stacked GRU cells

    Larger hidden_dim → sharper loss landscape → more curvature →
    earlier instability at high learning rates.

    Approximate parameter counts (vocab=8, seq_len=4):
      small:  hidden=32,  embed=16  → ~2K params
      medium: hidden=64,  embed=32  → ~10K params
      large:  hidden=128, embed=64  → ~40K params
    """
    hidden_dim: int = 32
    embed_dim:  int = 16
    vocab_size: int = 8
    seq_len:    int = 4
    name:       str = "small"


SCALE_CONFIGS = {
    "small":  PolicyConfig(hidden_dim=32,  embed_dim=16,  name="small"),
    "medium": PolicyConfig(hidden_dim=64,  embed_dim=32,  name="medium"),
    "large":  PolicyConfig(hidden_dim=128, embed_dim=64,  name="large"),
}


# ─── Policy ───────────────────────────────────────────────────────────────────

class TinyRNNPolicy(nn.Module):
    """
    Autoregressive policy: p(x_1, ..., x_T) = ∏_t p(x_t | x_1, ..., x_{t-1})

    Architecture:
      BOS → Embedding → GRUCell (hidden_dim) → Linear → softmax → x_1
      x_1 → Embedding → GRUCell → Linear → softmax → x_2  ...

    The BOS (beginning-of-sequence) token has index vocab_size.
    """
    def __init__(self, cfg: Optional[PolicyConfig] = None,
                 vocab_size: int = 8, embed_dim: int = 16,
                 hidden_dim: int = 32, seq_len: int = 4):
        super().__init__()
        # Accept either a PolicyConfig or individual kwargs (backward compat)
        if cfg is not None:
            vocab_size = cfg.vocab_size
            embed_dim  = cfg.embed_dim
            hidden_dim = cfg.hidden_dim
            seq_len    = cfg.seq_len

        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.hidden_dim = hidden_dim
        self.BOS        = vocab_size  # sentinel start token

        self.embed = nn.Embedding(vocab_size + 1, embed_dim)
        self.rnn   = nn.GRUCell(embed_dim, hidden_dim)
        self.head  = nn.Linear(hidden_dim, vocab_size)

    # ------------------------------------------------------------------
    def _init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim)

    def _step_logits(self, token: torch.Tensor, h: torch.Tensor):
        """One RNN step. token: (B,) int64. Returns new_h (B, H), logits (B, V)."""
        emb    = self.embed(token)
        h      = self.rnn(emb, h)
        logits = self.head(h)
        return h, logits

    # ------------------------------------------------------------------
    def sample_sequence(self, batch_size: int = 1):
        """
        Autoregressively sample seq_len tokens.

        Returns
        -------
        tokens    : (B, T)    int64 — sampled token ids
        log_probs : (B, T)    float — log-probability of each sampled token
        hiddens   : (B, T, H) float — GRU hidden states (for value net)
        """
        B = batch_size
        h = self._init_hidden(B)
        x = torch.full((B,), self.BOS, dtype=torch.long)

        tokens_list, lp_list, h_list = [], [], []
        for _ in range(self.seq_len):
            h, logits = self._step_logits(x, h)
            dist = torch.distributions.Categorical(logits=logits)
            tok  = dist.sample()
            lp   = dist.log_prob(tok)
            tokens_list.append(tok)
            lp_list.append(lp)
            h_list.append(h)
            x = tok

        tokens    = torch.stack(tokens_list, dim=1)
        log_probs = torch.stack(lp_list,    dim=1)
        hiddens   = torch.stack(h_list,     dim=1)
        return tokens, log_probs, hiddens

    # ------------------------------------------------------------------
    def log_probs_of(self, tokens: torch.Tensor):
        """
        Compute log p(tokens) under the current policy.

        tokens : (B, T) int64
        Returns: log_probs (B, T), hiddens (B, T, H)
        """
        B, T = tokens.shape
        h = self._init_hidden(B)
        x = torch.full((B,), self.BOS, dtype=torch.long)

        lp_list, h_list = [], []
        for t in range(T):
            h, logits = self._step_logits(x, h)
            dist = torch.distributions.Categorical(logits=logits)
            lp   = dist.log_prob(tokens[:, t])
            lp_list.append(lp)
            h_list.append(h)
            x = tokens[:, t]

        log_probs = torch.stack(lp_list, dim=1)
        hiddens   = torch.stack(h_list,  dim=1)
        return log_probs, hiddens

    # ------------------------------------------------------------------
    def entropy(self) -> torch.Tensor:
        """Approximate mean token entropy (nats)."""
        with torch.no_grad():
            B = 64
            h = self._init_hidden(B)
            x = torch.full((B,), self.BOS, dtype=torch.long)
            entropies = []
            for _ in range(self.seq_len):
                h, logits = self._step_logits(x, h)
                probs = F.softmax(logits, dim=-1)
                ent = -(probs * probs.log().clamp(min=-20)).sum(dim=-1)
                entropies.append(ent.mean())
                x = torch.distributions.Categorical(logits=logits).sample()
        return torch.stack(entropies).mean()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─── Factory ──────────────────────────────────────────────────────────────────

def make_policy(size: str = "small") -> TinyRNNPolicy:
    """
    Create a policy at a given scale.

    Parameters
    ----------
    size : 'small' | 'medium' | 'large'
        Corresponds to hidden_dim = 32 / 64 / 128.

    The larger the model, the sharper the loss landscape (higher curvature),
    which means instability appears at lower learning rates.
    """
    cfg = SCALE_CONFIGS[size]
    return TinyRNNPolicy(cfg=cfg)


# ─── Value Network ────────────────────────────────────────────────────────────

class TinyValueNet(nn.Module):
    """Critic: GRU hidden state → scalar V(s_t)."""
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


# ─── Reward Model ─────────────────────────────────────────────────────────────

class TinyRewardModel(nn.Module):
    """Reward model: token sequence → scalar reward."""
    def __init__(self, vocab_size: int = 8, embed_dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.net   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) int64. Returns reward (B,)."""
        emb  = self.embed(tokens)
        pool = emb.mean(dim=1)
        return self.net(pool).squeeze(-1)


# ─── Replay Buffer ────────────────────────────────────────────────────────────

@dataclass
class Transition:
    """One batch of experience from a single rollout."""
    tokens:    torch.Tensor   # (B, T)
    log_probs: torch.Tensor   # (B, T) — log probs at collection time
    rewards:   torch.Tensor   # (B,)
    age:       int = 0        # how many updates old is this experience?


class ReplayBuffer:
    """
    Circular buffer of Transitions.

    Used in Day 2 to simulate off-policy training.

    Key concept: 'age' tracks how many gradient updates have happened
    since the experience was collected.  Age > 0 means off-policy.

    Importance sampling weight: w = π_θ(a|s) / π_old(a|s)
    For sequences: w = exp(Σ_t log π_θ(a_t|s_t) - log π_old(a_t|s_t))
    """
    def __init__(self, max_size: int = 10):
        self._buf: deque = deque(maxlen=max_size)

    def push(self, transition: Transition):
        self._buf.append(transition)

    def age_all(self):
        """Increment age counter on all stored transitions."""
        for t in self._buf:
            t.age += 1

    def sample(self, n: Optional[int] = None) -> List[Transition]:
        """Return n most recent transitions (or all if n is None)."""
        buf = list(self._buf)
        if n is None or n >= len(buf):
            return buf
        return buf[-n:]

    def __len__(self):
        return len(self._buf)


# ─── Rollout ──────────────────────────────────────────────────────────────────

def rollout(policy: TinyRNNPolicy, batch_size: int = 32):
    """
    Sample a batch of sequences from `policy`.

    Returns a dict with:
      'tokens'    : (B, T)
      'log_probs' : (B, T)
      'hiddens'   : (B, T, H)
    """
    tokens, log_probs, hiddens = policy.sample_sequence(batch_size)
    return {"tokens": tokens, "log_probs": log_probs, "hiddens": hiddens}


# ─── Return / Advantage ───────────────────────────────────────────────────────

def compute_returns(step_rewards: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """
    Monte-Carlo returns G_t = Σ_{k≥t} γ^{k-t} r_k.

    step_rewards : (B, T)
    Returns      : (B, T)
    """
    B, T = step_rewards.shape
    G = torch.zeros_like(step_rewards)
    g = torch.zeros(B)
    for t in reversed(range(T)):
        g = step_rewards[:, t] + gamma * g
        G[:, t] = g
    return G


def compute_gae(
    rewards: torch.Tensor,
    values:  torch.Tensor,
    gamma:   float = 0.99,
    lam:     float = 0.95,
) -> torch.Tensor:
    """
    Generalised Advantage Estimation.

    rewards : (T,)   per-step rewards
    values  : (T+1,) V(s_0), ..., V(s_T) — last entry = 0 (terminal)
    Returns : (T,)   A_t^GAE

    δ_t = r_t + γ V(s_{t+1}) - V(s_t)
    A_t = Σ_{l≥0} (γλ)^l δ_{t+l}
    """
    T = rewards.shape[0]
    advantages = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        delta       = rewards[t] + gamma * values[t + 1] - values[t]
        gae         = delta + gamma * lam * gae
        advantages[t] = gae
    return advantages


# ─── KL / Entropy ─────────────────────────────────────────────────────────────

@torch.no_grad()
def kl_per_token(
    policy:     TinyRNNPolicy,
    ref_policy: TinyRNNPolicy,
    tokens:     torch.Tensor,
) -> float:
    """
    Mean per-token forward KL: KL(π_θ || π_ref)

    Sampled-token approximation (tokens ~ π_θ):
        KL ≈ E_{x~π_θ}[log π_θ(x) - log π_ref(x)]
    """
    lp_cur, _ = policy.log_probs_of(tokens)
    lp_ref, _ = ref_policy.log_probs_of(tokens)
    return (lp_cur - lp_ref).mean().item()


@torch.no_grad()
def token_entropy(policy: TinyRNNPolicy) -> float:
    """Mean token entropy of the policy (nats)."""
    return policy.entropy().item()


# ─── Diagnostics ─────────────────────────────────────────────────────────────

def gradient_norm(policy: TinyRNNPolicy) -> float:
    """
    L2 norm of all parameter gradients.

    A spike in gradient norm is an early-warning sign of instability.
    Best called after loss.backward() but before optimizer.step().

    Returns 0.0 if no gradients exist (e.g., called before first backward).
    """
    total = 0.0
    for p in policy.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm().item() ** 2
    return total ** 0.5


def reward_volatility(reward_history: List[float], window: int = 20) -> float:
    """
    Rolling standard deviation of reward over the last `window` steps.

    High volatility = training is unstable.
    Low volatility = either stable convergence or mode collapse.
    """
    if len(reward_history) < 2:
        return 0.0
    arr = np.array(reward_history[-window:])
    return float(arr.std())


def sharpness_proxy(
    policy:   TinyRNNPolicy,
    loss_fn,
    epsilon:  float = 1e-3,
    n_probes: int   = 10,
) -> float:
    """
    Estimate loss-landscape sharpness by random perturbation.

    Sharpness ≈ max_{||δ||≤ε} [L(θ+δ) - L(θ)] / ε

    Larger sharpness → higher curvature → harder to optimize →
    requires smaller learning rates.  This is the key mechanism behind
    'scaling changes optimization curvature'.
    """
    base_loss = loss_fn().item()
    max_delta = 0.0
    original_params = [p.data.clone() for p in policy.parameters()]

    for _ in range(n_probes):
        # Apply random perturbation of norm epsilon
        for p in policy.parameters():
            noise = torch.randn_like(p.data)
            noise = noise / (noise.norm() + 1e-8) * epsilon
            p.data.add_(noise)

        perturbed_loss = loss_fn().item()
        delta = abs(perturbed_loss - base_loss)
        max_delta = max(max_delta, delta)

        # Restore
        for p, orig in zip(policy.parameters(), original_params):
            p.data.copy_(orig)

    return max_delta / epsilon


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def smooth(x, w: int = 10):
    """Moving average with window w."""
    arr = np.asarray(x, dtype=float)
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid")
