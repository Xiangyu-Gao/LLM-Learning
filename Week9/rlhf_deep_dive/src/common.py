"""
common.py — Shared infrastructure for Week 9 RLHF experiments.
===============================================================
All toy experiments in Days 3–6 share this module.

Architecture overview
---------------------
TinyRNNPolicy   — A GRU-based autoregressive policy.
                  vocab_size=8, seq_len=4 by default.
                  Generates one token at a time, conditioned on history.

TinyValueNet    — Critic for PPO.  Maps a partial sequence to a scalar V(s_t).

TinyRewardModel — Maps a complete sequence to a scalar reward (used in Day 3).

Utilities
---------
rollout()             — Sample B sequences from the policy.
compute_returns()     — Monte-Carlo returns G_t.
compute_gae()         — Generalized Advantage Estimation.
kl_per_token()        — Mean token-level KL between two policies.
token_entropy()       — Mean per-token Shannon entropy of the policy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


# ─── Policy ───────────────────────────────────────────────────────────────────

class TinyRNNPolicy(nn.Module):
    """
    Autoregressive policy: p(x_1, ..., x_T) = ∏_t p(x_t | x_1, ..., x_{t-1})

    Architecture:
      BOS → Embedding → GRUCell (hidden_dim) → Linear → softmax → x_1
      x_1 → Embedding → GRUCell → Linear → softmax → x_2  ...

    The BOS (beginning-of-sequence) token has index vocab_size.
    """
    def __init__(self, vocab_size: int = 8, embed_dim: int = 16,
                 hidden_dim: int = 32, seq_len: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.BOS = vocab_size  # sentinel start token

        self.embed = nn.Embedding(vocab_size + 1, embed_dim)
        self.rnn   = nn.GRUCell(embed_dim, hidden_dim)
        self.head  = nn.Linear(hidden_dim, vocab_size)

    # ------------------------------------------------------------------
    def _init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim)

    def _step_logits(self, token: torch.Tensor, h: torch.Tensor):
        """One RNN step.  token: (B,) int64.  Returns new_h (B, H), logits (B, V)."""
        emb = self.embed(token)          # (B, embed_dim)
        h   = self.rnn(emb, h)           # (B, hidden_dim)
        logits = self.head(h)            # (B, vocab_size)
        return h, logits

    # ------------------------------------------------------------------
    def sample_sequence(self, batch_size: int = 1):
        """
        Autoregressively sample `seq_len` tokens.

        Returns
        -------
        tokens    : (B, T)  int64 — sampled token ids
        log_probs : (B, T)  float — log-probability of each sampled token
        hiddens   : (B, T, H) float — GRU hidden states (for value net)
        """
        B = batch_size
        h = self._init_hidden(B)
        x = torch.full((B,), self.BOS, dtype=torch.long)  # start with BOS

        tokens_list, lp_list, h_list = [], [], []
        for _ in range(self.seq_len):
            h, logits = self._step_logits(x, h)
            dist = torch.distributions.Categorical(logits=logits)
            tok  = dist.sample()           # (B,)
            lp   = dist.log_prob(tok)      # (B,)
            tokens_list.append(tok)
            lp_list.append(lp)
            h_list.append(h)
            x = tok

        tokens    = torch.stack(tokens_list, dim=1)   # (B, T)
        log_probs = torch.stack(lp_list,    dim=1)    # (B, T)
        hiddens   = torch.stack(h_list,     dim=1)    # (B, T, H)
        return tokens, log_probs, hiddens

    # ------------------------------------------------------------------
    def log_probs_of(self, tokens: torch.Tensor):
        """
        Compute log p(tokens) under the current policy (for ratio computation).

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
            lp   = dist.log_prob(tokens[:, t])   # (B,)
            lp_list.append(lp)
            h_list.append(h)
            x = tokens[:, t]

        log_probs = torch.stack(lp_list, dim=1)   # (B, T)
        hiddens   = torch.stack(h_list,  dim=1)   # (B, T, H)
        return log_probs, hiddens

    # ------------------------------------------------------------------
    def entropy(self) -> torch.Tensor:
        """
        Approximate mean token entropy by sampling a batch and computing
        -Σ p log p at each step.  Returns scalar.
        """
        with torch.no_grad():
            B = 64
            h = self._init_hidden(B)
            x = torch.full((B,), self.BOS, dtype=torch.long)
            entropies = []
            for _ in range(self.seq_len):
                h, logits = self._step_logits(x, h)
                probs = F.softmax(logits, dim=-1)
                ent = -(probs * probs.log().clamp(min=-20)).sum(dim=-1)  # (B,)
                entropies.append(ent.mean())
                # sample to continue chain
                x = torch.distributions.Categorical(logits=logits).sample()
        return torch.stack(entropies).mean()


# ─── Value Network ────────────────────────────────────────────────────────────

class TinyValueNet(nn.Module):
    """
    Critic: maps the GRU hidden state at each step to V(s_t).

    Input:  hidden state h_t  (B, hidden_dim)
    Output: scalar value V    (B,)
    """
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, hidden_dim) or (B, T, hidden_dim).  Returns values of same batch shape."""
        return self.net(h).squeeze(-1)


# ─── Reward Model ─────────────────────────────────────────────────────────────

class TinyRewardModel(nn.Module):
    """
    Reward model: maps a complete token sequence to a scalar reward.

    Architecture:
      tokens → Embedding → mean-pool → MLP → scalar
    """
    def __init__(self, vocab_size: int = 8, embed_dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.net   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) int64.  Returns reward (B,)."""
        emb  = self.embed(tokens)          # (B, T, embed_dim)
        pool = emb.mean(dim=1)             # (B, embed_dim)
        return self.net(pool).squeeze(-1)  # (B,)


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
    Monte-Carlo returns for each time step.

    step_rewards : (B, T)  reward received at each step
    Returns      : (B, T)  G_t = Σ_{k≥t} γ^{k-t} r_k
    """
    B, T = step_rewards.shape
    G = torch.zeros_like(step_rewards)
    g = torch.zeros(B)
    for t in reversed(range(T)):
        g = step_rewards[:, t] + gamma * g
        G[:, t] = g
    return G


def compute_gae(
    rewards:  torch.Tensor,
    values:   torch.Tensor,
    gamma:    float = 0.99,
    lam:      float = 0.95,
) -> torch.Tensor:
    """
    Generalised Advantage Estimation.

    rewards : (T,)  per-step rewards
    values  : (T+1,) V(s_0), ..., V(s_T)  — last entry is V(terminal)=0
    Returns : (T,)  advantages A_t^GAE

    A_t^GAE = Σ_{l=0}^{T-t-1} (γλ)^l δ_{t+l}
    δ_t     = r_t + γ V(s_{t+1}) - V(s_t)   (TD-error)
    """
    T = rewards.shape[0]
    advantages = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae   = delta + gamma * lam * gae
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
    Mean per-token forward KL:  KL(π_θ || π_ref)

    Using the sampled-token approximation (tokens were sampled from π_θ):

        KL(π_θ || π_ref) = E_{x~π_θ} [ log π_θ(x) - log π_ref(x) ]

    This is always ≥ 0, and is the quantity subtracted as the KL penalty
    in RLHF:  r_eff = r_RM - β · KL(π_θ || π_ref)
    """
    lp_cur, _ = policy.log_probs_of(tokens)      # (B, T)
    lp_ref, _ = ref_policy.log_probs_of(tokens)  # (B, T)
    kl = (lp_cur - lp_ref).mean().item()          # ≥ 0
    return kl


@torch.no_grad()
def token_entropy(policy: TinyRNNPolicy) -> float:
    """Mean token entropy of the policy (nats)."""
    return policy.entropy().item()


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def smooth(x, w: int = 10):
    """Moving average with window w."""
    import numpy as np
    arr = np.asarray(x, dtype=float)
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid")
