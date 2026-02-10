"""
Step 1: Scaled dot-product attention from scratch.
No torch.nn.MultiheadAttention — everything built manually.

Shapes walkthrough:
  X:  (B, C, E)  — batch of sequences, each C tokens, each E-dim embedding
  Wq: (E, D)     — projects embeddings into query space (D = head_dim)
  Wk: (E, D)     — projects embeddings into key space
  Wv: (E, D)     — projects embeddings into value space
  Q = X @ Wq:        (B, C, D)
  K = X @ Wk:        (B, C, D)
  V = X @ Wv:        (B, C, D)
  scores = Q @ K^T:  (B, C, C)  — raw attention scores
  scaled = scores / sqrt(D)
  masked = scaled + mask         — causal mask: -inf for future positions
  weights = softmax(masked)      — (B, C, C) attention weights
  output = weights @ V           — (B, C, D) context-weighted values
"""

import torch
import torch.nn.functional as F
import math

# ---------------------------------------------------------------------------
# Core: scaled dot-product attention (function, no nn.Module)
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q: (B, C, D) queries
        K: (B, C, D) keys
        V: (B, C, D) values
        mask: (C, C) or (B, C, C) — additive mask (0 = attend, -inf = block)
    Returns:
        output:  (B, C, D) — weighted sum of values
        weights: (B, C, C) — attention weights (for visualization)
    """
    D = Q.size(-1)

    # Step 1: raw attention scores
    scores = Q @ K.transpose(-2, -1)          # (B, C, C)

    # Step 2: scale by sqrt(head_dim) to prevent softmax saturation
    scores = scores / math.sqrt(D)

    # Step 3: apply mask (causal or padding)
    if mask is not None:
        scores = scores + mask

    # Step 4: normalize to get attention weights
    weights = F.softmax(scores, dim=-1)       # (B, C, C)

    # Step 5: weighted sum of values
    output = weights @ V                      # (B, C, D)

    return output, weights

# ---------------------------------------------------------------------------
# Causal mask: prevents attending to future positions
# ---------------------------------------------------------------------------
def make_causal_mask(seq_len):
    """
    Returns (seq_len, seq_len) upper-triangular mask filled with -inf.
    Position i can only attend to positions <= i.
    """
    mask = torch.full((seq_len, seq_len), float('-inf'))
    mask = torch.triu(mask, diagonal=1)  # -inf above diagonal, 0 on and below
    return mask

# ---------------------------------------------------------------------------
# Single-head attention module (manual projection matrices)
# ---------------------------------------------------------------------------
class SingleHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        # Learned projection matrices — no bias for clarity
        self.W_q = torch.nn.Linear(embed_dim, head_dim, bias=False)
        self.W_k = torch.nn.Linear(embed_dim, head_dim, bias=False)
        self.W_v = torch.nn.Linear(embed_dim, head_dim, bias=False)

    def forward(self, x, mask=None):
        """
        x: (B, C, E)
        Returns: (B, C, D) output, (B, C, C) attention weights
        """
        Q = self.W_q(x)    # (B, C, D)
        K = self.W_k(x)    # (B, C, D)
        V = self.W_v(x)    # (B, C, D)
        return scaled_dot_product_attention(Q, K, V, mask)

# ===========================================================================
# Tests & demonstrations
# ===========================================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    B, C, E, D = 2, 8, 16, 4  # batch=2, seq_len=8, embed=16, head_dim=4
    x = torch.randn(B, C, E)

    print(f"Input shape: {x.shape}  (B={B}, C={C}, E={E})")
    print(f"Head dim: {D}")
    print()

    # --- Test 1: no mask (full attention) ---
    print("=" * 50)
    print("Test 1: Full attention (no mask)")
    print("=" * 50)
    head = SingleHeadAttention(E, D)
    out, weights = head(x)
    print(f"Output shape:  {out.shape}   (expected: {B}, {C}, {D})")
    print(f"Weights shape: {weights.shape}  (expected: {B}, {C}, {C})")
    print(f"Weights row sums: {weights[0].sum(dim=-1)}")  # should be all 1s
    print(f"  (all 1.0 = softmax is normalized correctly)")
    print()

    # --- Test 2: causal mask ---
    print("=" * 50)
    print("Test 2: Causal attention (masked)")
    print("=" * 50)
    mask = make_causal_mask(C)
    out_causal, weights_causal = head(x, mask=mask)
    print(f"Causal mask:\n{mask}")
    print()
    print(f"Attention weights for batch 0 (rounded):")
    print(f"  (each row should only have non-zero entries at columns <= row index)")
    w = weights_causal[0]
    for i in range(C):
        row = "  ".join(f"{w[i,j]:.2f}" for j in range(C))
        print(f"  pos {i}: [{row}]")
    print()

    # Verify causality: no attention to future
    future_attn = torch.triu(weights_causal, diagonal=1)
    assert future_attn.abs().max() < 1e-6, "FAIL: attending to future positions!"
    print("  PASS: no attention to future positions")
    print()

    # --- Test 3: why scaling matters ---
    print("=" * 50)
    print("Test 3: Why scaling by sqrt(D) matters")
    print("=" * 50)
    Q = torch.randn(1, C, D)
    K = torch.randn(1, C, D)
    V = torch.randn(1, C, D)

    scores_raw = Q @ K.transpose(-2, -1)
    scores_scaled = scores_raw / math.sqrt(D)

    weights_unscaled = F.softmax(scores_raw, dim=-1)
    weights_scaled = F.softmax(scores_scaled, dim=-1)

    print(f"  Head dim D = {D}")
    print(f"  Raw scores   — mean: {scores_raw.mean():.3f}, std: {scores_raw.std():.3f}")
    print(f"  Scaled scores — mean: {scores_scaled.mean():.3f}, std: {scores_scaled.std():.3f}")
    print()
    print(f"  Unscaled attention weights (row 0): {weights_unscaled[0, 0].tolist()}")
    print(f"  Scaled attention weights   (row 0): {weights_scaled[0, 0].tolist()}")
    print()

    # With large D, the difference is dramatic
    D_large = 512
    Q_big = torch.randn(1, C, D_large)
    K_big = torch.randn(1, C, D_large)

    scores_big = Q_big @ K_big.transpose(-2, -1)
    w_unscaled = F.softmax(scores_big, dim=-1)
    w_scaled = F.softmax(scores_big / math.sqrt(D_large), dim=-1)

    print(f"  With D={D_large}:")
    print(f"  Raw scores std:    {scores_big.std():.1f}  (grows with sqrt(D))")
    print(f"  Unscaled entropy:  {-(w_unscaled * w_unscaled.log()).sum(-1).mean():.3f}")
    print(f"  Scaled entropy:    {-(w_scaled * w_scaled.log()).sum(-1).mean():.3f}")
    print(f"  (higher entropy = more uniform = gradients flow to all positions)")
    print(f"  (unscaled → peaky softmax → vanishing gradients for most positions)")
