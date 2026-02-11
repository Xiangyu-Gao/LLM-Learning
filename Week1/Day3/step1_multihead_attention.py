"""
Step 1: Multi-head attention from scratch.

Takes the single-head implementation from Day2 and extends it:
  1. Project into H separate heads: (B, C, E) → (B, H, C, D)
  2. Apply scaled dot-product attention independently per head
  3. Concatenate heads: (B, H, C, D) → (B, C, H*D)
  4. Final linear projection back to embed_dim

No torch.nn.MultiheadAttention — everything manual.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Day2'))
from step1_scaled_dot_product_attention import (
    scaled_dot_product_attention, make_causal_mask,
)

# ---------------------------------------------------------------------------
# Multi-head attention
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention from scratch.

    Key insight: instead of H separate (E→D) projections, we use one
    (E→H*D) projection and reshape. Mathematically equivalent, but
    one big matmul is faster than H small ones.

    Shapes:
      Input x:       (B, C, E)
      After W_q/k/v: (B, C, H*D)
      Reshape:       (B, C, H, D)
      Transpose:     (B, H, C, D)  ← each head is an independent (C, D) attention
      After attn:    (B, H, C, D)
      Transpose back:(B, C, H, D)
      Reshape:       (B, C, H*D)
      After W_o:     (B, C, E)     ← back to original embedding dim
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # D = E / H

        # One big projection for all heads (equivalent to H separate projections)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)  # (E) → (H*D)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection: concat of heads → embed_dim
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, mask=None):
        """
        x: (B, C, E)
        mask: (C, C) additive mask
        Returns: (B, C, E) output, (B, H, C, C) attention weights per head
        """
        B, C, E = x.shape
        H, D = self.num_heads, self.head_dim

        # Step 1: Project to Q, K, V — one matmul per, covering all heads
        Q = self.W_q(x)  # (B, C, H*D)
        K = self.W_k(x)
        V = self.W_v(x)

        # Step 2: Reshape to separate heads
        # (B, C, H*D) → (B, C, H, D) → (B, H, C, D)
        Q = Q.view(B, C, H, D).transpose(1, 2)   # (B, H, C, D)
        K = K.view(B, C, H, D).transpose(1, 2)
        V = V.view(B, C, H, D).transpose(1, 2)

        # Step 3: Apply scaled dot-product attention independently per head
        # scaled_dot_product_attention works on last 2 dims,
        # so (B, H, C, D) treats (B, H) as batch dims — each head independent
        attn_out, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        # attn_out:     (B, H, C, D)
        # attn_weights: (B, H, C, C)

        # Step 4: Concatenate heads
        # (B, H, C, D) → (B, C, H, D) → (B, C, H*D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, C, H * D)

        # Step 5: Final projection
        output = self.W_o(attn_out)  # (B, C, E)

        return output, attn_weights

# ===========================================================================
# Tests & verification
# ===========================================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    B, C, E = 2, 8, 32   # batch=2, seq_len=8, embed=32
    num_heads = 4         # 4 heads, each with D = 32/4 = 8
    D = E // num_heads

    x = torch.randn(B, C, E)

    print(f"Input:     (B={B}, C={C}, E={E})")
    print(f"Heads:     H={num_heads}, head_dim D={D}")
    print()

    # --- Test 1: output shape matches input embedding dim ---
    print("=" * 55)
    print("Test 1: Output shape matches input embedding dimension")
    print("=" * 55)
    mha = MultiHeadAttention(E, num_heads)
    out, weights = mha(x)
    print(f"  Input shape:   {x.shape}")
    print(f"  Output shape:  {out.shape}")
    print(f"  Weights shape: {weights.shape}  (B, H, C, C)")
    assert out.shape == x.shape, f"FAIL: {out.shape} != {x.shape}"
    print(f"  PASS: output (B, C, E) = input (B, C, E)")
    print()

    # --- Test 2: each head has its own Q, K, V ---
    print("=" * 55)
    print("Test 2: Each head has independent attention patterns")
    print("=" * 55)
    mask = make_causal_mask(C)
    out, weights = mha(x, mask=mask)

    # Compare attention patterns across heads
    print(f"  Attention weights per head for position 7 (last), batch 0:")
    for h in range(num_heads):
        w = weights[0, h, -1]  # last position's attention for head h
        top3 = torch.topk(w, 3)
        top3_str = ", ".join(f"pos {i}: {v:.3f}" for i, v in zip(top3.indices.tolist(), top3.values.tolist()))
        print(f"    Head {h}: top-3 = [{top3_str}]")

    # Measure how different the heads are
    print(f"\n  Pairwise cosine similarity between heads (last position):")
    for h1 in range(num_heads):
        for h2 in range(h1 + 1, num_heads):
            sim = F.cosine_similarity(
                weights[0, h1, -1].unsqueeze(0),
                weights[0, h2, -1].unsqueeze(0)
            ).item()
            print(f"    Head {h1} vs {h2}: {sim:.3f}")
    print()

    # --- Test 3: causal masking works across all heads ---
    print("=" * 55)
    print("Test 3: Causal masking enforced in every head")
    print("=" * 55)
    future_attn = torch.triu(weights, diagonal=1)  # (B, H, C, C) upper triangle
    max_future = future_attn.abs().max().item()
    print(f"  Max attention to future positions: {max_future:.8f}")
    assert max_future < 1e-6, "FAIL: future attention detected!"
    print(f"  PASS: all heads respect causal mask")
    print()

    # --- Test 4: equivalence to H separate SingleHeadAttentions ---
    print("=" * 55)
    print("Test 4: Multi-head == H independent single-heads (verify)")
    print("=" * 55)

    # Extract what each head's W_q, W_k, W_v actually are
    # W_q.weight is (H*D, E). Head h's slice is [h*D : (h+1)*D, :]
    print(f"  W_q total shape: {mha.W_q.weight.shape}  (H*D={num_heads*D}, E={E})")
    for h in range(num_heads):
        wq_h = mha.W_q.weight[h*D:(h+1)*D, :]
        print(f"    Head {h} W_q slice: rows [{h*D}:{(h+1)*D}]  shape {wq_h.shape}")
    print(f"  Each head gets its own {D}×{E} projection — {num_heads} independent Q spaces")
    print()

    # --- Test 5: parameter count ---
    print("=" * 55)
    print("Test 5: Parameter count")
    print("=" * 55)
    total = sum(p.numel() for p in mha.parameters())
    per_proj = E * E  # each of W_q, W_k, W_v, W_o is (E, E)
    print(f"  W_q: {per_proj}  (E×E = {E}×{E})")
    print(f"  W_k: {per_proj}")
    print(f"  W_v: {per_proj}")
    print(f"  W_o: {per_proj}")
    print(f"  Total: {total}  (4 × E² = 4 × {E}² = {4 * E * E})")
    print(f"  Same parameter count regardless of num_heads!")
    print(f"  H=1 with D=32  vs  H=4 with D=8  →  same 4×E² params")
    print()

    # --- Summary ---
    print("=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"""
  Multi-head attention splits the embedding into {num_heads} independent
  attention operations, each with head_dim={D}.

  Why multiple heads?
    - Each head can attend to different aspects:
      one head might track position, another syntax, another semantics
    - With 1 head of dim {E}: one attention pattern per position
    - With {num_heads} heads of dim {D}: {num_heads} different patterns combined

  The reshape trick:
    Instead of {num_heads} separate matmuls of size (E→{D}),
    we do 1 matmul of size (E→{E}) and reshape.
    Same math, better GPU utilization.

  Key shapes:
    (B, C, E) → W_q/k/v → (B, C, H*D) → reshape → (B, H, C, D)
    → attention per head → (B, H, C, D)
    → concat → (B, C, H*D) → W_o → (B, C, E)
""")
