"""
Step 3: Remove scaling — show that 1/sqrt(d_k) is necessary, not a hack.

Compares attention with and without scaling at increasing head dimensions
to demonstrate softmax saturation and gradient instability.
"""

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from step1_scaled_dot_product_attention import make_causal_mask

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

torch.manual_seed(42)

sentence = "the cat sat on the mat".split()
seq_len = len(sentence)
mask = make_causal_mask(seq_len)

# ---------------------------------------------------------------------------
# Compare scaled vs unscaled at different head dimensions
# ---------------------------------------------------------------------------
head_dims = [4, 16, 64, 256, 512]

print(f"{BOLD}Scaled vs Unscaled attention across head dimensions{RESET}")
print(f"{'='*70}")
print(f"  {'D':>5}  {'Scores std':>11}  {'Unscaled entropy':>17}  {'Scaled entropy':>15}  {'Max weight (unscaled)':>22}")
print(f"  {'-'*5}  {'-'*11}  {'-'*17}  {'-'*15}  {'-'*22}")

results = []
for D in head_dims:
    Q = torch.randn(1, seq_len, D)
    K = torch.randn(1, seq_len, D)
    V = torch.randn(1, seq_len, D, requires_grad=True)
    V_scaled = V.detach().clone().requires_grad_(True)

    scores = Q @ K.transpose(-2, -1)

    # Unscaled
    w_unscaled = F.softmax(scores + mask, dim=-1)
    out_unscaled = (w_unscaled @ V)
    out_unscaled.sum().backward()
    grad_unscaled = V.grad.clone()

    # Scaled
    w_scaled = F.softmax(scores / math.sqrt(D) + mask, dim=-1)
    out_scaled = (w_scaled @ V_scaled)
    out_scaled.sum().backward()
    grad_scaled = V_scaled.grad.clone()

    # Entropy: higher = more uniform attention
    entropy_unscaled = -(w_unscaled * (w_unscaled + 1e-10).log()).sum(-1).mean().item()
    entropy_scaled = -(w_scaled * (w_scaled + 1e-10).log()).sum(-1).mean().item()
    max_w = w_unscaled.max().item()
    score_std = scores.std().item()

    results.append({
        'D': D, 'score_std': score_std,
        'entropy_unscaled': entropy_unscaled, 'entropy_scaled': entropy_scaled,
        'max_w': max_w,
        'w_unscaled': w_unscaled[0].detach(),
        'w_scaled': w_scaled[0].detach(),
        'grad_unscaled': grad_unscaled, 'grad_scaled': grad_scaled,
    })

    color = RED if entropy_unscaled < 0.5 else RESET
    print(f"  {D:>5}  {score_std:>11.2f}  {color}{entropy_unscaled:>17.3f}{RESET}  "
          f"{entropy_scaled:>15.3f}  {color}{max_w:>22.4f}{RESET}")

# ---------------------------------------------------------------------------
# Show attention matrices side-by-side for D=4 vs D=512
# ---------------------------------------------------------------------------
for r in results:
    if r['D'] in (4, 512):
        print(f"\n{BOLD}D = {r['D']}:{RESET}")
        print(f"  Unscaled weights (row = query, col = key):")
        for i in range(seq_len):
            row = "  ".join(f"{r['w_unscaled'][i,j]:.3f}" for j in range(seq_len))
            print(f"    {sentence[i]:>5s}  [{row}]")

        print(f"  Scaled weights:")
        for i in range(seq_len):
            row = "  ".join(f"{r['w_scaled'][i,j]:.3f}" for j in range(seq_len))
            print(f"    {sentence[i]:>5s}  [{row}]")

# ---------------------------------------------------------------------------
# Gradient analysis
# ---------------------------------------------------------------------------
print(f"\n{BOLD}Gradient flow through V (how well does learning signal reach all values){RESET}")
print(f"{'='*70}")
print(f"  {'D':>5}  {'Grad std (unscaled)':>20}  {'Grad std (scaled)':>18}  {'Ratio':>8}")
print(f"  {'-'*5}  {'-'*20}  {'-'*18}  {'-'*8}")
for r in results:
    g_un = r['grad_unscaled'].std().item()
    g_sc = r['grad_scaled'].std().item()
    ratio = g_un / g_sc if g_sc > 0 else float('inf')
    color = RED if ratio < 0.1 else RESET
    print(f"  {r['D']:>5}  {color}{g_un:>20.6f}{RESET}  {g_sc:>18.6f}  {color}{ratio:>8.3f}{RESET}")

# ---------------------------------------------------------------------------
# Heatmaps: D=512 scaled vs unscaled
# ---------------------------------------------------------------------------
r512 = [r for r in results if r['D'] == 512][0]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, w, title in [
    (axes[0], r512['w_unscaled'], f"Unscaled (D=512)\nentropy={r512['entropy_unscaled']:.3f}"),
    (axes[1], r512['w_scaled'], f"Scaled (D=512)\nentropy={r512['entropy_scaled']:.3f}"),
]:
    im = ax.imshow(w.numpy(), cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(sentence, rotation=45, ha='right')
    ax.set_yticklabels(sentence)
    ax.set_title(title)
    for i in range(seq_len):
        for j in range(seq_len):
            v = w[i, j].item()
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=9, color='white' if v > 0.5 else 'black')

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'scaling_comparison.png')
plt.savefig(out_path, dpi=150)
print(f"\nSaved heatmap to {out_path}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"""
{'='*70}
{BOLD}WHY 1/sqrt(d_k) IS NECESSARY{RESET}
{'='*70}

{BOLD}The problem:{RESET}
  Q @ K^T computes D independent dot products, each ~ N(0,1).
  The sum of D such terms has variance = D.
  So score std ≈ sqrt(D):  D=4 → std≈2,  D=512 → std≈23.

{BOLD}What happens without scaling:{RESET}
  Large scores → softmax saturates → nearly one-hot weights.
  At D=512: max weight ≈ 1.0, entropy ≈ 0.0
  → One value vector gets weight ~1.0, all others get ~0.0
  → Gradients vanish for all but one position in V
  → The model can barely learn to redistribute attention

{BOLD}What scaling fixes:{RESET}
  Dividing by sqrt(D) normalizes variance back to ~1,
  regardless of head dimension.
  → Softmax stays smooth, entropy stays high
  → Gradients flow to ALL value vectors
  → The model can learn nuanced attention patterns

{BOLD}This is not optional:{RESET}
  Without scaling, deeper/wider models break harder.
  It's the reason the original paper is called
  "Attention Is All You Need" and not "Attention Plus
  Careful Score Normalization Is All You Need" —
  they baked the fix into the definition itself.
""")
