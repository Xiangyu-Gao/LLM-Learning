"""
Step 2: Build a toy input and visualize attention.

Uses "the cat sat on the mat" — word-level tokens so the attention
matrix is small enough to read. Compares:
  1. Full attention (no mask)
  2. Causal attention (masked)
  3. What breaks without masking
"""

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from step1_scaled_dot_product_attention import (
    scaled_dot_product_attention, make_causal_mask,
)

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Toy input: "the cat sat on the mat"
# ---------------------------------------------------------------------------
sentence = "the cat sat on the mat".split()
vocab = {w: i for i, w in enumerate(sorted(set(sentence)))}
tokens = [vocab[w] for w in sentence]
seq_len = len(sentence)

print(f"{BOLD}Input:{RESET} {' '.join(sentence)}")
print(f"Tokens: {tokens}  (vocab: {vocab})")
print()

# Create random but fixed embeddings for each word
torch.manual_seed(42)
embed_dim = 16
head_dim = 8

# Embedding lookup
embed_table = torch.randn(len(vocab), embed_dim)
x = embed_table[torch.tensor(tokens)].unsqueeze(0)  # (1, 6, 16)

# Manual Q, K, V projections
W_q = torch.randn(embed_dim, head_dim) * 0.1
W_k = torch.randn(embed_dim, head_dim) * 0.1
W_v = torch.randn(embed_dim, head_dim) * 0.1

Q = x @ W_q  # (1, 6, 8)
K = x @ W_k
V = x @ W_v

# ---------------------------------------------------------------------------
# 1. Full attention (no mask)
# ---------------------------------------------------------------------------
out_full, weights_full = scaled_dot_product_attention(Q, K, V, mask=None)
w_full = weights_full[0].detach()

# ---------------------------------------------------------------------------
# 2. Causal attention
# ---------------------------------------------------------------------------
mask = make_causal_mask(seq_len)
out_causal, weights_causal = scaled_dot_product_attention(Q, K, V, mask=mask)
w_causal = weights_causal[0].detach()

# ---------------------------------------------------------------------------
# Print attention matrices in terminal
# ---------------------------------------------------------------------------
def print_attention_matrix(weights, labels, title):
    print(f"{BOLD}{title}{RESET}")
    header = "         " + "  ".join(f"{l:>5s}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row_vals = "  ".join(f"{weights[i,j]:.2f}" for j in range(len(labels)))
        # Highlight diagonal
        print(f"  {row_label:>5s}  [{row_vals}]")
    print()

print_attention_matrix(w_full, sentence, "Full Attention (no mask)")
print_attention_matrix(w_causal, sentence, "Causal Attention (masked)")

# ---------------------------------------------------------------------------
# Highlight diagonal dominance
# ---------------------------------------------------------------------------
diag_full = w_full.diag()
diag_causal = w_causal.diag()
off_diag_full = (w_full.sum(dim=-1) - diag_full) / (seq_len - 1)
off_diag_causal_avg = []
for i in range(seq_len):
    if i == 0:
        off_diag_causal_avg.append(0.0)
    else:
        off_diag_causal_avg.append((w_causal[i, :i].sum() / i).item())

print(f"{BOLD}Diagonal dominance analysis:{RESET}")
print(f"  {'Token':<6} {'Diag(full)':>10} {'Avg off(full)':>13}  {'Diag(causal)':>12} {'Avg off(causal)':>15}")
for i, word in enumerate(sentence):
    print(f"  {word:<6} {diag_full[i]:>10.3f} {off_diag_full[i]:>13.3f}  "
          f"{diag_causal[i]:>12.3f} {off_diag_causal_avg[i]:>15.3f}")
print()

# ---------------------------------------------------------------------------
# Why does "the" attend to "the"? (same token = similar Q and K)
# ---------------------------------------------------------------------------
print(f"{BOLD}Why 'the' (pos 0) and 'the' (pos 4) attend to each other:{RESET}")
scores_raw = Q[0] @ K[0].T / math.sqrt(head_dim)
print(f"  Score matrix (scaled):")
for i in range(seq_len):
    row = "  ".join(f"{scores_raw[i,j]:>6.2f}" for j in range(seq_len))
    print(f"    {sentence[i]:>5s}  [{row}]")
print()
print(f"  'the'(0) → 'the'(4) score: {scores_raw[0,4]:.3f}")
print(f"  'the'(0) → 'cat'(1) score: {scores_raw[0,1]:.3f}")
print(f"  Same words produce similar Q and K vectors → higher dot product")
print()

# ---------------------------------------------------------------------------
# What breaks without masking? Show information leakage
# ---------------------------------------------------------------------------
print(f"{'='*60}")
print(f"{BOLD}What breaks if you remove masking?{RESET}")
print(f"{'='*60}")
print()

# Demonstrate: position 0 ("the") should only know about itself in causal
print(f"  In causal mode, 'the' (pos 0) can only see itself:")
print(f"    weights: {w_causal[0].tolist()}")
print(f"    → output is just its own value vector")
print()
print(f"  Without mask, 'the' (pos 0) sees the ENTIRE sentence:")
print(f"    weights: {[f'{v:.2f}' for v in w_full[0].tolist()]}")
print(f"    → it already 'knows' that 'mat' comes at the end!")
print()
print(f"  This is {BOLD}information leakage{RESET} — the model can cheat:")
print(f"    - To predict 'cat' after 'the', it peeks at 'cat' in the future")
print(f"    - Training loss drops easily (the answer is in the input)")
print(f"    - But at generation time, future tokens don't exist yet")
print(f"    - The model has learned to rely on information that won't be there")
print()

# Quantify: how much future information does each position see?
print(f"  {BOLD}Future attention (information leakage per position):{RESET}")
for i in range(seq_len):
    future_weight = w_full[i, i+1:].sum().item()
    past_weight = w_full[i, :i+1].sum().item()
    print(f"    '{sentence[i]}' (pos {i}): {future_weight:.1%} of attention goes to future tokens")
print()

# ---------------------------------------------------------------------------
# Save attention heatmaps
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, weights, title in [
    (axes[0], w_full, "Full Attention (no mask)"),
    (axes[1], w_causal, "Causal Attention (masked)"),
]:
    im = ax.imshow(weights.numpy(), cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(sentence, rotation=45, ha='right')
    ax.set_yticklabels(sentence)
    ax.set_xlabel("Key (attending to)")
    ax.set_ylabel("Query (attending from)")
    ax.set_title(title)

    # Annotate cells with values
    for i in range(seq_len):
        for j in range(seq_len):
            val = weights[i, j].item()
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=color)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'attention_heatmaps.png')
plt.savefig(out_path, dpi=150)
print(f"Saved attention heatmaps to {out_path}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"""
{'='*60}
{BOLD}STEP 2 SUMMARY{RESET}
{'='*60}

{BOLD}Why does self-attention (diagonal) dominate?{RESET}
  Each token's Q and K come from the same embedding.
  A token is always the best match for itself because
  Q_i · K_i is maximized when Q and K share the same source.
  This is especially strong with random (untrained) weights.

{BOLD}What breaks without masking?{RESET}
  Without causal masking, every position sees the full sequence
  including future tokens. This causes:
    1. Information leakage — the model peeks at the answer
    2. Training loss is artificially low (cheating)
    3. At generation time, no future exists → distribution mismatch
    4. The model fails because it learned to depend on future context

  The causal mask is what makes autoregressive generation possible.
""")
