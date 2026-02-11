"""
Step 2: Visualize head patterns — do heads specialize or collapse?

Trains a multi-head attention LM, then visualizes each head's attention
pattern on a real sentence to see if different heads learn different roles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Day1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Day2'))
sys.path.insert(0, os.path.dirname(__file__))

from train_tiny_char_level_lm import vocab_size, encode, decode, data, CONTEXT_LEN
from step1_scaled_dot_product_attention import make_causal_mask
from step1_multihead_attention import MultiHeadAttention

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Multi-head attention LM
# ---------------------------------------------------------------------------
class MultiHeadAttentionLM(nn.Module):
    def __init__(self, vocab_size, context_len, embed_dim, num_heads):
        super().__init__()
        self.context_len = context_len
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(context_len, embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.out_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, C = x.shape
        emb = self.embed(x) + self.pos_embed(torch.arange(C, device=x.device))
        mask = make_causal_mask(C).to(x.device)
        attn_out, self._attn_weights = self.attn(emb, mask)
        return self.out_proj(attn_out)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
EMBED_DIM = 64
NUM_HEADS = 4
STEPS = 800

print(f"{BOLD}Training multi-head attention LM ({NUM_HEADS} heads)...{RESET}")
torch.manual_seed(42)
model = MultiHeadAttentionLM(vocab_size, CONTEXT_LEN, EMBED_DIM, NUM_HEADS)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

for step in range(1, STEPS + 1):
    ix = torch.randint(0, len(data) - CONTEXT_LEN - 1, (128,))
    x = torch.stack([data[i : i + CONTEXT_LEN] for i in ix])
    y = torch.stack([data[i + 1 : i + CONTEXT_LEN + 1] for i in ix])
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 200 == 0:
        print(f"  Step {step:>4d} | Loss: {loss.item():.4f}")

print()

# ---------------------------------------------------------------------------
# Get attention patterns on a real sentence
# ---------------------------------------------------------------------------
test_text = "To be, or not to be, that is the"
test_tokens = encode(test_text)
test_input = torch.tensor([test_tokens], dtype=torch.long)
test_chars = list(test_text)

model.eval()
with torch.no_grad():
    _ = model(test_input)
    # _attn_weights: (1, H, C, C)
    all_weights = model._attn_weights[0]  # (H, C, C)

H = NUM_HEADS
C = len(test_chars)

# ---------------------------------------------------------------------------
# Analyze each head's behavior
# ---------------------------------------------------------------------------
print(f"{'='*65}")
print(f"{BOLD}HEAD PATTERN ANALYSIS on: \"{test_text}\"{RESET}")
print(f"{'='*65}\n")

head_labels = []

for h in range(H):
    w = all_weights[h]  # (C, C)

    # Metric 1: average attention distance (how far back does it look?)
    positions = torch.arange(C, dtype=torch.float)
    avg_dist = 0.0
    for i in range(C):
        # weighted average distance from position i
        dists = (positions[i] - positions[:i+1]).float()
        avg_dist += (w[i, :i+1] * dists).sum().item()
    avg_dist /= C

    # Metric 2: diagonal dominance (self-attention strength)
    diag_weight = w.diag().mean().item()

    # Metric 3: neighbor attention (positions i-1, i-2)
    neighbor_weight = 0.0
    count = 0
    for i in range(1, C):
        neighbor_weight += w[i, i-1].item()
        count += 1
    neighbor_weight /= count

    # Metric 4: "the"-to-"the" attention (repeated word pattern)
    the_positions = [i for i in range(C - 2)
                     if test_text[i:i+3] == "the" or test_text[i:i+3] == " th"]
    repeated_attn = 0.0
    if len(the_positions) >= 2:
        for i in the_positions:
            for j in the_positions:
                if j < i:
                    repeated_attn += w[i, j].item()

    # Metric 5: entropy (how spread out is attention?)
    entropy = -(w * (w + 1e-10).log()).sum(-1).mean().item()

    # Classify the head
    if diag_weight > 0.3:
        label = "SELF-FOCUSED"
    elif avg_dist < 2.0:
        label = "LOCAL (nearby)"
    elif entropy > 2.0:
        label = "BROAD (distributed)"
    else:
        label = "SELECTIVE"
    head_labels.append(label)

    print(f"  {BOLD}Head {h}: {label}{RESET}")
    print(f"    Avg attention distance: {avg_dist:.2f} positions back")
    print(f"    Self-attention (diag):  {diag_weight:.3f}")
    print(f"    Prev-token attention:   {neighbor_weight:.3f}")
    print(f"    Entropy:                {entropy:.3f}")
    if repeated_attn > 0:
        print(f"    Repeated-word attention: {repeated_attn:.3f}")
    print()

# ---------------------------------------------------------------------------
# Head similarity (redundancy check)
# ---------------------------------------------------------------------------
print(f"{'='*65}")
print(f"{BOLD}HEAD REDUNDANCY CHECK{RESET}")
print(f"{'='*65}")
print(f"  Pairwise cosine similarity (averaged over all positions):\n")

sim_matrix = torch.zeros(H, H)
for h1 in range(H):
    for h2 in range(H):
        # Average cosine similarity across all query positions
        sims = F.cosine_similarity(
            all_weights[h1],  # (C, C)
            all_weights[h2],  # (C, C)
            dim=-1             # compare along key dimension
        )
        sim_matrix[h1, h2] = sims.mean()

header = "         " + "  ".join(f"Head {h}" for h in range(H))
print(f"  {header}")
for h1 in range(H):
    row = "  ".join(f" {sim_matrix[h1, h2]:.3f}" for h2 in range(H))
    print(f"  Head {h1}   {row}")

# Check for collapsed heads (very high similarity)
collapsed = []
for h1 in range(H):
    for h2 in range(h1 + 1, H):
        if sim_matrix[h1, h2] > 0.95:
            collapsed.append((h1, h2))

print()
if collapsed:
    print(f"  {BOLD}Collapsed heads:{RESET} {collapsed}")
    print(f"  These heads learned nearly identical patterns — wasted capacity.")
else:
    print(f"  {GREEN}No collapsed heads — all {H} heads are distinct.{RESET}")
print()

# ---------------------------------------------------------------------------
# Plot attention heatmaps per head
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, H, figsize=(5 * H, 5))
if H == 1:
    axes = [axes]

for h in range(H):
    ax = axes[h]
    w = all_weights[h].numpy()
    im = ax.imshow(w, cmap='Blues', vmin=0, vmax=w.max())
    ax.set_xticks(range(C))
    ax.set_yticks(range(C))
    ax.set_xticklabels(test_chars, fontsize=7, rotation=90)
    ax.set_yticklabels(test_chars, fontsize=7)
    ax.set_title(f"Head {h}\n{head_labels[h]}", fontsize=10)
    if h == 0:
        ax.set_ylabel("Query (from)")
    ax.set_xlabel("Key (to)")

plt.suptitle(f"Multi-Head Attention Patterns ({H} heads)", fontsize=12, y=1.02)
plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'head_patterns.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved head pattern heatmaps to {out_path}")

# ---------------------------------------------------------------------------
# Write-up
# ---------------------------------------------------------------------------
print(f"""
{'='*65}
{BOLD}DO HEADS SPECIALIZE OR COLLAPSE?{RESET}
{'='*65}

{BOLD}What we observed:{RESET}
  Different heads develop different attention strategies:
  - Some focus on the immediately previous token (local/bigram)
  - Some attend broadly across the sequence (distributed)
  - Some focus heavily on self (diagonal dominant)
  - Some look for specific patterns (selective)

{BOLD}Specialization:{RESET}
  With enough training, heads tend to specialize because:
  - The output projection W_o learns to combine head outputs
  - Redundant heads waste capacity — gradient pressure pushes
    them toward covering different patterns
  - But with a tiny corpus, specialization may be weak

{BOLD}Collapse:{RESET}
  Heads can collapse (become identical) when:
  - The model is too large for the data (not enough signal)
  - Learning rate is too high (heads converge to same local minimum)
  - Not enough diversity in the data to warrant multiple strategies

{BOLD}Why this matters:{RESET}
  In real transformers (GPT, etc.), heads clearly specialize:
  - "Previous token" heads, "induction" heads, "syntax" heads
  - Some heads are prunable (redundant) — model compression
  - Understanding heads helps interpret what the model "sees"
""")
