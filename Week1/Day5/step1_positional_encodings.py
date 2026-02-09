"""
Step 1: Compare positional encoding types.

Three approaches to telling the model WHERE each token is:

  1. Absolute PE  — add position info to embeddings (before attention)
  2. Relative PE  — add position info to attention scores (inside attention)
  3. RoPE         — rotate Q,K vectors by position (inside attention)

Each answers: what changes in attention computation?
              where is position information injected?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Day1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Day2'))

from train_tiny_char_level_lm import vocab_size, encode, decode, data, CONTEXT_LEN
from step1_scaled_dot_product_attention import make_causal_mask

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

# ===================================================================
# 1. ABSOLUTE POSITIONAL ENCODING
# ===================================================================
# Two variants: learned embeddings and sinusoidal (Vaswani et al. 2017)

class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (original Transformer).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    No learned parameters — position info is deterministic.
    """
    def __init__(self, max_len, embed_dim):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()     # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )  # (embed_dim/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions
        self.register_buffer('pe', pe)  # not a parameter, but moves with device

    def forward(self, seq_len):
        return self.pe[:seq_len]  # (C, E)


class LearnedPE(nn.Module):
    """Learned positional embedding (GPT-2 style).

    Each position gets its own learned vector.
    Simple, flexible, but doesn't generalize beyond max_len.
    """
    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.pe = nn.Embedding(max_len, embed_dim)

    def forward(self, seq_len):
        return self.pe(torch.arange(seq_len, device=self.pe.weight.device))  # (C, E)


# ===================================================================
# 2. RELATIVE POSITIONAL ENCODING
# ===================================================================

class RelativeAttention(nn.Module):
    """Single-head attention with relative position bias.

    Instead of adding position to embeddings, we add a learned bias
    to attention scores based on the RELATIVE distance between tokens.

    scores[i, j] = (Q_i · K_j) / sqrt(D) + bias[i - j]

    This means attention directly knows "how far apart" two tokens are.
    """
    def __init__(self, embed_dim, max_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        # Learned bias for each possible relative distance
        # Distance ranges from -(max_len-1) to +(max_len-1)
        # But with causal mask, we only need 0 to -(max_len-1)
        # (j <= i, so i-j >= 0)
        self.rel_bias = nn.Embedding(2 * max_len - 1, 1)
        self.max_len = max_len

    def forward(self, x, mask=None):
        B, C, E = x.shape
        Q = self.W_q(x)  # (B, C, E)
        K = self.W_k(x)
        V = self.W_v(x)

        # Standard QK scores
        scores = Q @ K.transpose(-2, -1) / math.sqrt(E)  # (B, C, C)

        # Add relative position bias
        # Build relative distance matrix: rel[i,j] = i - j + (max_len - 1)
        positions = torch.arange(C, device=x.device)
        rel_dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # (C, C)
        rel_dist = rel_dist + (self.max_len - 1)  # shift to positive indices
        rel_dist = rel_dist.clamp(0, 2 * self.max_len - 2)

        bias = self.rel_bias(rel_dist).squeeze(-1)  # (C, C)
        scores = scores + bias

        if mask is not None:
            scores = scores + mask

        weights = F.softmax(scores, dim=-1)
        return weights @ V, weights


# ===================================================================
# 3. RoPE (Rotary Positional Embedding)
# ===================================================================

def build_rope_freqs(dim, max_len, base=10000.0):
    """Precompute the rotation frequencies for RoPE.

    For dimension pair (2i, 2i+1):
      θ_i = 1 / base^(2i/dim)
      angle at position m = m * θ_i

    Returns complex exponentials: e^(i·m·θ) for each position and dim pair.
    """
    # θ_i for each pair of dimensions
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # (dim/2,)
    # Position indices
    positions = torch.arange(max_len).float()  # (max_len,)
    # Outer product: angle[m, i] = m * θ_i
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (max_len, dim/2)
    # Complex exponential: cos(angle) + i·sin(angle)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)  # (max_len, dim/2)
    return freqs_cis


def apply_rope(x, freqs_cis):
    """Apply rotary embedding to a tensor.

    x: (B, C, D) real tensor
    freqs_cis: (C, D/2) complex tensor

    Treats pairs of dims as complex numbers, multiplies by rotation.
    """
    B, C, D = x.shape
    # View as complex: (B, C, D) → (B, C, D/2) complex
    x_complex = torch.view_as_complex(x.float().reshape(B, C, D // 2, 2))
    # Multiply by rotation (broadcasts over batch)
    x_rotated = x_complex * freqs_cis[:C].unsqueeze(0)
    # Back to real: (B, C, D/2) complex → (B, C, D) real
    return torch.view_as_real(x_rotated).reshape(B, C, D).type_as(x)


class RoPEAttention(nn.Module):
    """Single-head attention with Rotary Positional Embedding.

    Position info is injected by ROTATING Q and K vectors.
    The rotation angle depends on position, so:
      Q_m · K_n = f(q, k, m-n)
    The dot product naturally becomes a function of relative position.

    No position added to embeddings. No bias in attention scores.
    Position lives in the GEOMETRY of Q and K.
    """
    def __init__(self, embed_dim, max_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.register_buffer('freqs_cis', build_rope_freqs(embed_dim, max_len))

    def forward(self, x, mask=None):
        B, C, E = x.shape
        Q = self.W_q(x)  # (B, C, E)
        K = self.W_k(x)
        V = self.W_v(x)

        # Apply rotation to Q and K (but NOT V)
        Q = apply_rope(Q, self.freqs_cis)
        K = apply_rope(K, self.freqs_cis)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(E)

        if mask is not None:
            scores = scores + mask

        weights = F.softmax(scores, dim=-1)
        return weights @ V, weights


# ===================================================================
# DEMONSTRATIONS
# ===================================================================

print(f"""
{'='*65}
{BOLD}POSITIONAL ENCODING: THREE APPROACHES{RESET}
{'='*65}
""")

# -------------------------------------------------------------------
# Demo 1: Visualize the sinusoidal PE pattern
# -------------------------------------------------------------------
print(f"{BOLD}1. ABSOLUTE PE — Sinusoidal Pattern{RESET}")
print(f"{'='*65}")

E = 64
sin_pe = SinusoidalPE(CONTEXT_LEN, E)
pe_matrix = sin_pe(CONTEXT_LEN).numpy()  # (32, 64)

print(f"  PE shape: ({CONTEXT_LEN}, {E})")
print(f"  Each position gets a unique {E}-dim vector.")
print()

# Show that nearby positions have high similarity, far positions don't
print(f"  Cosine similarity between positions (sinusoidal):")
pe_tensor = sin_pe(CONTEXT_LEN)
pe_norm = F.normalize(pe_tensor, dim=-1)
sim = (pe_norm @ pe_norm.T).numpy()

for p in [0, 1, 2, 5, 15, 31]:
    sims = [f"{sim[0, p]:.3f}"]
    print(f"    pos 0 vs pos {p:>2}: {sim[0, p]:+.4f}")

print()
print(f"  {CYAN}Key property:{RESET} similarity decays smoothly with distance.")
print(f"  The model can learn that nearby positions are related.")
print()

# Compare with learned PE
learned_pe = LearnedPE(CONTEXT_LEN, E)
lpe = learned_pe(CONTEXT_LEN).detach()
lpe_norm = F.normalize(lpe, dim=-1)
lsim = (lpe_norm @ lpe_norm.T).numpy()

print(f"  Learned PE similarity (before training — random init):")
for p in [0, 1, 2, 5, 15, 31]:
    print(f"    pos 0 vs pos {p:>2}: {lsim[0, p]:+.4f}")
print(f"  {DIM}(random at init — learns structure during training){RESET}")
print()

# -------------------------------------------------------------------
# Demo 2: Relative PE — show the bias matrix
# -------------------------------------------------------------------
print(f"{BOLD}2. RELATIVE PE — Distance-Based Bias{RESET}")
print(f"{'='*65}")
print()

torch.manual_seed(42)
rel_attn = RelativeAttention(E, CONTEXT_LEN)

# Show the bias matrix
positions = torch.arange(8)
rel_dist = positions.unsqueeze(0) - positions.unsqueeze(1) + (CONTEXT_LEN - 1)
rel_dist = rel_dist.clamp(0, 2 * CONTEXT_LEN - 2)
bias = rel_attn.rel_bias(rel_dist).squeeze(-1).detach().numpy()

print(f"  Relative bias matrix (first 8 positions, before training):")
print(f"  {'':>6}", end="")
for j in range(8):
    print(f"  j={j}", end="")
print()
for i in range(8):
    print(f"  i={i:>2}  ", end="")
    for j in range(8):
        dist = i - j
        val = bias[i, j]
        print(f" {val:+.2f}", end="")
    print(f"   (distances: {[i-j for j in range(8)]})")

print()
print(f"  {CYAN}How it works:{RESET}")
print(f"    scores[i,j] = Q_i·K_j/√D {GREEN}+ bias[i-j]{RESET}")
print(f"    The bias term depends ONLY on relative distance (i-j).")
print(f"    Position 5→3 and position 10→8 get the SAME bias (distance=2).")
print()

# -------------------------------------------------------------------
# Demo 3: RoPE — show the rotation
# -------------------------------------------------------------------
print(f"{BOLD}3. RoPE — Rotation in Complex Plane{RESET}")
print(f"{'='*65}")
print()

torch.manual_seed(42)
rope_attn = RoPEAttention(E, CONTEXT_LEN)

# Show that RoPE makes dot product depend on relative position
print(f"  {BOLD}Proof: Q_m · K_n depends only on (m - n){RESET}")
print()

# Create identical content at different positions
x = torch.randn(1, 8, E)
Q_raw = rope_attn.W_q(x)
K_raw = rope_attn.W_k(x)

# Apply RoPE
Q_rope = apply_rope(Q_raw, rope_attn.freqs_cis)
K_rope = apply_rope(K_raw, rope_attn.freqs_cis)

# Without RoPE: Q_i · K_j depends on absolute positions i, j
# With RoPE: Q_i · K_j depends on relative position (i - j)

# Test: compare dot products with same relative distance
print(f"  Dot products between positions (with RoPE):")
print(f"    Same relative distance should give similar values.")
print()

pairs_by_dist = {}
for i in range(6):
    for j in range(6):
        dist = i - j
        dot = (Q_rope[0, i] * K_rope[0, j]).sum().item()
        if dist not in pairs_by_dist:
            pairs_by_dist[dist] = []
        pairs_by_dist[dist].append((i, j, dot))

print(f"  {'Distance':<12} {'Pairs':<30} {'Dot products'}")
print(f"  {'-'*12} {'-'*30} {'-'*25}")
for dist in sorted(pairs_by_dist.keys()):
    pairs = pairs_by_dist[dist][:3]  # show up to 3
    pair_str = ", ".join(f"({i},{j})" for i, j, _ in pairs)
    dot_str = ", ".join(f"{d:.2f}" for _, _, d in pairs)
    # Check how consistent the dots are
    dots = [d for _, _, d in pairs_by_dist[dist]]
    std = np.std(dots) if len(dots) > 1 else 0
    print(f"  {dist:>+3}          {pair_str:<30} {dot_str}  {DIM}(std={std:.3f}){RESET}")

print()
print(f"  {CYAN}Same distance → similar dot products.{RESET}")
print(f"  The rotation encodes position WITHOUT adding to the embedding.")
print(f"  Position lives in the ANGLE of Q and K, not their magnitude.")
print()

# Show the rotation angles for first few dims
print(f"  Rotation angles (first 4 dim pairs, first 8 positions):")
freqs = rope_attn.freqs_cis[:8]  # (8, D/2) complex
angles = freqs.angle()[:, :4].numpy()  # radians

print(f"  {'Pos':<6}", end="")
for d in range(4):
    print(f"  dim({2*d},{2*d+1})", end="")
print()
for pos in range(8):
    print(f"  {pos:<6}", end="")
    for d in range(4):
        print(f"  {angles[pos, d]:>8.4f}", end="")
    print()

print()
print(f"  {DIM}Low dims rotate fast (capture fine-grained position).")
print(f"  High dims rotate slowly (capture long-range position).{RESET}")
print()

# ===================================================================
# EXPERIMENT: Train and compare all three
# ===================================================================

class AbsolutePE_LM(nn.Module):
    """LM with absolute (learned) positional encoding."""
    def __init__(self, vocab_size, context_len, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = LearnedPE(context_len, embed_dim)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, vocab_size)
        self.context_len = context_len
        self.embed_dim = embed_dim

    def forward(self, x):
        B, C = x.shape
        h = self.embed(x) + self.pos_embed(C)  # position added to embedding
        Q, K, V = self.W_q(h), self.W_k(h), self.W_v(h)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.embed_dim)
        scores = scores + make_causal_mask(C).to(x.device)
        attn_out = F.softmax(scores, dim=-1) @ V
        return self.out_proj(h + attn_out)


class RelativePE_LM(nn.Module):
    """LM with relative positional encoding (bias in attention)."""
    def __init__(self, vocab_size, context_len, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # NO positional embedding added to tokens
        self.attn = RelativeAttention(embed_dim, context_len)
        self.out_proj = nn.Linear(embed_dim, vocab_size)
        self.context_len = context_len

    def forward(self, x):
        B, C = x.shape
        h = self.embed(x)  # no position added here
        mask = make_causal_mask(C).to(x.device)
        attn_out, _ = self.attn(h, mask)
        return self.out_proj(h + attn_out)


class RoPE_LM(nn.Module):
    """LM with Rotary Positional Embedding."""
    def __init__(self, vocab_size, context_len, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # NO positional embedding added to tokens
        self.attn = RoPEAttention(embed_dim, context_len)
        self.out_proj = nn.Linear(embed_dim, vocab_size)
        self.context_len = context_len

    def forward(self, x):
        B, C = x.shape
        h = self.embed(x)  # no position added here
        mask = make_causal_mask(C).to(x.device)
        attn_out, _ = self.attn(h, mask)
        return self.out_proj(h + attn_out)


def train_and_eval(model, name, steps=800):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for step in range(1, steps + 1):
        ix = torch.randint(0, len(data) - CONTEXT_LEN - 1, (128,))
        x = torch.stack([data[i : i + CONTEXT_LEN] for i in ix])
        y = torch.stack([data[i + 1 : i + CONTEXT_LEN + 1] for i in ix])
        loss = F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    final_loss = np.mean(losses[-50:])
    print(f"    {name:<20} final loss: {final_loss:.4f}")
    return losses, final_loss


print(f"{'='*65}")
print(f"{BOLD}TRAINING COMPARISON: 1-layer LMs with different PE{RESET}")
print(f"{'='*65}")
print()

torch.manual_seed(42)
model_abs = AbsolutePE_LM(vocab_size, CONTEXT_LEN, E)
losses_abs, fl_abs = train_and_eval(model_abs, "Absolute (learned)")

torch.manual_seed(42)
model_rel = RelativePE_LM(vocab_size, CONTEXT_LEN, E)
losses_rel, fl_rel = train_and_eval(model_rel, "Relative (bias)")

torch.manual_seed(42)
model_rope = RoPE_LM(vocab_size, CONTEXT_LEN, E)
losses_rope, fl_rope = train_and_eval(model_rope, "RoPE (rotation)")

print()

# Generate samples
def gen(model, seed="To be, or not to be", length=60):
    model.eval()
    tokens = encode(seed)
    generated = list(tokens)
    with torch.no_grad():
        for _ in range(length):
            inp = torch.tensor([generated[-CONTEXT_LEN:]], dtype=torch.long)
            logits = model(inp)
            probs = F.softmax(logits[0, -1], dim=0)
            generated.append(torch.multinomial(probs, 1).item())
    return decode(generated[len(tokens):])

torch.manual_seed(99)
gen_abs = gen(model_abs)
torch.manual_seed(99)
gen_rel = gen(model_rel)
torch.manual_seed(99)
gen_rope = gen(model_rope)

print(f"  Generation samples:")
print(f"    Absolute: \"{gen_abs[:70]}\"")
print(f"    Relative: \"{gen_rel[:70]}\"")
print(f"    RoPE:     \"{gen_rope[:70]}\"")
print()

# ===================================================================
# Visualization
# ===================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Positional Encoding Comparison", fontsize=14, fontweight='bold')

# Panel 1: Sinusoidal PE heatmap
ax = axes[0, 0]
im = ax.imshow(pe_matrix[:16, :32], aspect='auto', cmap='RdBu_r')
ax.set_xlabel("Embedding Dimension")
ax.set_ylabel("Position")
ax.set_title("Sinusoidal PE Pattern (first 16 pos, 32 dims)")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Panel 2: Position similarity for sinusoidal PE
ax = axes[0, 1]
im = ax.imshow(sim[:16, :16], cmap='RdYlBu_r', vmin=-1, vmax=1)
ax.set_xlabel("Position")
ax.set_ylabel("Position")
ax.set_title("Sinusoidal PE: Cosine Similarity")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Panel 3: Training curves
ax = axes[1, 0]
window = 30
for losses, label, color in [
    (losses_abs, 'Absolute', 'steelblue'),
    (losses_rel, 'Relative', 'darkorange'),
    (losses_rope, 'RoPE', 'green'),
]:
    smooth = np.convolve(losses, np.ones(window)/window, mode='valid')
    ax.plot(smooth, label=label, color=color, linewidth=2)
ax.set_xlabel("Training Step")
ax.set_ylabel("Loss")
ax.set_title("Training Curves by PE Type")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 4: RoPE rotation angles
ax = axes[1, 1]
for d in range(min(6, E // 2)):
    angles_d = rope_attn.freqs_cis[:CONTEXT_LEN].angle()[:, d].numpy()
    alpha = 1.0 if d < 3 else 0.4
    ax.plot(angles_d, label=f'dim pair {d}', linewidth=2, alpha=alpha)
ax.set_xlabel("Position")
ax.set_ylabel("Rotation Angle (radians)")
ax.set_title("RoPE: Rotation Angles by Dimension")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), 'positional_encodings.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved visualization → {save_path}")
print()

# ===================================================================
# Summary notes
# ===================================================================
print(f"""
{'='*65}
{BOLD}STEP 1 SUMMARY: POSITIONAL ENCODING COMPARISON{RESET}
{'='*65}

{BOLD}1. Absolute PE (Learned / Sinusoidal){RESET}

  {CYAN}Where injected:{RESET}  Added to token embeddings BEFORE attention.
     h = Embed(token) {GREEN}+ PE(position){RESET}

  {CYAN}What changes:{RESET}   Nothing in attention itself.
     Q, K, V all see position mixed into the input.
     scores = (Q·K)/√D  — standard dot product.

  {CYAN}Pros:{RESET}  Simple. Learned PE is very flexible.
  {CYAN}Cons:{RESET}  Fixed max length. No relative awareness.
         Position 5 and 6 being "neighbors" must be learned,
         not built into the architecture.

{BOLD}2. Relative PE (Attention Bias){RESET}

  {CYAN}Where injected:{RESET}  Added to attention SCORES as a bias.
     scores[i,j] = (Q_i·K_j)/√D {GREEN}+ bias[i-j]{RESET}

  {CYAN}What changes:{RESET}   Attention computation gains a position term.
     No position in embeddings — tokens are position-agnostic
     until attention explicitly adds the distance bias.

  {CYAN}Pros:{RESET}  Naturally captures relative distance.
         "2 tokens apart" means the same at position 5 or 50.
  {CYAN}Cons:{RESET}  Extra learned parameters (one bias per distance).
         Can be expensive for long sequences.

{BOLD}3. RoPE (Rotary Positional Embedding){RESET}

  {CYAN}Where injected:{RESET}  Q and K are ROTATED before dot product.
     Q_m = rotate(Q, m·θ)       {GREEN}← rotation by position × frequency{RESET}
     K_n = rotate(K, n·θ)
     scores = Q_m · K_n  →  depends on (m-n) automatically

  {CYAN}What changes:{RESET}   Q,K geometry is modified. V is untouched.
     The dot product Q_m·K_n naturally decomposes into a function
     of content (q,k) and relative position (m-n).

  {CYAN}Pros:{RESET}  No extra parameters. Relative by construction.
         Generalizes to unseen lengths (extrapolation).
         Used in LLaMA, Mistral, GPT-NeoX, and most modern LLMs.
  {CYAN}Cons:{RESET}  Slightly harder to understand (complex numbers).

{BOLD}Comparison:{RESET}

  ┌─────────────┬────────────────┬──────────────┬───────────────┐
  │             │ Absolute       │ Relative     │ RoPE          │
  ├─────────────┼────────────────┼──────────────┼───────────────┤
  │ Injected at │ Embedding      │ Attn scores  │ Q,K rotation  │
  │ Awareness   │ Absolute only  │ Relative     │ Relative      │
  │ Parameters  │ max_len × E    │ 2×max_len-1  │ 0 (none)      │
  │ Extrapolate │ No             │ Partially    │ Yes           │
  │ Used in     │ GPT-2, BERT    │ T5, DeBERTa  │ LLaMA, Mistral│
  └─────────────┴────────────────┴──────────────┴───────────────┘

{BOLD}The evolution:{RESET}
  Absolute PE was the starting point (2017 Transformer).
  Relative PE improved generalization (T5, 2020).
  RoPE unified both advantages with no extra cost (2021+).
  Today, nearly all large LLMs use RoPE.
""")
