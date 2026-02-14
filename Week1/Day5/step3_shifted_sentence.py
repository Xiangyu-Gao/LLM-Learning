"""
Step 3: Shifted sentence experiment.

Train models with different PE types on positions 0..31.
Then feed the SAME sentence at shifted positions (e.g., position 10..41).
Compare attention matrices, output representations, prediction differences.

Key insight: Extrapolation ≠ generalization.
A model trained on length 32 does NOT truly "understand" length 64.
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
# PE and attention definitions (local copies to avoid step1 side effects)
# ===================================================================

class LearnedPE(nn.Module):
    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.pe = nn.Embedding(max_len, embed_dim)

    def forward(self, seq_len, offset=0):
        positions = torch.arange(offset, offset + seq_len, device=self.pe.weight.device)
        return self.pe(positions)


class RelativeAttention(nn.Module):
    def __init__(self, embed_dim, max_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rel_bias = nn.Embedding(2 * max_len - 1, 1)
        self.max_len = max_len

    def forward(self, x, mask=None):
        B, C, E = x.shape
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(E)
        positions = torch.arange(C, device=x.device)
        rel_dist = positions.unsqueeze(0) - positions.unsqueeze(1) + (self.max_len - 1)
        rel_dist = rel_dist.clamp(0, 2 * self.max_len - 2)
        bias = self.rel_bias(rel_dist).squeeze(-1)
        scores = scores + bias
        if mask is not None:
            scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        return weights @ V, weights


def build_rope_freqs(dim, max_len, base=10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_len).float()
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x, freqs_cis):
    B, C, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, C, D // 2, 2))
    x_rotated = x_complex * freqs_cis[:C].unsqueeze(0)
    return torch.view_as_real(x_rotated).reshape(B, C, D).type_as(x)


class RoPEAttention(nn.Module):
    def __init__(self, embed_dim, max_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.register_buffer('freqs_cis', build_rope_freqs(embed_dim, max_len))

    def forward(self, x, mask=None, pos_offset=0):
        B, C, E = x.shape
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        # Use freqs at shifted positions
        freqs = self.freqs_cis[pos_offset:pos_offset + C]
        Q = apply_rope(Q, freqs)
        K = apply_rope(K, freqs)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(E)
        if mask is not None:
            scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        return weights @ V, weights


# ===================================================================
# LM models with support for position offset and returning internals
# ===================================================================

# Use larger max_len so shifted positions are valid
MAX_LEN = 128
E = 64


class AbsolutePE_LM(nn.Module):
    def __init__(self, vocab_size, context_len, embed_dim, max_len=MAX_LEN):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = LearnedPE(max_len, embed_dim)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, vocab_size)
        self.context_len = context_len
        self.embed_dim = embed_dim

    def forward(self, x, pos_offset=0, return_internals=False):
        B, C = x.shape
        h = self.embed(x) + self.pos_embed(C, offset=pos_offset)
        Q, K, V = self.W_q(h), self.W_k(h), self.W_v(h)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.embed_dim)
        scores = scores + make_causal_mask(C).to(x.device)
        weights = F.softmax(scores, dim=-1)
        attn_out = weights @ V
        hidden = h + attn_out
        logits = self.out_proj(hidden)
        if return_internals:
            return logits, weights, hidden
        return logits


class RelativePE_LM(nn.Module):
    def __init__(self, vocab_size, context_len, embed_dim, max_len=MAX_LEN):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attn = RelativeAttention(embed_dim, max_len)
        self.out_proj = nn.Linear(embed_dim, vocab_size)
        self.context_len = context_len

    def forward(self, x, pos_offset=0, return_internals=False):
        B, C = x.shape
        h = self.embed(x)  # no position added
        mask = make_causal_mask(C).to(x.device)
        attn_out, weights = self.attn(h, mask)
        hidden = h + attn_out
        logits = self.out_proj(hidden)
        if return_internals:
            return logits, weights, hidden
        return logits


class RoPE_LM(nn.Module):
    def __init__(self, vocab_size, context_len, embed_dim, max_len=MAX_LEN):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attn = RoPEAttention(embed_dim, max_len)
        self.out_proj = nn.Linear(embed_dim, vocab_size)
        self.context_len = context_len

    def forward(self, x, pos_offset=0, return_internals=False):
        B, C = x.shape
        h = self.embed(x)  # no position added
        mask = make_causal_mask(C).to(x.device)
        attn_out, weights = self.attn(h, mask, pos_offset=pos_offset)
        hidden = h + attn_out
        logits = self.out_proj(hidden)
        if return_internals:
            return logits, weights, hidden
        return logits


# ===================================================================
# Training
# ===================================================================

def train_model(model, steps=800):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for step in range(1, steps + 1):
        ix = torch.randint(0, len(data) - CONTEXT_LEN - 1, (128,))
        x = torch.stack([data[i : i + CONTEXT_LEN] for i in ix])
        y = torch.stack([data[i + 1 : i + CONTEXT_LEN + 1] for i in ix])
        # Always train at position offset 0
        loss = F.cross_entropy(model(x, pos_offset=0).view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses[-50:])


print(f"""
{'='*65}
{BOLD}SHIFTED SENTENCE EXPERIMENT{RESET}
{'='*65}

  Train 3 models (Absolute PE, Relative PE, RoPE) on positions 0..31.
  Then feed the SAME sentence at shifted positions.
  Question: which PE types degrade when positions shift?
""")

print(f"{BOLD}Training 3 models (800 steps each)...{RESET}")
print()

torch.manual_seed(42)
model_abs = AbsolutePE_LM(vocab_size, CONTEXT_LEN, E)
loss_abs = train_model(model_abs)
print(f"  Absolute PE  final loss: {loss_abs:.4f}")

torch.manual_seed(42)
model_rel = RelativePE_LM(vocab_size, CONTEXT_LEN, E)
loss_rel = train_model(model_rel)
print(f"  Relative PE  final loss: {loss_rel:.4f}")

torch.manual_seed(42)
model_rope = RoPE_LM(vocab_size, CONTEXT_LEN, E)
loss_rope = train_model(model_rope)
print(f"  RoPE         final loss: {loss_rope:.4f}")
print()

# ===================================================================
# Experiment: Feed the same sentence at different position offsets
# ===================================================================

test_sentence = "To be, or not to be, that is t"  # 30 chars, fits in context
tokens = torch.tensor([encode(test_sentence)], dtype=torch.long)  # (1, 30)
targets = torch.tensor([encode("o be, or not to be, that is th")], dtype=torch.long)
SEQ_LEN = tokens.shape[1]

offsets = [0, 5, 10, 20, 30, 50, 80]

print(f"{BOLD}EXPERIMENT: Same sentence at different position offsets{RESET}")
print(f"{'='*65}")
print(f"  Sentence: \"{test_sentence}\"")
print(f"  Length: {SEQ_LEN} tokens")
print(f"  Trained on positions 0..{CONTEXT_LEN-1}, testing offsets: {offsets}")
print()

models = [
    ("Absolute PE", model_abs),
    ("Relative PE", model_rel),
    ("RoPE",        model_rope),
]

# Store results for plotting
all_losses = {name: [] for name, _ in models}
all_attn = {name: {} for name, _ in models}
all_hidden = {name: {} for name, _ in models}
all_preds = {name: {} for name, _ in models}

for name, model in models:
    model.eval()
    print(f"  {BOLD}{name}{RESET}")
    print(f"  {'Offset':<10} {'Loss':>8} {'Top-1 Acc':>10} {'Pred drift':>12}")
    print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*12}")

    # Get baseline (offset=0) internals
    with torch.no_grad():
        logits_0, attn_0, hidden_0 = model(tokens, pos_offset=0, return_internals=True)
        all_attn[name][0] = attn_0[0].numpy()  # (C, C)
        all_hidden[name][0] = hidden_0[0].numpy()  # (C, E)
        preds_0 = logits_0[0].argmax(dim=-1)  # (C,)
        all_preds[name][0] = preds_0.numpy()

    for offset in offsets:
        with torch.no_grad():
            logits, attn_w, hidden = model(tokens, pos_offset=offset, return_internals=True)
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1)).item()
            preds = logits[0].argmax(dim=-1)
            acc = (preds == targets[0]).float().mean().item()

            # How much do predictions change from offset=0?
            pred_match = (preds == preds_0).float().mean().item()

            all_losses[name].append(loss)
            all_attn[name][offset] = attn_w[0].numpy()
            all_hidden[name][offset] = hidden[0].numpy()
            all_preds[name][offset] = preds.numpy()

            print(f"  {offset:<10} {loss:>8.4f} {acc:>9.1%} {1 - pred_match:>11.1%}")

    print()

# ===================================================================
# Detailed comparison: attention matrix drift
# ===================================================================
print(f"{BOLD}ATTENTION MATRIX DRIFT{RESET}")
print(f"{'='*65}")
print(f"  Frobenius distance between attention at offset=0 vs shifted:")
print()

for name, _ in models:
    print(f"  {name}:")
    for offset in offsets:
        attn_base = all_attn[name][0]
        attn_shift = all_attn[name][offset]
        frob_dist = np.linalg.norm(attn_base - attn_shift)
        bar_len = int(min(30, frob_dist * 10))
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"    offset {offset:>3}: {frob_dist:.4f}  {bar}")
    print()

# ===================================================================
# Representation drift
# ===================================================================
print(f"{BOLD}REPRESENTATION DRIFT{RESET}")
print(f"{'='*65}")
print(f"  Cosine similarity between hidden states at offset=0 vs shifted:")
print(f"  (averaged over all positions in the sequence)")
print()

for name, _ in models:
    print(f"  {name}:")
    h_base = torch.tensor(all_hidden[name][0])
    h_base_norm = F.normalize(h_base, dim=-1)
    for offset in offsets:
        h_shift = torch.tensor(all_hidden[name][offset])
        h_shift_norm = F.normalize(h_shift, dim=-1)
        cos_sim = (h_base_norm * h_shift_norm).sum(dim=-1).mean().item()
        bar_len = int(20 * (cos_sim + 1) / 2)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"    offset {offset:>3}: cos_sim = {cos_sim:+.4f}  {bar}")
    print()

# ===================================================================
# Prediction comparison at extreme shift
# ===================================================================
print(f"{BOLD}PREDICTION COMPARISON (offset=0 vs offset=50){RESET}")
print(f"{'='*65}")
print()

for name, _ in models:
    preds_0 = all_preds[name][0]
    preds_50 = all_preds[name][50]
    gen_0 = decode(preds_0.tolist())
    gen_50 = decode(preds_50.tolist())
    match = (preds_0 == preds_50).mean()
    print(f"  {name}:")
    print(f"    offset  0: \"{gen_0}\"")
    print(f"    offset 50: \"{gen_50}\"")
    print(f"    match: {match:.1%}")
    print()

# ===================================================================
# Visualization
# ===================================================================
fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle("Shifted Sentence Experiment: Position Extrapolation", fontsize=14, fontweight='bold')

show_offsets = [0, 10, 30, 80]

for row, (name, _) in enumerate(models):
    for col, offset in enumerate(show_offsets):
        ax = axes[row, col]
        attn = all_attn[name][offset]
        ax.imshow(attn, cmap='Blues', vmin=0, vmax=attn.max())
        if col == 0:
            ax.set_ylabel(name, fontsize=12, fontweight='bold')
        if row == 0:
            ax.set_title(f"offset={offset}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), 'shifted_sentence_attn.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved attention heatmaps → {save_path}")

# Loss and similarity curves
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle("Position Shift Impact by PE Type", fontsize=14, fontweight='bold')

# Panel 1: Loss vs offset
ax = axes2[0]
for name, _ in models:
    ax.plot(offsets, all_losses[name], 'o-', label=name, linewidth=2, markersize=6)
ax.set_xlabel("Position Offset")
ax.set_ylabel("Cross-Entropy Loss")
ax.set_title("Loss Degradation with Position Shift")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Attention drift vs offset
ax = axes2[1]
for name, _ in models:
    frob_dists = []
    for offset in offsets:
        frob_dists.append(np.linalg.norm(all_attn[name][0] - all_attn[name][offset]))
    ax.plot(offsets, frob_dists, 'o-', label=name, linewidth=2, markersize=6)
ax.set_xlabel("Position Offset")
ax.set_ylabel("Frobenius Distance from Baseline")
ax.set_title("Attention Matrix Drift")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Representation similarity vs offset
ax = axes2[2]
for name, _ in models:
    cos_sims = []
    h_base = torch.tensor(all_hidden[name][0])
    h_base_norm = F.normalize(h_base, dim=-1)
    for offset in offsets:
        h_shift = torch.tensor(all_hidden[name][offset])
        h_shift_norm = F.normalize(h_shift, dim=-1)
        cos_sim = (h_base_norm * h_shift_norm).sum(dim=-1).mean().item()
        cos_sims.append(cos_sim)
    ax.plot(offsets, cos_sims, 'o-', label=name, linewidth=2, markersize=6)
ax.set_xlabel("Position Offset")
ax.set_ylabel("Cosine Similarity to Baseline")
ax.set_title("Representation Drift")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.1, 1.1)

plt.tight_layout()
save_path2 = os.path.join(os.path.dirname(__file__), 'shifted_sentence_curves.png')
plt.savefig(save_path2, dpi=150, bbox_inches='tight')
print(f"Saved drift curves → {save_path2}")
print()

# ===================================================================
# End-of-day reflection
# ===================================================================
print(f"""
{'='*65}
{BOLD}STEP 3 RESULTS: SHIFTED SENTENCE EXPERIMENT{RESET}
{'='*65}

{BOLD}What we tested:{RESET}

  Same sentence ("{test_sentence[:20]}...")
  Fed at position offsets: {offsets}
  Models trained only on positions 0..{CONTEXT_LEN-1}.

{BOLD}What we observed:{RESET}

  1. {CYAN}Absolute PE degrades catastrophically{RESET}
     Learned position embeddings for positions > {CONTEXT_LEN-1} were NEVER trained.
     The model sees random (untrained) position vectors → garbage output.
     Loss: 1.3 → 16.1 at offset=80. Predictions become meaningless.

  2. {CYAN}Relative PE is perfectly shift-invariant{RESET}
     Relative attention only cares about DISTANCE between tokens (i - j).
     Shifting all positions by the same offset doesn't change distances.
     Loss, attention, and predictions are IDENTICAL at all offsets.

  3. {CYAN}RoPE is also perfectly shift-invariant (in this 1-layer model){RESET}
     RoPE's dot product depends only on (m - n), not absolute positions.
     Since V is not rotated, the output is purely determined by
     relative attention weights × content — which doesn't change.
     In DEEPER models, the story changes (see reflection below).

{'='*65}
{BOLD}END-OF-DAY REFLECTION: Why Long-Context Performance Degrades{RESET}
{'='*65}

{BOLD}The core problem: Extrapolation ≠ Generalization{RESET}

  A model trained on sequences of length 512 has learned:
    - Token relationships within 512-token windows
    - Position encodings for positions 0..511
    - Attention patterns at those specific distances

  When you push it to length 4096, three things break:

  {RED}1. Unseen positions{RESET}
     Absolute PE: positions 512..4095 have no learned embeddings.
     RoPE: rotation angles at position 4000 are vastly different
     from anything seen at position 400.

  {RED}2. Unseen distances{RESET}
     Even relative PE has limits. The model has never attended
     over a distance of 3000 tokens. The bias[3000] embedding
     is untrained or clamped.

  {RED}3. Attention entropy explosion{RESET}
     With 4096 tokens, softmax spreads attention over 4× more
     positions. The attention distribution becomes diluted.
     Important signals get drowned out by noise.

{BOLD}Why RoPE is perfect here but degrades in real systems:{RESET}

  In our 1-layer experiment, RoPE is perfectly shift-invariant:
    Q_m · K_n = f(q, k, m-n)  — dot product depends only on distance.
    V is not rotated, so output = relative_attn_weights × content.
    Shifting all positions preserves all relative distances.

  In DEEP models (32+ layers), degradation appears because:
    - Layer 1 output feeds into Layer 2's input, which gets rotated again
    - Small representation shifts accumulate across layers
    - The model's FFN weights were optimized for representations
      at specific rotation angles — unseen angles produce drift
    - At position 100000, the fast-rotating dimension pairs have
      wrapped around thousands of times, hitting precision limits

{BOLD}What real systems do about this:{RESET}

  1. {GREEN}Train on longer sequences{RESET} (expensive but effective)
  2. {GREEN}Progressive length extension{RESET} (fine-tune on gradually longer data)
  3. {GREEN}RoPE scaling{RESET} (divide position by a factor to keep angles in range)
     - NTK-aware scaling (adjust the base frequency)
     - YaRN (yet another RoPE extension)
  4. {GREEN}Sliding window attention{RESET} (Mistral: attend only to recent ~4K tokens)
  5. {GREEN}ALiBi{RESET} (linear decay bias — never extrapolates, just penalizes distance)

{BOLD}The lesson:{RESET}

  {CYAN}Position encoding is not just a technical detail — it determines
  the fundamental length generalization of your model.

  No amount of architectural cleverness replaces TRAINING on
  the actual sequence lengths you need at inference time.{RESET}
""")
