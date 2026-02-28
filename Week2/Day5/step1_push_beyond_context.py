"""
Step 1: Push Beyond Trained Context Length.

Our model is trained with CONTEXT_LEN=32.
What happens when we feed it longer sequences?

Three failure modes:
  1. Drift — predictions degrade gradually
  2. Attention collapse — weights become uniform (entropy maxes out)
  3. Looping — the model repeats itself

We build two models:
  - Absolute PE (nn.Embedding): cannot extrapolate at all (crashes beyond training length)
  - RoPE: can attempt extrapolation, but quality degrades

Key insight: "long context" doesn't mean "long memory."
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Week1', 'Day1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Week1', 'Day2'))

from train_tiny_char_level_lm import vocab_size, encode, decode, data, CONTEXT_LEN, idx_to_char, CORPUS
from step1_scaled_dot_product_attention import make_causal_mask

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

EMBED_DIM = 64
NUM_HEADS = 4
FF_DIM = 128
HEAD_DIM = EMBED_DIM // NUM_HEADS

print(f"""
{'='*65}
{BOLD}PUSH BEYOND TRAINED CONTEXT LENGTH{RESET}
{'='*65}

  Training context: {CONTEXT_LEN} tokens
  Corpus: {len(CORPUS)} characters of Shakespeare
  What happens at 48, 64, 96, 128 tokens?
""")


# ===================================================================
# Build models: Absolute PE vs RoPE
# ===================================================================

def apply_rope(x, start_pos=0):
    """Apply Rotary Position Embeddings."""
    B, H, C, D = x.shape
    positions = torch.arange(start_pos, start_pos + C, device=x.device).float()
    dim_pairs = D // 2
    freqs = 1.0 / (10000.0 ** (torch.arange(0, D, 2, device=x.device).float() / D))
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (C, D//2)

    cos_vals = torch.cos(angles)  # (C, D//2)
    sin_vals = torch.sin(angles)

    x1 = x[..., 0::2]  # (B, H, C, D//2)
    x2 = x[..., 1::2]

    rotated = torch.stack([
        x1 * cos_vals - x2 * sin_vals,
        x1 * sin_vals + x2 * cos_vals,
    ], dim=-1)  # (B, H, C, D//2, 2)

    return rotated.reshape(B, H, C, D)


class MultiHeadAttentionRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, mask=None):
        B, C, E = x.shape
        H, D = self.num_heads, self.head_dim

        Q = self.W_q(x).view(B, C, H, D).transpose(1, 2)
        K = self.W_k(x).view(B, C, H, D).transpose(1, 2)
        V = self.W_v(x).view(B, C, H, D).transpose(1, 2)

        Q = apply_rope(Q)
        K = apply_rope(K)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(D)
        if mask is not None:
            scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        attn_out = weights @ V

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, C, E)
        return self.W_o(attn_out), weights


class TransformerBlockRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionRoPE(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attn(self.ln1(x), mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, attn_weights


class TransformerLM_RoPE(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # No positional embedding — RoPE handles it
        self.blocks = nn.ModuleList([
            TransformerBlockRoPE(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, C = x.shape
        emb = self.embed(x)
        mask = make_causal_mask(C).to(x.device)
        h = emb
        for block in self.blocks:
            h, _ = block(h, mask)
        return self.out_proj(self.ln_final(h))

    def forward_with_intermediates(self, x):
        B, C = x.shape
        emb = self.embed(x)
        mask = make_causal_mask(C).to(x.device)
        h = emb
        attn_weights_all = []
        for block in self.blocks:
            h, attn_w = block(h, mask)
            attn_weights_all.append(attn_w.detach())
        return self.out_proj(self.ln_final(h)), attn_weights_all


class TransformerLM_AbsPE(nn.Module):
    """Absolute PE model that can extrapolate (with clamping or wrapping)."""
    def __init__(self, vocab_size, train_len, embed_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        self.train_len = train_len
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(train_len, embed_dim)

        from step1_scaled_dot_product_attention import scaled_dot_product_attention
        # Use regular multi-head attention
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.Module()
            block.ln1 = nn.LayerNorm(embed_dim)
            block.ln2 = nn.LayerNorm(embed_dim)

            block.attn_W_q = nn.Linear(embed_dim, embed_dim, bias=False)
            block.attn_W_k = nn.Linear(embed_dim, embed_dim, bias=False)
            block.attn_W_v = nn.Linear(embed_dim, embed_dim, bias=False)
            block.attn_W_o = nn.Linear(embed_dim, embed_dim, bias=False)

            block.ffn = nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.GELU(),
                nn.Linear(ff_dim, embed_dim),
            )
            self.blocks.append(block)

        self.ln_final = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, vocab_size)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

    def _run_block(self, h, block, mask):
        H, D = self.num_heads, self.head_dim
        h_norm = block.ln1(h)
        B, C, E = h_norm.shape

        Q = block.attn_W_q(h_norm).view(B, C, H, D).transpose(1, 2)
        K = block.attn_W_k(h_norm).view(B, C, H, D).transpose(1, 2)
        V = block.attn_W_v(h_norm).view(B, C, H, D).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(D)
        if mask is not None:
            scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        attn_out = weights @ V
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, C, E)
        attn_out = block.attn_W_o(attn_out)

        h = h + attn_out
        h = h + block.ffn(block.ln2(h))
        return h, weights

    def forward(self, x, extrapolate_mode='clamp'):
        B, C = x.shape
        emb = self.embed(x)

        # Handle positions beyond training length
        positions = torch.arange(C, device=x.device)
        if extrapolate_mode == 'clamp':
            positions = positions.clamp(max=self.train_len - 1)
        elif extrapolate_mode == 'wrap':
            positions = positions % self.train_len

        emb = emb + self.pos_embed(positions)
        mask = make_causal_mask(C).to(x.device)

        h = emb
        for block in self.blocks:
            h, _ = self._run_block(h, block, mask)
        return self.out_proj(self.ln_final(h))

    def forward_with_intermediates(self, x, extrapolate_mode='clamp'):
        B, C = x.shape
        emb = self.embed(x)

        positions = torch.arange(C, device=x.device)
        if extrapolate_mode == 'clamp':
            positions = positions.clamp(max=self.train_len - 1)
        elif extrapolate_mode == 'wrap':
            positions = positions % self.train_len

        emb = emb + self.pos_embed(positions)
        mask = make_causal_mask(C).to(x.device)

        h = emb
        attn_weights_all = []
        for block in self.blocks:
            h, attn_w = self._run_block(h, block, mask)
            attn_weights_all.append(attn_w.detach())
        return self.out_proj(self.ln_final(h)), attn_weights_all


# ===================================================================
# Train both models
# ===================================================================
TRAIN_CONTEXT = CONTEXT_LEN  # 32
NUM_LAYERS = 4

print(f"{BOLD}Training Absolute PE model (4 layers, 1200 steps, context={TRAIN_CONTEXT})...{RESET}")
torch.manual_seed(42)
model_abs = TransformerLM_AbsPE(vocab_size, TRAIN_CONTEXT, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS)
opt_abs = torch.optim.Adam(model_abs.parameters(), lr=3e-3)

for step in range(1, 1201):
    ix = torch.randint(0, len(data) - TRAIN_CONTEXT - 1, (128,))
    xb = torch.stack([data[i : i + TRAIN_CONTEXT] for i in ix])
    yb = torch.stack([data[i + 1 : i + TRAIN_CONTEXT + 1] for i in ix])
    loss = F.cross_entropy(model_abs(xb).view(-1, vocab_size), yb.view(-1))
    opt_abs.zero_grad(); loss.backward(); opt_abs.step()

print(f"  Final loss: {loss.item():.4f}")
model_abs.eval()

print(f"{BOLD}Training RoPE model (4 layers, 1200 steps, context={TRAIN_CONTEXT})...{RESET}")
torch.manual_seed(42)
model_rope = TransformerLM_RoPE(vocab_size, TRAIN_CONTEXT, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS)
opt_rope = torch.optim.Adam(model_rope.parameters(), lr=3e-3)

for step in range(1, 1201):
    ix = torch.randint(0, len(data) - TRAIN_CONTEXT - 1, (128,))
    xb = torch.stack([data[i : i + TRAIN_CONTEXT] for i in ix])
    yb = torch.stack([data[i + 1 : i + TRAIN_CONTEXT + 1] for i in ix])
    loss = F.cross_entropy(model_rope(xb).view(-1, vocab_size), yb.view(-1))
    opt_rope.zero_grad(); loss.backward(); opt_rope.step()

print(f"  Final loss: {loss.item():.4f}")
model_rope.eval()
print()


# ===================================================================
# 1. Feed progressively longer sequences
# ===================================================================
print(f"{BOLD}1. FEEDING LONGER SEQUENCES{RESET}")
print(f"{'='*65}")
print()

# Use the corpus directly — it's 640+ chars, plenty to test
# We'll feed windows of increasing size and measure per-position loss
test_lengths = [16, 32, 48, 64, 96, 128]

# Prepare long test sequences from the corpus
corpus_tokens = data.tolist()

print(f"  Measuring per-position cross-entropy loss as we extend beyond")
print(f"  the training context of {TRAIN_CONTEXT} tokens.")
print()

abs_losses_by_len = {}
rope_losses_by_len = {}

with torch.no_grad():
    for test_len in test_lengths:
        if test_len > len(corpus_tokens) - 1:
            continue

        # Take a chunk from the corpus
        chunk = corpus_tokens[:test_len]
        target = corpus_tokens[1:test_len + 1]

        inp = torch.tensor([chunk], dtype=torch.long)
        tgt = torch.tensor([target], dtype=torch.long)

        # Absolute PE (clamped extrapolation)
        logits_abs = model_abs(inp, extrapolate_mode='clamp')
        per_pos_loss_abs = F.cross_entropy(
            logits_abs[0], tgt[0], reduction='none'
        ).numpy()
        abs_losses_by_len[test_len] = per_pos_loss_abs

        # RoPE (natural extrapolation)
        logits_rope = model_rope(inp)
        per_pos_loss_rope = F.cross_entropy(
            logits_rope[0], tgt[0], reduction='none'
        ).numpy()
        rope_losses_by_len[test_len] = per_pos_loss_rope

        # Summary
        in_context = per_pos_loss_abs[:TRAIN_CONTEXT].mean()
        beyond_abs = per_pos_loss_abs[TRAIN_CONTEXT:].mean() if test_len > TRAIN_CONTEXT else float('nan')
        beyond_rope = per_pos_loss_rope[TRAIN_CONTEXT:].mean() if test_len > TRAIN_CONTEXT else float('nan')

        marker = f"  {GREEN}(within training context){RESET}" if test_len <= TRAIN_CONTEXT else ""
        print(f"  Length {test_len:>3}:{marker}")
        print(f"    Abs PE  — in-context avg: {per_pos_loss_abs[:min(test_len, TRAIN_CONTEXT)].mean():.3f}"
              + (f",  beyond: {beyond_abs:.3f}" if not np.isnan(beyond_abs) else ""))
        print(f"    RoPE    — in-context avg: {per_pos_loss_rope[:min(test_len, TRAIN_CONTEXT)].mean():.3f}"
              + (f",  beyond: {beyond_rope:.3f}" if not np.isnan(beyond_rope) else ""))
        print()

print(f"  {BOLD}What's happening:{RESET}")
print(f"    Within context ({TRAIN_CONTEXT} tokens): both models perform well.")
print(f"    Beyond context: loss increases — predictions become unreliable.")
print()


# ===================================================================
# 2. Generation beyond context — observe the three failure modes
# ===================================================================
print(f"{BOLD}2. GENERATION BEYOND CONTEXT — FAILURE MODES{RESET}")
print(f"{'='*65}")
print()

def generate_long(model, seed_tokens, length, model_type='rope'):
    """Generate tokens, allowing sequences longer than training context."""
    generated = list(seed_tokens)
    with torch.no_grad():
        for _ in range(length):
            # Use ALL generated tokens (or max allowed)
            inp = torch.tensor([generated], dtype=torch.long)
            if model_type == 'abs':
                logits = model(inp, extrapolate_mode='clamp')
            else:
                logits = model(inp)
            next_token = logits[0, -1].argmax().item()
            generated.append(next_token)
    return generated[len(seed_tokens):]


seed = "To be"
seed_tokens = encode(seed)
gen_length = 120  # well beyond 32-token training context

print(f"  Seed: \"{seed}\" ({len(seed_tokens)} tokens)")
print(f"  Generating {gen_length} tokens (training context was {TRAIN_CONTEXT})")
print()

with torch.no_grad():
    gen_abs = generate_long(model_abs, seed_tokens, gen_length, 'abs')
    gen_rope = generate_long(model_rope, seed_tokens, gen_length, 'rope')

abs_text = decode(gen_abs)
rope_text = decode(gen_rope)

# Show with position markers
print(f"  {BOLD}Absolute PE (clamped extrapolation):{RESET}")
for i in range(0, len(abs_text), 40):
    chunk = abs_text[i:i+40]
    pos_start = len(seed_tokens) + i
    pos_end = pos_start + len(chunk)
    color = GREEN if pos_end <= TRAIN_CONTEXT else (YELLOW if pos_start < TRAIN_CONTEXT else RED)
    zone = "in-ctx" if pos_end <= TRAIN_CONTEXT else ("boundary" if pos_start < TRAIN_CONTEXT else "BEYOND")
    print(f"    pos {pos_start:>3}-{pos_end:>3} [{zone:>8}]: {color}\"{chunk}\"{RESET}")
print()

print(f"  {BOLD}RoPE (natural extrapolation):{RESET}")
for i in range(0, len(rope_text), 40):
    chunk = rope_text[i:i+40]
    pos_start = len(seed_tokens) + i
    pos_end = pos_start + len(chunk)
    color = GREEN if pos_end <= TRAIN_CONTEXT else (YELLOW if pos_start < TRAIN_CONTEXT else RED)
    zone = "in-ctx" if pos_end <= TRAIN_CONTEXT else ("boundary" if pos_start < TRAIN_CONTEXT else "BEYOND")
    print(f"    pos {pos_start:>3}-{pos_end:>3} [{zone:>8}]: {color}\"{chunk}\"{RESET}")
print()

# Detect looping: check for repeated n-grams
def detect_loops(text, n=5):
    """Find repeated n-grams."""
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    from collections import Counter
    counts = Counter(ngrams)
    repeated = {ng: c for ng, c in counts.items() if c >= 3}
    return repeated

abs_loops = detect_loops(abs_text)
rope_loops = detect_loops(rope_text)

print(f"  {BOLD}Loop detection (5-grams appearing 3+ times):{RESET}")
print(f"    Abs PE:  {len(abs_loops)} looping patterns")
if abs_loops:
    for ng, count in sorted(abs_loops.items(), key=lambda x: -x[1])[:5]:
        print(f"      \"{ng}\" × {count}")
print(f"    RoPE:    {len(rope_loops)} looping patterns")
if rope_loops:
    for ng, count in sorted(rope_loops.items(), key=lambda x: -x[1])[:5]:
        print(f"      \"{ng}\" × {count}")
print()

# Check corpus match
def corpus_match_rate(text, corpus, n=5):
    """What fraction of generated n-grams appear in the corpus?"""
    gen_ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    matches = sum(1 for ng in gen_ngrams if ng in corpus)
    return matches / len(gen_ngrams) if gen_ngrams else 0

abs_match_in = corpus_match_rate(abs_text[:TRAIN_CONTEXT - len(seed_tokens)], CORPUS)
abs_match_beyond = corpus_match_rate(abs_text[TRAIN_CONTEXT - len(seed_tokens):], CORPUS)
rope_match_in = corpus_match_rate(rope_text[:TRAIN_CONTEXT - len(seed_tokens)], CORPUS)
rope_match_beyond = corpus_match_rate(rope_text[TRAIN_CONTEXT - len(seed_tokens):], CORPUS)

print(f"  {BOLD}Corpus match rate (5-gram):{RESET}")
print(f"    {'':>12} {'In-context':>12} {'Beyond':>12}")
print(f"    {'Abs PE':>12} {abs_match_in:>11.1%} {abs_match_beyond:>11.1%}")
print(f"    {'RoPE':>12} {rope_match_in:>11.1%} {rope_match_beyond:>11.1%}")
print()


# ===================================================================
# 3. Attention patterns at different positions
# ===================================================================
print(f"{BOLD}3. ATTENTION PATTERNS — IN-CONTEXT vs BEYOND{RESET}")
print(f"{'='*65}")
print()

# Feed a long sequence and look at attention weights
long_seq = corpus_tokens[:96]
inp = torch.tensor([long_seq], dtype=torch.long)

with torch.no_grad():
    _, attn_rope = model_rope.forward_with_intermediates(inp)
    _, attn_abs = model_abs.forward_with_intermediates(inp, extrapolate_mode='clamp')

# Compute attention entropy at each position
def attn_entropy(weights):
    """Compute entropy of attention distribution at each position."""
    # weights: (B, H, C, C), take mean over heads
    w = weights[0].mean(dim=0)  # (C, C)
    # For each query position, compute entropy of its attention distribution
    entropies = []
    for i in range(w.shape[0]):
        # Only look at positions 0..i (causal)
        p = w[i, :i+1]
        p = p / p.sum()  # renormalize
        p = p.clamp(min=1e-10)
        H = -(p * p.log()).sum().item()
        entropies.append(H)
    return entropies

# Layer 0 and last layer
for layer_idx, layer_name in [(0, "Layer 1 (first)"), (NUM_LAYERS-1, f"Layer {NUM_LAYERS} (last)")]:
    abs_ent = attn_entropy(attn_abs[layer_idx])
    rope_ent = attn_entropy(attn_rope[layer_idx])

    print(f"  {BOLD}{layer_name} — Attention entropy per position:{RESET}")
    print(f"    {'Position':>10} {'Abs PE':>10} {'RoPE':>10} {'Max possible':>14}")
    print(f"    {'-'*10} {'-'*10} {'-'*10} {'-'*14}")

    positions_to_show = [4, 8, 16, 24, 31, 40, 56, 72, 88, 95]
    for pos in positions_to_show:
        if pos >= len(abs_ent):
            continue
        max_ent = math.log(pos + 1)
        marker = "" if pos < TRAIN_CONTEXT else f"  {RED}← BEYOND{RESET}"
        print(f"    {pos:>10} {abs_ent[pos]:>10.3f} {rope_ent[pos]:>10.3f} {max_ent:>14.3f}{marker}")

    print()

print(f"  {BOLD}Interpretation:{RESET}")
print(f"  Max possible entropy = log(position+1) — uniform attention over all prior tokens.")
print(f"  If attention entropy approaches max → {RED}attention dilution{RESET}.")
print(f"  The model spreads probability equally — it can't decide what to attend to.")
print(f"  This is NOT useful attention; it's the model's way of saying \"I'm lost.\"")
print()


# ===================================================================
# 4. Per-position prediction quality
# ===================================================================
print(f"{BOLD}4. PREDICTION QUALITY vs POSITION{RESET}")
print(f"{'='*65}")
print()

# Feed the corpus and check per-position accuracy and loss
test_seq = corpus_tokens[:128]
test_tgt = corpus_tokens[1:129]

inp = torch.tensor([test_seq], dtype=torch.long)
tgt = torch.tensor(test_tgt, dtype=torch.long)

with torch.no_grad():
    logits_abs = model_abs(inp, extrapolate_mode='clamp')
    logits_rope = model_rope(inp)

    preds_abs = logits_abs[0].argmax(dim=-1).numpy()
    preds_rope = logits_rope[0].argmax(dim=-1).numpy()
    tgt_np = tgt.numpy()

# Compute running accuracy in windows
window = 8
positions = []
acc_abs_list = []
acc_rope_list = []

for start in range(0, len(test_seq) - window, 4):
    end = start + window
    acc_abs = (preds_abs[start:end] == tgt_np[start:end]).mean()
    acc_rope = (preds_rope[start:end] == tgt_np[start:end]).mean()
    positions.append((start + end) / 2)
    acc_abs_list.append(acc_abs)
    acc_rope_list.append(acc_rope)

print(f"  Prediction accuracy (8-token sliding window):")
print()
print(f"  {'Position':>10} {'Abs PE':>10} {'RoPE':>10} {'Zone'}")
print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

for pos, acc_a, acc_r in zip(positions, acc_abs_list, acc_rope_list):
    zone = "in-ctx" if pos < TRAIN_CONTEXT else "BEYOND"
    color = GREEN if zone == "in-ctx" else RED
    print(f"  {pos:>10.0f} {acc_a:>9.0%} {acc_r:>9.0%}  {color}{zone}{RESET}")

print()


# ===================================================================
# 5. Visualization
# ===================================================================
print(f"{BOLD}5. GENERATING VISUALIZATION...{RESET}")
print(f"{'='*65}")
print()

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(f"Context Length Extrapolation: Trained at {TRAIN_CONTEXT}, Tested Beyond",
             fontsize=14, fontweight='bold')

# --- Panel 1: Per-position loss curves ---
ax = axes[0, 0]
for test_len in sorted(rope_losses_by_len.keys()):
    if test_len >= 48:  # only show interesting lengths
        losses = rope_losses_by_len[test_len]
        ax.plot(range(len(losses)), losses, label=f'len={test_len}', alpha=0.8, linewidth=1.5)

ax.axvline(x=TRAIN_CONTEXT, color='red', linestyle='--', linewidth=2, label=f'Training ctx={TRAIN_CONTEXT}')
ax.set_xlabel("Position")
ax.set_ylabel("Cross-entropy loss")
ax.set_title("RoPE: Per-Position Loss")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, min(10, ax.get_ylim()[1]))

# --- Panel 2: Abs PE per-position loss ---
ax = axes[0, 1]
for test_len in sorted(abs_losses_by_len.keys()):
    if test_len >= 48:
        losses = abs_losses_by_len[test_len]
        ax.plot(range(len(losses)), losses, label=f'len={test_len}', alpha=0.8, linewidth=1.5)

ax.axvline(x=TRAIN_CONTEXT, color='red', linestyle='--', linewidth=2, label=f'Training ctx={TRAIN_CONTEXT}')
ax.set_xlabel("Position")
ax.set_ylabel("Cross-entropy loss")
ax.set_title("Abs PE (clamped): Per-Position Loss")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, min(10, ax.get_ylim()[1]))

# --- Panel 3: Accuracy vs position ---
ax = axes[0, 2]
ax.plot(positions, acc_abs_list, 'r-o', markersize=3, label='Abs PE', linewidth=1.5)
ax.plot(positions, acc_rope_list, 'g-o', markersize=3, label='RoPE', linewidth=1.5)
ax.axvline(x=TRAIN_CONTEXT, color='black', linestyle='--', linewidth=2, alpha=0.7)
ax.axhline(y=1/vocab_size, color='gray', linestyle=':', alpha=0.5, label='Random chance')
ax.fill_betweenx([0, 1], 0, TRAIN_CONTEXT, color='green', alpha=0.05)
ax.fill_betweenx([0, 1], TRAIN_CONTEXT, 128, color='red', alpha=0.05)
ax.set_xlabel("Position")
ax.set_ylabel("Accuracy (8-token window)")
ax.set_title("Prediction Accuracy vs Position")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# --- Panel 4: Attention entropy (last layer) ---
ax = axes[1, 0]
abs_ent_last = attn_entropy(attn_abs[-1])
rope_ent_last = attn_entropy(attn_rope[-1])
max_ent = [math.log(i + 1) for i in range(len(abs_ent_last))]

ax.plot(range(len(abs_ent_last)), abs_ent_last, 'r-', alpha=0.7, label='Abs PE', linewidth=1.5)
ax.plot(range(len(rope_ent_last)), rope_ent_last, 'g-', alpha=0.7, label='RoPE', linewidth=1.5)
ax.plot(range(len(max_ent)), max_ent, 'k--', alpha=0.3, label='Max entropy (uniform)')
ax.axvline(x=TRAIN_CONTEXT, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.set_xlabel("Position")
ax.set_ylabel("Attention entropy (nats)")
ax.set_title(f"Layer {NUM_LAYERS} Attention Entropy")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 5: Attention heatmap (RoPE, last layer, head 0) ---
ax = axes[1, 1]
attn_map = attn_rope[-1][0, 0].numpy()  # layer -1, batch 0, head 0
im = ax.imshow(attn_map, aspect='auto', cmap='viridis')
ax.axhline(y=TRAIN_CONTEXT - 0.5, color='red', linestyle='--', linewidth=2)
ax.axvline(x=TRAIN_CONTEXT - 0.5, color='red', linestyle='--', linewidth=2)
ax.set_xlabel("Key position")
ax.set_ylabel("Query position")
ax.set_title(f"RoPE Attention Map (Layer {NUM_LAYERS}, Head 1)")
plt.colorbar(im, ax=ax, fraction=0.046)

# --- Panel 6: Summary diagram ---
ax = axes[1, 2]
ax.axis('off')
ax.set_title("Three Failure Modes Beyond Context", fontsize=11)

modes = [
    ("1. DRIFT", "#e74c3c",
     "Predictions gradually degrade.\n"
     "Loss increases, accuracy drops.\n"
     "Model produces plausible but\n"
     "wrong continuations."),
    ("2. ATTENTION COLLAPSE", "#e67e22",
     "Softmax spreads mass uniformly.\n"
     "Entropy → max (uniform dist).\n"
     "Model can't decide what to\n"
     "attend to → random mixing."),
    ("3. LOOPING", "#9b59b6",
     "Model repeats same patterns.\n"
     "Gets stuck in attractor states.\n"
     "No new information enters —\n"
     "generation cycles endlessly."),
]

y = 0.92
for title, color, desc in modes:
    bbox = dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.15,
                edgecolor=color, linewidth=2)
    ax.text(0.05, y, title, fontsize=11, fontweight='bold',
            transform=ax.transAxes, va='top',
            bbox=bbox, color=color)
    ax.text(0.05, y - 0.07, desc, fontsize=8, transform=ax.transAxes,
            va='top', fontfamily='monospace', color='#333333')
    y -= 0.32

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), 'push_beyond_context.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"  Saved visualization → {save_path}")
print()


# ===================================================================
# Summary
# ===================================================================
print(f"""
{'='*65}
{BOLD}STEP 1 SUMMARY: PUSH BEYOND TRAINED CONTEXT{RESET}
{'='*65}

{BOLD}Setup:{RESET}
  Trained at context length {TRAIN_CONTEXT}, tested at {test_lengths}.
  Two models: Absolute PE (clamped) and RoPE.

{BOLD}Three failure modes beyond training context:{RESET}

  {RED}1. Drift{RESET}
     Per-position loss increases beyond position {TRAIN_CONTEXT}.
     The model's predictions become unreliable — not garbage,
     but increasingly wrong. Like a GPS losing signal.

  {RED}2. Attention collapse (dilution){RESET}
     As sequence grows, softmax spreads attention more thinly.
     Entropy approaches maximum → uniform distribution.
     The model attends to everything equally = attends to nothing.

  {RED}3. Looping{RESET}
     Without reliable position information, the model falls
     into attractor states and repeats patterns endlessly.
     No new information can break the cycle.

{BOLD}Abs PE vs RoPE:{RESET}
  Absolute PE: positions beyond training length are CLAMPED.
    Multiple tokens get the same position embedding → aliased.
    The model thinks pos 33 = pos 32 = pos 31 = ...

  RoPE: can extrapolate to unseen positions, but rotation
    frequencies at high positions create phase patterns
    never seen during training → degraded attention.

{BOLD}Why this matters (next step):{RESET}
  Step 2 will diagnose the two root causes:
    • Attention dilution (softmax scaling problem)
    • Positional aliasing (PE extrapolation failure)
""")
