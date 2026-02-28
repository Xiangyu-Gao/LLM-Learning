"""
Step 2: Diagnose — Attention Dilution & Positional Aliasing.

Step 1 showed WHAT happens when we exceed the training context.
Now we'll understand WHY.

Two root causes:

  1. Attention Dilution
     softmax over n tokens: each token gets ~1/n of the mass.
     At n=1000, even the "most relevant" token might get 0.1%.
     The model can't focus — information gets washed out.

  2. Positional Aliasing
     RoPE: rotation angle = position × frequency.
     At unseen positions, the rotation creates phase patterns
     the model has never learned to interpret.
     Absolute PE: positions beyond max_len don't exist.

Critical insight:
  Long context ≠ long memory.
  Transformers do content MIXING, not structured memory RETRIEVAL.
  This is why RAG exists.
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

from train_tiny_char_level_lm import vocab_size, encode, decode, data, CONTEXT_LEN, CORPUS
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
HEAD_DIM = EMBED_DIM // NUM_HEADS

print(f"""
{'='*65}
{BOLD}DIAGNOSE: ATTENTION DILUTION & POSITIONAL ALIASING{RESET}
{'='*65}
""")


# ===================================================================
# 1. ATTENTION DILUTION — the softmax problem
# ===================================================================
print(f"{BOLD}1. ATTENTION DILUTION — THE SOFTMAX PROBLEM{RESET}")
print(f"{'='*65}")
print()
print(f"  Softmax converts logits to probabilities: p_i = exp(s_i) / Σexp(s_j)")
print(f"  As sequence grows, the denominator grows → each p_i shrinks.")
print()

# Demonstrate with synthetic attention scores
print(f"  {BOLD}Experiment: one \"relevant\" key among n distractors{RESET}")
print()
print(f"  Setup: query has score 2.0 with one key, score 0.0 with all others.")
print(f"  How much attention does the relevant key get as n grows?")
print()

print(f"  {'n tokens':>10} {'P(relevant)':>13} {'P(each other)':>15} {'Entropy':>10} {'Max entropy':>13}")
print(f"  {'-'*10} {'-'*13} {'-'*15} {'-'*10} {'-'*13}")

relevant_probs = []
entropy_values = []
n_values = []

for n in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096]:
    # Attention scores: one relevant token at score 2.0, rest at 0.0
    scores = torch.zeros(n)
    scores[0] = 2.0
    probs = F.softmax(scores, dim=0)

    p_relevant = probs[0].item()
    p_other = probs[1].item() if n > 1 else 0
    entropy = -(probs * probs.log()).sum().item()
    max_entropy = math.log(n)

    relevant_probs.append(p_relevant)
    entropy_values.append(entropy)
    n_values.append(n)

    print(f"  {n:>10} {p_relevant:>12.4%} {p_other:>14.6f} {entropy:>9.3f} {max_entropy:>12.3f}")

print()
print(f"  {RED}At n=4096, the relevant token gets only 0.2% of attention!{RESET}")
print(f"  The \"signal\" is drowned in a sea of equally-weighted distractors.")
print()

# The scaling fix
print(f"  {BOLD}Why sqrt(d) scaling isn't enough:{RESET}")
print()
print(f"  The score scaling (÷ √d) prevents softmax saturation for a")
print(f"  FIXED n. But it doesn't help with the 1/n dilution problem.")
print(f"  As n grows, even well-separated scores get diluted.")
print()

# What if the relevant score is higher?
print(f"  {BOLD}What if the model learns sharper attention?{RESET}")
print(f"  (Relevant key gets score = k, others get 0)")
print()
print(f"  {'n tokens':>10} {'k=2':>8} {'k=5':>8} {'k=10':>8} {'k=20':>8}")
print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

for n in [4, 16, 64, 256, 1024, 4096]:
    row = f"  {n:>10}"
    for k in [2, 5, 10, 20]:
        scores = torch.zeros(n)
        scores[0] = float(k)
        probs = F.softmax(scores, dim=0)
        row += f" {probs[0].item():>7.1%}"
    print(row)

print()
print(f"  Even with score=20, at n=4096 the model retains 100% attention")
print(f"  — but this requires learning EXTREME logit magnitudes.")
print(f"  In practice, learned attention scores are moderate (typically 1-5).")
print(f"  The model can't reliably produce scores that overcome the 1/n dilution.")
print()


# ===================================================================
# 2. The information bottleneck
# ===================================================================
print(f"{BOLD}2. THE INFORMATION BOTTLENECK{RESET}")
print(f"{'='*65}")
print()
print(f"  At each position, attention produces a SINGLE vector of size d={EMBED_DIM}.")
print(f"  This vector must summarize ALL relevant information from the context.")
print()
print(f"  Context length    Bits per context token    Total info    Bottleneck")
print(f"  {'-'*15} {'-'*22} {'-'*13} {'-'*12}")

for n in [8, 32, 128, 512, 2048, 8192, 131072]:
    # Each attention output is d floats = d * 32 bits
    # But effective information is limited by entropy of attention weights
    # Max info from attention: min(d * 32, n * bits_per_token)
    bottleneck_bits = EMBED_DIM * 32  # output vector capacity
    info_per_token = EMBED_DIM * 32 / n  # spread across n tokens
    total_info = n * math.log2(vocab_size)  # raw information in context

    bottleneck = "OK" if info_per_token > 10 else ("tight" if info_per_token > 1 else "CRUSHED")
    color = GREEN if bottleneck == "OK" else (YELLOW if bottleneck == "tight" else RED)

    print(f"  {n:>15,} {info_per_token:>21.1f} {total_info:>12,.0f} {color}{bottleneck:>12}{RESET}")

print()
print(f"  The output vector has {EMBED_DIM * 32:,} bits of capacity.")
print(f"  At 128K context, each token's contribution is {EMBED_DIM * 32 / 131072:.3f} bits.")
print(f"  {RED}That's less than 1 bit per token — massive information loss.{RESET}")
print()
print(f"  Multi-head attention helps (H={NUM_HEADS} heads = {NUM_HEADS} independent summaries)")
print(f"  but doesn't change the fundamental O(d) bottleneck per layer.")
print()


# ===================================================================
# 3. POSITIONAL ALIASING — the RoPE problem
# ===================================================================
print(f"{BOLD}3. POSITIONAL ALIASING — THE RoPE PROBLEM{RESET}")
print(f"{'='*65}")
print()
print(f"  RoPE encodes position by rotating Q,K vectors.")
print(f"  Rotation angle = position × frequency.")
print(f"  Different dim pairs use different frequencies.")
print()

# Show the rotation angles at training vs extrapolation positions
print(f"  {BOLD}Rotation angles at different positions:{RESET}")
print()

frequencies = 1.0 / (10000.0 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
dim_pairs = HEAD_DIM // 2

print(f"  We have {dim_pairs} dimension pairs with frequencies:")
for pair_idx in range(dim_pairs):
    freq = frequencies[pair_idx].item()
    period = 2 * math.pi / freq if freq > 0 else float('inf')
    print(f"    Pair {pair_idx}: freq = {freq:.6f}, period = {period:.1f} positions")
print()

# Show angle values for training positions vs extrapolation
print(f"  {BOLD}Angles (radians) — Training vs Extrapolation:{RESET}")
print()
print(f"  {'Position':>10}", end="")
for pair_idx in range(dim_pairs):
    print(f" {'Pair '+str(pair_idx):>10}", end="")
print(f"  {'Zone'}")
print(f"  {'-'*10}", end="")
for _ in range(dim_pairs):
    print(f" {'-'*10}", end="")
print(f"  {'-'*10}")

positions_to_show = [0, 8, 16, 24, 31, 32, 48, 64, 96, 128, 256]
for pos in positions_to_show:
    zone = "train" if pos < CONTEXT_LEN else "EXTRAP"
    color = GREEN if zone == "train" else RED
    row = f"  {pos:>10}"
    for pair_idx in range(dim_pairs):
        angle = pos * frequencies[pair_idx].item()
        row += f" {angle:>10.3f}"
    print(f"{row}  {color}{zone}{RESET}")

print()
print(f"  {BOLD}The problem:{RESET}")
print(f"  Pair 0 (fastest): at position 128, angle = {128 * frequencies[0].item():.1f} radians")
print(f"    = {128 * frequencies[0].item() / (2*math.pi):.1f} full rotations")
print(f"    The model has never seen this many rotations during training!")
print()
print(f"  Pair {dim_pairs-1} (slowest): at position 128, angle = {128 * frequencies[-1].item():.4f} radians")
print(f"    = {128 * frequencies[-1].item() / (2*math.pi):.4f} full rotations")
print(f"    Barely moved — still within training range, but low-frequency")
print(f"    pairs can't distinguish positions 31 from 128.")
print()

# Show how the dot product between Q and K changes
print(f"  {BOLD}Dot product decay with distance (RoPE's relative position signal):{RESET}")
print()
print(f"  RoPE makes Q·K depend on the DISTANCE between positions.")
print(f"  For a fixed query at position p, how does Q_p · K_j change with |p-j|?")
print()

# Simulate with random Q, K vectors
torch.manual_seed(42)
q = torch.randn(HEAD_DIM)
k = torch.randn(HEAD_DIM)

def rotate_vector(v, pos, freqs):
    """Apply RoPE rotation to a vector."""
    v = v.clone()
    for pair_idx in range(len(freqs)):
        angle = pos * freqs[pair_idx].item()
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        i = pair_idx * 2
        v_i, v_i1 = v[i].item(), v[i+1].item()
        v[i] = v_i * cos_a - v_i1 * sin_a
        v[i+1] = v_i * sin_a + v_i1 * cos_a
    return v

# Query at position 16, keys at various distances
query_pos = 16
q_rotated = rotate_vector(q, query_pos, frequencies)

print(f"  Query at position {query_pos}:")
print(f"  {'Key position':>14} {'Distance':>10} {'Q·K':>10} {'Zone'}")
print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10}")

distances = [0, 1, 2, 4, 8, 15, 16, 20, 30, 40, 60, 100, 200]
dot_products = []
for key_pos in [query_pos - d for d in distances if query_pos - d >= 0] + \
               [query_pos + d for d in [20, 30, 40, 60, 100, 200]]:
    if key_pos < 0:
        continue
    k_rotated = rotate_vector(k, key_pos, frequencies)
    dot = torch.dot(q_rotated, k_rotated).item()
    dist = abs(query_pos - key_pos)
    zone = "train" if key_pos < CONTEXT_LEN else "EXTRAP"
    color = GREEN if zone == "train" else RED
    print(f"  {key_pos:>14} {dist:>10} {dot:>10.3f}  {color}{zone}{RESET}")
    dot_products.append((dist, dot))

print()
print(f"  {YELLOW}Note: dot products at extrapolated positions aren't inherently wrong,{RESET}")
print(f"  {YELLOW}but the model never learned to interpret these rotation patterns.{RESET}")
print()


# ===================================================================
# 4. Why long context ≠ long memory
# ===================================================================
print(f"{BOLD}4. WHY LONG CONTEXT ≠ LONG MEMORY{RESET}")
print(f"{'='*65}")
print()
print(f"  {BOLD}What transformers actually do:{RESET}")
print(f"    Content MIXING: each token's representation is a weighted average")
print(f"    of all tokens it attends to. This is fundamentally different from")
print(f"    structured memory retrieval (like a database lookup).")
print()
print(f"  {BOLD}The analogy:{RESET}")
print(f"    Imagine remembering a conversation by averaging all the words spoken.")
print(f"    With 10 words, the average is meaningful.")
print(f"    With 10,000 words, the average is noise.")
print()
print(f"    That's what attention does at long context — it AVERAGES,")
print(f"    and averages over many items lose information.")
print()
print(f"  {BOLD}Concrete example — needle in a haystack:{RESET}")
print()

# Simulate: can attention find a specific token in a long sequence?
# Place a "target" token at a specific position, surrounded by noise
target_score = 3.0  # the query matches this key well

print(f"  A query needs to find one specific key (score={target_score})")
print(f"  among n-1 distractors (score=0).")
print()
print(f"  {'Context len':>12} {'P(target)':>11} {'Effective info':>16} {'Status'}")
print(f"  {'-'*12} {'-'*11} {'-'*16} {'-'*10}")

for n in [8, 32, 128, 512, 2048, 8192, 32768, 131072]:
    scores = torch.zeros(n)
    scores[n // 2] = target_score  # needle in the middle
    probs = F.softmax(scores, dim=0)
    p_target = probs[n // 2].item()

    # Effective information: if we sample from this distribution,
    # how much does knowing the sample tell us?
    # Shannon: H = -Σ p log p
    entropy = -(probs * probs.clamp(min=1e-10).log()).sum().item()
    max_entropy = math.log(n)
    info_ratio = 1 - entropy / max_entropy  # 1 = perfect focus, 0 = uniform

    status = "FOUND" if p_target > 0.5 else ("weak" if p_target > 0.1 else "LOST")
    color = GREEN if status == "FOUND" else (YELLOW if status == "weak" else RED)
    print(f"  {n:>12,} {p_target:>10.2%} {info_ratio:>15.3f} {color}{status:>10}{RESET}")

print()
print(f"  {RED}At 128K context, a single needle with score 3.0 gets < 0.01% attention.{RESET}")
print(f"  The model would need score ≈ {math.log(131072):.0f} to reliably find it.")
print(f"  But learned attention scores rarely exceed 5-10.")
print()

print(f"  {BOLD}This is why RAG (Retrieval-Augmented Generation) exists:{RESET}")
print()
print(f"    Instead of:  stuff everything into context → hope attention finds it")
print(f"    RAG does:    search → retrieve relevant chunks → small focused context")
print()
print(f"    RAG converts the O(n) attention search into an O(log n) index lookup.")
print(f"    The retriever does the hard work, not the transformer's attention.")
print()
print(f"    Other solutions:")
print(f"    • {GREEN}Sliding window attention{RESET}: only attend to nearby tokens")
print(f"    • {GREEN}Sparse attention{RESET}: attend to selected positions")
print(f"    • {GREEN}Memory-augmented models{RESET}: external structured memory")
print(f"    • {GREEN}Chunked processing{RESET}: process in segments, summarize")
print()


# ===================================================================
# 5. Visualization
# ===================================================================
print(f"{BOLD}5. GENERATING VISUALIZATION...{RESET}")
print(f"{'='*65}")
print()

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Why Long Context Fails: Attention Dilution & Positional Aliasing",
             fontsize=14, fontweight='bold')

# --- Panel 1: Attention to relevant token vs context length ---
ax = axes[0, 0]
ax.plot(n_values, [p * 100 for p in relevant_probs], 'r-o', linewidth=2, markersize=5)
ax.set_xscale('log')
ax.set_xlabel("Context length (n)")
ax.set_ylabel("Attention to relevant token (%)")
ax.set_title("Attention Dilution\n(1 relevant key, score=2, rest=0)")
ax.grid(True, alpha=0.3)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax.text(100, 52, '50% threshold', fontsize=8, color='gray')

# --- Panel 2: Attention entropy vs max entropy ---
ax = axes[0, 1]
ax.plot(n_values, entropy_values, 'r-o', linewidth=2, markersize=5, label='Actual entropy')
ax.plot(n_values, [math.log(n) for n in n_values], 'k--', linewidth=1.5,
        alpha=0.5, label='Max entropy (uniform)')
ax.fill_between(n_values,
                entropy_values,
                [math.log(n) for n in n_values],
                alpha=0.1, color='green', label='Remaining signal')
ax.set_xscale('log')
ax.set_xlabel("Context length (n)")
ax.set_ylabel("Entropy (nats)")
ax.set_title("Entropy Approaches Maximum\n→ Attention becomes uniform")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 3: Sharper scores help, but not enough ---
ax = axes[0, 2]
n_range_log = np.logspace(0.5, 4, 50)
for k, color in [(2, '#e74c3c'), (5, '#e67e22'), (10, '#f1c40f'), (20, '#2ecc71')]:
    probs_k = []
    for n in n_range_log:
        n_int = int(n)
        scores = torch.zeros(n_int)
        scores[0] = float(k)
        p = F.softmax(scores, dim=0)[0].item()
        probs_k.append(p * 100)
    ax.plot(n_range_log, probs_k, color=color, linewidth=2, label=f'score={k}')

ax.set_xscale('log')
ax.set_xlabel("Context length (n)")
ax.set_ylabel("P(relevant) %")
ax.set_title("Higher Scores Fight Dilution\n(but need exponential growth)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3)

# --- Panel 4: RoPE rotation angles across positions ---
ax = axes[1, 0]
positions = np.arange(0, 129)
for pair_idx in range(dim_pairs):
    freq = frequencies[pair_idx].item()
    angles = positions * freq
    label = f'Pair {pair_idx} (f={freq:.4f})'
    ax.plot(positions, angles, linewidth=1.5, label=label)

ax.axvline(x=CONTEXT_LEN, color='red', linestyle='--', linewidth=2)
ax.text(CONTEXT_LEN + 1, ax.get_ylim()[1] * 0.9, 'Training\nboundary',
        fontsize=8, color='red')
ax.set_xlabel("Position")
ax.set_ylabel("Rotation angle (radians)")
ax.set_title("RoPE Rotation Angles\n(each dim pair rotates at different speed)")
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, alpha=0.3)

# --- Panel 5: Needle-in-haystack at different context lengths ---
ax = axes[1, 1]
context_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
for target_s in [2.0, 3.0, 5.0, 10.0]:
    needle_probs = []
    for n in context_sizes:
        scores = torch.zeros(n)
        scores[n // 2] = target_s
        p = F.softmax(scores, dim=0)[n // 2].item()
        needle_probs.append(p * 100)
    ax.plot(context_sizes, needle_probs, '-o', markersize=4, linewidth=1.5,
            label=f'needle score={target_s}')

ax.set_xscale('log')
ax.set_xlabel("Context length")
ax.set_ylabel("P(finding needle) %")
ax.set_title("Needle in a Haystack\n(finding 1 relevant token among n-1)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3)

# --- Panel 6: The solution landscape ---
ax = axes[1, 2]
ax.axis('off')
ax.set_title("Long Context ≠ Long Memory", fontsize=12)

sections = [
    ("THE PROBLEM", "#e74c3c",
     "Transformers do content MIXING\n"
     "(weighted averaging), not structured\n"
     "RETRIEVAL (database lookup).\n"
     "Averages over many items → noise."),
    ("WHY IT FAILS", "#e67e22",
     "1. Attention dilution: P ∝ 1/n\n"
     "2. Positional aliasing: PE breaks\n"
     "3. Info bottleneck: d-dim output\n"
     "   can't hold n items' worth of info"),
    ("THE SOLUTIONS", "#2ecc71",
     "• RAG: retrieve first, then attend\n"
     "• Sparse attention: attend selectively\n"
     "• Memory augmentation: external store\n"
     "• This is why RAG exists."),
]

y = 0.95
for title, color, desc in sections:
    bbox = dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.15,
                edgecolor=color, linewidth=2)
    ax.text(0.02, y, title, fontsize=11, fontweight='bold',
            transform=ax.transAxes, va='top', bbox=bbox, color=color)
    ax.text(0.02, y - 0.06, desc, fontsize=8.5, transform=ax.transAxes,
            va='top', fontfamily='monospace', color='#333333')
    y -= 0.35

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), 'diagnose_failures.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"  Saved visualization → {save_path}")
print()


# ===================================================================
# Summary
# ===================================================================
print(f"""
{'='*65}
{BOLD}STEP 2 SUMMARY: WHY LONG CONTEXT FAILS{RESET}
{'='*65}

{BOLD}Issue 1 — Attention Dilution:{RESET}

  softmax(scores) distributes probability mass across ALL tokens.
  As context grows, each token's share shrinks:

    n=32:    relevant token gets ~20% attention  → {GREEN}usable{RESET}
    n=1024:  relevant token gets ~0.7% attention → {YELLOW}borderline{RESET}
    n=131072: relevant token gets ~0.002%        → {RED}lost{RESET}

  The model would need exponentially higher scores to compensate.
  But learned attention scores are bounded by the weight magnitudes
  learned during training → the model can't adapt.

{BOLD}Issue 2 — Positional Aliasing:{RESET}

  Absolute PE: positions beyond max_len simply don't exist.
    Clamping → multiple tokens share the same position → confusion.
    Wrapping → position 33 = position 1 → confusion.

  RoPE: rotation angles at unseen positions create novel patterns.
    Fast-rotating pairs: phase wraps many times → aliasing.
    Slow-rotating pairs: barely distinguishes positions.
    Neither regime provides reliable position information.

{BOLD}The deep insight:{RESET}

  {CYAN}Long context ≠ long memory.{RESET}

  Transformers compute attention = weighted average of values.
  This is {RED}content mixing{RESET}, not {GREEN}structured retrieval{RESET}.

  A weighted average over 128K tokens produces a blurry summary,
  not a precise lookup. It's like searching for a face in a crowd
  by averaging all the faces — you get a generic face, not the one
  you're looking for.

  This is fundamentally different from:
    • A database: SELECT value WHERE key = target → exact match
    • A hash table: O(1) lookup by key → exact match
    • RAG: embed query → nearest neighbor search → exact documents

{BOLD}Why RAG exists:{RESET}

  RAG replaces the O(n) soft attention search with:
    1. {GREEN}Embed the query{RESET} → dense vector
    2. {GREEN}Search an index{RESET} → O(log n) or O(1) lookup
    3. {GREEN}Retrieve top-k documents{RESET} → small, focused context
    4. {GREEN}Feed to transformer{RESET} → attend over k << n tokens

  The retriever does the hard part (finding relevant info).
  The transformer does what it's good at (reasoning over small context).

  This is why "just make context longer" doesn't solve everything.
  128K context helps with some tasks, but for reliable retrieval
  of specific facts, you need structured memory — not more averaging.
""")
