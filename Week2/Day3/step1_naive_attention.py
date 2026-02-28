"""
Step 1: Implement Attention Naively — The O(n³) Problem.

Standard autoregressive decoding:
  For each new token, recompute attention over the ENTIRE sequence.

  Step 1: attend over 1 token    → 1² = 1 operations
  Step 2: attend over 2 tokens   → 2² = 4 operations
  Step 3: attend over 3 tokens   → 3² = 9 operations
  ...
  Step n: attend over n tokens   → n² operations

  Total: 1² + 2² + ... + n² ≈ n³/3 = O(n³)

This is the baseline. Step 2 will introduce KV-cache to fix this.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Week1', 'Day1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Week1', 'Day2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Week1', 'Day3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Week1', 'Day4'))

from train_tiny_char_level_lm import vocab_size, encode, decode, data, CONTEXT_LEN, idx_to_char
from step1_scaled_dot_product_attention import make_causal_mask
from step1_inspect_representations import (
    TransformerBlock, TransformerLM, EMBED_DIM, NUM_HEADS, FF_DIM
)

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

# ===================================================================
# Train model
# ===================================================================
print(f"""
{'='*65}
{BOLD}NAIVE ATTENTION: THE O(n³) PROBLEM{RESET}
{'='*65}

  Autoregressive generation: produce one token at a time.
  Each new token requires attending over ALL previous tokens.
  This means we recompute everything from scratch at each step.
""")

print(f"{BOLD}Training 4-layer model (1200 steps)...{RESET}")
torch.manual_seed(42)
model = TransformerLM(vocab_size, CONTEXT_LEN, EMBED_DIM, NUM_HEADS, FF_DIM, 4)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

for step in range(1, 1201):
    ix = torch.randint(0, len(data) - CONTEXT_LEN - 1, (128,))
    xb = torch.stack([data[i : i + CONTEXT_LEN] for i in ix])
    yb = torch.stack([data[i + 1 : i + CONTEXT_LEN + 1] for i in ix])
    loss = F.cross_entropy(model(xb).view(-1, vocab_size), yb.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"  Final loss: {loss.item():.4f}")
print()
model.eval()

# ===================================================================
# 1. What happens during naive generation
# ===================================================================
print(f"{BOLD}1. WHAT HAPPENS AT EACH GENERATION STEP{RESET}")
print(f"{'='*65}")
print()
print(f"  Seed: \"To be\" (5 tokens)")
print(f"  Generating 10 more tokens naively.")
print()

seed = "To be"
tokens = encode(seed)
generated = list(tokens)

print(f"  {'Step':<6} {'Seq len':<9} {'Input':>30} {'Attn ops':>10} {'→ Token'}")
print(f"  {'-'*6} {'-'*9} {'-'*30} {'-'*10} {'-'*8}")

total_attn_ops = 0

with torch.no_grad():
    for step in range(10):
        seq_len = len(generated)
        # THIS IS THE KEY: we feed the ENTIRE sequence every time
        inp = torch.tensor([generated], dtype=torch.long)
        logits = model(inp)
        next_token = logits[0, -1].argmax().item()

        # Attention cost: seq_len² per head per layer
        # (Q @ K^T is seq_len × seq_len)
        attn_ops = seq_len * seq_len * NUM_HEADS * 4  # 4 layers
        total_attn_ops += attn_ops

        input_str = decode(generated)
        if len(input_str) > 28:
            input_str = "..." + input_str[-25:]
        next_char = idx_to_char[next_token]

        print(f"  {step+1:<6} {seq_len:<9} {input_str:>30} {attn_ops:>10,} '{next_char}'")
        generated.append(next_token)

print(f"  {'':>6} {'':>9} {'Total:':>30} {total_attn_ops:>10,}")
print()

# ===================================================================
# 2. The redundancy problem
# ===================================================================
print(f"{BOLD}2. THE REDUNDANCY PROBLEM{RESET}")
print(f"{'='*65}")
print()
print(f"  Look at what gets recomputed at each step:")
print()

# Show the Q, K, V computations
print(f"  Step 1: tokens = [T, o, ' ', b, e]")
print(f"    Compute Q,K,V for: {RED}T, o, ' ', b, e{RESET}  ← 5 tokens")
print(f"    Attention:          5×5 matrix")
print(f"    We only NEED the last row (prediction for 'e')")
print()

print(f"  Step 2: tokens = [T, o, ' ', b, e, ,]")
print(f"    Compute Q,K,V for: {RED}T, o, ' ', b, e{RESET}, ,  ← 6 tokens")
print(f"    Attention:          6×6 matrix")
print(f"    {RED}First 5 Q,K,V are identical to Step 1!{RESET}")
print(f"    We recomputed them for nothing.")
print()

print(f"  Step 3: tokens = [T, o, ' ', b, e, ,, ' ']")
print(f"    Compute Q,K,V for: {RED}T, o, ' ', b, e, ,{RESET}, ' '  ← 7 tokens")
print(f"    Attention:          7×7 matrix")
print(f"    {RED}First 6 Q,K,V are identical to Step 2!{RESET}")
print()

print(f"  Pattern: at step t, we recompute Q,K,V for ALL t tokens.")
print(f"  But only the LAST token's Q,K,V are new.")
print(f"  The first (t-1) tokens' K and V haven't changed!")
print()
print(f"  {CYAN}This is the insight that leads to KV-cache (Step 2).{RESET}")
print()

# ===================================================================
# 3. Count the wasted computation
# ===================================================================
print(f"{BOLD}3. WASTED COMPUTATION{RESET}")
print(f"{'='*65}")
print()

seed_len = 5
gen_lengths = [10, 20, 32]

for gen_len in gen_lengths:
    total_ops = 0
    new_ops = 0

    for step in range(gen_len):
        seq_len = seed_len + step
        step_ops = seq_len * seq_len  # attention matrix size
        total_ops += step_ops

        # Only the new token's interactions are truly new
        # New Q (1 token) attending to all K (seq_len tokens) = seq_len ops
        new_ops += seq_len

    wasted = total_ops - new_ops
    waste_pct = 100 * wasted / total_ops

    print(f"  Generate {gen_len} tokens (from seed of {seed_len}):")
    print(f"    Naive total:    {total_ops:>10,} attention ops")
    print(f"    Actually needed: {new_ops:>10,} attention ops")
    print(f"    Wasted:          {wasted:>10,} ({waste_pct:.1f}%)")
    print()

print(f"  {RED}As sequence grows, waste approaches ~100%.{RESET}")
print(f"  At step t, we do t² work but only t is new → waste = (t²-t)/t² = 1 - 1/t")
print()

# ===================================================================
# 4. Measure actual wall-clock time
# ===================================================================
print(f"{BOLD}4. WALL-CLOCK TIMING{RESET}")
print(f"{'='*65}")
print()

def naive_generate(model, seed_tokens, length):
    """Naive generation: recompute everything at each step."""
    generated = list(seed_tokens)
    with torch.no_grad():
        for _ in range(length):
            inp = torch.tensor([generated[-CONTEXT_LEN:]], dtype=torch.long)
            logits = model(inp)
            next_token = logits[0, -1].argmax().item()
            generated.append(next_token)
    return generated[len(seed_tokens):]


seed_tokens = encode("To be")

# Warmup
_ = naive_generate(model, seed_tokens, 5)

gen_lengths_timing = [5, 10, 15, 20, 25, 30]
times = []
n_trials = 5

print(f"  {'Gen length':<12} {'Time (ms)':>10} {'ms/token':>10} {'Tokens/sec':>11}")
print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*11}")

for gen_len in gen_lengths_timing:
    trial_times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        _ = naive_generate(model, seed_tokens, gen_len)
        elapsed = time.perf_counter() - start
        trial_times.append(elapsed)

    avg_time = np.mean(trial_times) * 1000  # ms
    ms_per_token = avg_time / gen_len
    tokens_per_sec = gen_len / (avg_time / 1000)
    times.append(avg_time)

    print(f"  {gen_len:<12} {avg_time:>9.1f} {ms_per_token:>9.2f} {tokens_per_sec:>10.0f}")

print()

# Check if time grows super-linearly
if len(times) >= 2:
    ratio = times[-1] / times[0]
    len_ratio = gen_lengths_timing[-1] / gen_lengths_timing[0]
    scaling = math.log(ratio) / math.log(len_ratio) if len_ratio > 1 else 0
    print(f"  Time grew {ratio:.1f}× while length grew {len_ratio:.1f}×")
    print(f"  Empirical scaling: O(n^{scaling:.2f})")
    if scaling > 1.3:
        print(f"  {RED}Super-linear! This is the O(n²) per-step cost in action.{RESET}")
    else:
        print(f"  {YELLOW}Close to linear — our sequences are too short to see O(n²) clearly.{RESET}")
        print(f"  At sequence length 32, overhead dominates. The n² shows at length 1000+.")
print()

# ===================================================================
# 5. Theoretical scaling
# ===================================================================
print(f"{BOLD}5. THEORETICAL SCALING{RESET}")
print(f"{'='*65}")
print()

print(f"  {BOLD}Per-step cost:{RESET}")
print(f"    At step t, sequence length = t")
print(f"    Attention: Q(1×d) @ K(t×d)^T = O(t·d)  ... but we recompute ALL Q,K,V")
print(f"    Naive: recompute Q,K,V for all t tokens → O(t·d²)")
print(f"    Then: full t×t attention matrix → O(t²·d)")
print(f"    Per step: O(t²·d + t·d²)")
print()

print(f"  {BOLD}Total cost for n tokens:{RESET}")
print(f"    Σ(t=1..n) O(t²) = O(n³/3) ≈ O(n³)  [attention only]")
print(f"    Σ(t=1..n) O(t)  = O(n²/2) ≈ O(n²)  [Q,K,V projections]")
print(f"    Dominated by attention: {RED}O(n³){RESET}")
print()

# Show the growth concretely
print(f"  Concrete numbers (attention ops only, single head):")
print()
print(f"  {'n tokens':>10} {'Naive Σt²':>12} {'Optimal Σt':>12} {'Waste':>8}")
print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*8}")

for n in [10, 32, 100, 512, 1024, 4096]:
    naive = sum(t*t for t in range(1, n+1))
    optimal = sum(t for t in range(1, n+1))
    waste = naive - optimal
    print(f"  {n:>10,} {naive:>12,} {optimal:>12,} {waste/naive:>7.1%}")

print()
print(f"  At n=4096: naive does {sum(t*t for t in range(1, 4097)):,} ops")
print(f"  Optimal would do {sum(t for t in range(1, 4097)):,} ops")
print(f"  That's {sum(t*t for t in range(1, 4097)) / sum(t for t in range(1, 4097)):.0f}× more work than necessary!")
print()

# ===================================================================
# 6. Visualize what the attention matrix looks like during generation
# ===================================================================
print(f"{BOLD}6. ATTENTION MATRIX DURING GENERATION{RESET}")
print(f"{'='*65}")
print()

seed = "To be, or"
tokens = encode(seed)
generated = list(tokens)

# Collect attention weights at each generation step
step_attn_weights = []

with torch.no_grad():
    for step in range(8):
        inp = torch.tensor([generated[-CONTEXT_LEN:]], dtype=torch.long)
        _, _, attn_all = model.forward_with_intermediates(inp)
        # Layer 0, head 0 attention
        attn_w = attn_all[0][0, 0].numpy()  # (seq_len, seq_len)
        step_attn_weights.append(attn_w)

        # Get next token
        logits = model(inp)
        next_token = logits[0, -1].argmax().item()
        generated.append(next_token)

# Show what fraction of the matrix is actually used
print(f"  At each step, we compute a full seq_len × seq_len attention matrix.")
print(f"  But only the LAST ROW is used for the new token's prediction.")
print()

for step in range(min(5, len(step_attn_weights))):
    attn = step_attn_weights[step]
    seq_len = attn.shape[0]
    total_cells = seq_len * seq_len
    used_cells = seq_len  # only the last row
    # Causal mask means upper triangle is zero → actual computation is lower triangle
    causal_cells = seq_len * (seq_len + 1) // 2
    print(f"  Step {step+1}: {seq_len}×{seq_len} = {total_cells} cells computed, "
          f"causal = {causal_cells}, used = {used_cells} ({100*used_cells/causal_cells:.1f}%)")

print()
print(f"  {RED}We compute the full lower-triangular matrix but only read the last row.{RESET}")
print(f"  All other rows were already computed in previous steps!")
print()

# ===================================================================
# Visualization
# ===================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Naive Attention: Recomputing Everything at Each Step",
             fontsize=14, fontweight='bold')

# Top row: attention matrices at steps 1, 4, 8 — highlight what's new vs wasted
steps_to_show = [0, 3, 7]
for col, step_idx in enumerate(steps_to_show):
    ax = axes[0, col]
    attn = step_attn_weights[step_idx]
    seq_len = attn.shape[0]

    # Create a visualization showing wasted vs used
    display = np.zeros((seq_len, seq_len, 3))

    for i in range(seq_len):
        for j in range(i + 1):  # causal: j <= i
            if i == seq_len - 1:
                # Last row = the only row we actually NEED
                display[i, j] = [0.2, 0.7, 0.3]  # green = used
            else:
                # All other rows = wasted recomputation
                display[i, j] = [0.9, 0.3, 0.3]  # red = wasted

    ax.imshow(display, aspect='auto')
    ax.set_title(f"Step {step_idx+1}: {seq_len}×{seq_len}\n"
                 f"({seq_len} used / {seq_len*(seq_len+1)//2} computed)")
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")

    # Add text labels
    if seq_len <= 15:
        seq_text = decode(generated[:seq_len])
        for i in range(seq_len):
            ax.text(-0.7, i, f"'{seq_text[i]}'", fontsize=6, ha='right', va='center')

# Bottom left: scaling curves
ax = axes[1, 0]
n_range = np.arange(1, 201)
naive_total = np.cumsum(n_range**2)
optimal_total = np.cumsum(n_range)

ax.plot(n_range, naive_total, 'r-', linewidth=2.5, label='Naive: Σt² ≈ n³/3')
ax.plot(n_range, optimal_total, 'g-', linewidth=2.5, label='With KV-cache: Σt ≈ n²/2')
ax.set_xlabel("Sequence length n")
ax.set_ylabel("Total attention operations")
ax.set_title("Cumulative Cost: Naive vs Optimal")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Bottom middle: per-step cost
ax = axes[1, 1]
ax.plot(n_range, n_range**2, 'r-', linewidth=2.5, label='Naive: t² per step')
ax.plot(n_range, n_range, 'g-', linewidth=2.5, label='With KV-cache: t per step')
ax.set_xlabel("Step t (sequence length)")
ax.set_ylabel("Attention ops at step t")
ax.set_title("Per-Step Cost")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Bottom right: waste percentage
ax = axes[1, 2]
waste_pct = 1 - 1/n_range
ax.plot(n_range, 100 * waste_pct, 'r-', linewidth=2.5)
ax.axhline(y=95, color='gray', linestyle='--', alpha=0.5)
ax.text(100, 96, '95% waste', fontsize=9, color='gray')
ax.set_xlabel("Step t")
ax.set_ylabel("Wasted computation (%)")
ax.set_title("Waste: (t² - t) / t² = 1 - 1/t")
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), 'naive_attention.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved visualization → {save_path}")
print()

# ===================================================================
# Summary
# ===================================================================
print(f"""
{'='*65}
{BOLD}STEP 1 SUMMARY: NAIVE ATTENTION — THE O(n³) PROBLEM{RESET}
{'='*65}

{BOLD}The process:{RESET}

  Autoregressive decoding produces one token at a time.
  At step t, the sequence has t tokens.
  Naive approach: feed ALL t tokens, compute full attention.

  Step 1: [T, o]           → 2×2 attention  → predict next
  Step 2: [T, o, ' ']      → 3×3 attention  → predict next
  Step 3: [T, o, ' ', b]   → 4×4 attention  → predict next
  ...
  Step n: [all n tokens]    → n×n attention  → predict next

{BOLD}The cost:{RESET}

  Per step t:  O(t²)  attention operations
  Total for n: Σt² = O(n³/3) ≈ {RED}O(n³){RESET}

{BOLD}The waste:{RESET}

  At step t, we compute a t×t attention matrix.
  But only the LAST ROW matters (the new token attending to all others).
  The first (t-1) rows were already computed in previous steps!

  Waste per step: (t² - t) / t² = 1 - 1/t
  At step 100:  99% waste
  At step 1000: 99.9% waste

{BOLD}What's redundant:{RESET}

  {RED}1. Q,K,V projections for old tokens{RESET}
     Token "T" gets projected to Q_T, K_T, V_T at EVERY step.
     But K_T and V_T never change! Only Q matters for the new token.

  {RED}2. Attention scores for old query positions{RESET}
     Row i of the attention matrix (query i attending to keys 0..i)
     is identical at step t and step t+1.
     We recompute it anyway.

{BOLD}The fix (next step):{RESET}

  {GREEN}KV-Cache: store K and V from previous steps.{RESET}
  At step t, only compute Q,K,V for the NEW token.
  Attention for the new token: Q_new @ [K_0, K_1, ..., K_t]^T
  Cost per step: O(t) instead of O(t²)
  Total: O(n²) instead of O(n³)
""")
