"""
Step 2: Add KV-Cache — From O(n³) to O(n²).

The fix for Step 1's redundancy problem:
  Instead of recomputing K,V for ALL tokens at every step,
  STORE them and only compute K,V for the NEW token.

  Step 1: Compute Q₁,K₁,V₁ for token 1.  Cache: K=[K₁], V=[V₁]
  Step 2: Compute Q₂,K₂,V₂ for token 2.  Cache: K=[K₁,K₂], V=[V₁,V₂]
          Attend: Q₂ @ [K₁,K₂]^T = 2 ops (not 2² = 4)
  Step 3: Compute Q₃,K₃,V₃ for token 3.  Cache: K=[K₁,K₂,K₃], V=[V₁,V₂,V₃]
          Attend: Q₃ @ [K₁,K₂,K₃]^T = 3 ops (not 3² = 9)
  ...
  Step n: Attend Q_n @ [K₁..K_n]^T = n ops (not n² ops)

  Total: 1+2+3+...+n = n(n+1)/2 = O(n²)  (was O(n³))

Measurements:
  1. Wall-clock time: naive vs cached
  2. FLOPs estimate: theoretical savings
  3. Memory cost: the price we pay for speed
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
from step1_scaled_dot_product_attention import scaled_dot_product_attention, make_causal_mask
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
# Train model (same as step 1)
# ===================================================================
print(f"""
{'='*65}
{BOLD}KV-CACHE: FROM O(n³) TO O(n²){RESET}
{'='*65}

  Step 1 showed the problem: naive generation recomputes K,V for
  ALL tokens at every step → O(n³) total.

  The fix: cache K and V. At each step, only compute the new token's
  Q, K, V. Concatenate new K,V onto the cache. Attend Q_new to all
  cached K's. Cost per step: O(n) instead of O(n²).
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
# 1. Build KV-cached generation, step by step
# ===================================================================
print(f"{BOLD}1. KV-CACHE MECHANISM — STEP BY STEP{RESET}")
print(f"{'='*65}")
print()

# We need to manually extract weights from each layer's attention
# to demonstrate KV-cache. Let's build it from scratch.

def extract_attention_weights(model):
    """Extract Q,K,V,O projection matrices from each layer."""
    layers = []
    for block in model.blocks:
        attn = block.attn
        layers.append({
            'W_q': attn.W_q.weight.data,  # (E, E) — Linear stores as (out, in)
            'W_k': attn.W_k.weight.data,
            'W_v': attn.W_v.weight.data,
            'W_o': attn.W_o.weight.data,
            'ln1_weight': block.ln1.weight.data,
            'ln1_bias': block.ln1.bias.data,
            'ln2_weight': block.ln2.weight.data,
            'ln2_bias': block.ln2.bias.data,
            'ffn_0_weight': block.ffn[0].weight.data,
            'ffn_0_bias': block.ffn[0].bias.data,
            'ffn_2_weight': block.ffn[2].weight.data,
            'ffn_2_bias': block.ffn[2].bias.data,
        })
    return layers


def layer_norm(x, weight, bias, eps=1e-5):
    """Manual layer norm."""
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * weight + bias


def naive_generate(model, seed_tokens, length):
    """Naive: feed entire sequence at every step."""
    generated = list(seed_tokens)
    with torch.no_grad():
        for _ in range(length):
            inp = torch.tensor([generated[-CONTEXT_LEN:]], dtype=torch.long)
            logits = model(inp)
            next_token = logits[0, -1].argmax().item()
            generated.append(next_token)
    return generated[len(seed_tokens):]


def kv_cached_generate(model, seed_tokens, length):
    """
    KV-cached generation: only process the new token at each step.

    For the SEED (prefill): process all tokens at once (like normal forward).
    For GENERATION: process one token at a time, using cached K,V.
    """
    H = NUM_HEADS
    D = EMBED_DIM // H
    num_layers = len(model.blocks)

    # --- Phase 1: Prefill — process all seed tokens at once ---
    seed_tensor = torch.tensor([seed_tokens], dtype=torch.long)
    B, C = seed_tensor.shape

    emb = model.embed(seed_tensor) + model.pos_embed(torch.arange(C))
    mask = make_causal_mask(C)

    # Run through each layer, saving K and V
    kv_cache = []  # list of (K_cache, V_cache) per layer
    h = emb

    for block in model.blocks:
        # Pre-norm
        h_norm = layer_norm(h, block.ln1.weight.data, block.ln1.bias.data)

        # Compute Q, K, V for all positions
        Q = block.attn.W_q(h_norm)  # (1, C, E)
        K = block.attn.W_k(h_norm)
        V = block.attn.W_v(h_norm)

        # Reshape: (1, C, E) → (1, H, C, D)
        Q = Q.view(1, C, H, D).transpose(1, 2)
        K = K.view(1, C, H, D).transpose(1, 2)
        V = V.view(1, C, H, D).transpose(1, 2)

        # Save K, V to cache
        kv_cache.append((K.clone(), V.clone()))

        # Attention
        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(1, C, EMBED_DIM)
        attn_out = block.attn.W_o(attn_out)

        # Residual
        h = h + attn_out

        # FFN (pre-norm)
        h_norm2 = layer_norm(h, block.ln2.weight.data, block.ln2.bias.data)
        h = h + block.ffn(h_norm2)

    # Get first prediction
    h_final = layer_norm(h, model.ln_final.weight.data, model.ln_final.bias.data)
    logits = model.out_proj(h_final)
    next_token = logits[0, -1].argmax().item()

    generated = [next_token]

    # --- Phase 2: Decode — one token at a time with KV-cache ---
    for step in range(length - 1):
        pos = C + step  # position of the new token
        if pos >= CONTEXT_LEN:
            break

        # Embed ONLY the new token
        new_token = torch.tensor([[next_token]], dtype=torch.long)
        new_emb = model.embed(new_token) + model.pos_embed(torch.tensor([pos]))
        # new_emb: (1, 1, E)

        h = new_emb

        for layer_idx, block in enumerate(model.blocks):
            # Pre-norm
            h_norm = layer_norm(h, block.ln1.weight.data, block.ln1.bias.data)

            # Compute Q, K, V for NEW token only — this is the key savings
            q_new = block.attn.W_q(h_norm)  # (1, 1, E)
            k_new = block.attn.W_k(h_norm)
            v_new = block.attn.W_v(h_norm)

            # Reshape: (1, 1, E) → (1, H, 1, D)
            q_new = q_new.view(1, 1, H, D).transpose(1, 2)
            k_new = k_new.view(1, 1, H, D).transpose(1, 2)
            v_new = v_new.view(1, 1, H, D).transpose(1, 2)

            # Append new K, V to cache
            K_cached, V_cached = kv_cache[layer_idx]
            K_cached = torch.cat([K_cached, k_new], dim=2)  # (1, H, t+1, D)
            V_cached = torch.cat([V_cached, v_new], dim=2)
            kv_cache[layer_idx] = (K_cached, V_cached)

            # Attend: q_new (1,H,1,D) @ K_cached^T (1,H,D,t+1) → (1,H,1,t+1)
            # No mask needed: the new token can attend to ALL cached positions
            scores = q_new @ K_cached.transpose(-2, -1) / math.sqrt(D)
            weights = F.softmax(scores, dim=-1)
            attn_out = weights @ V_cached  # (1, H, 1, D)

            attn_out = attn_out.transpose(1, 2).contiguous().view(1, 1, EMBED_DIM)
            attn_out = block.attn.W_o(attn_out)

            # Residual
            h = h + attn_out

            # FFN
            h_norm2 = layer_norm(h, block.ln2.weight.data, block.ln2.bias.data)
            h = h + block.ffn(h_norm2)

        # Predict next token
        h_final = layer_norm(h, model.ln_final.weight.data, model.ln_final.bias.data)
        logits = model.out_proj(h_final)
        next_token = logits[0, -1].argmax().item()
        generated.append(next_token)

    return generated


# First: verify correctness — naive and cached must produce identical output
seed = "To be"
seed_tokens = encode(seed)

with torch.no_grad():
    naive_output = naive_generate(model, seed_tokens, 20)
    cached_output = kv_cached_generate(model, seed_tokens, 20)

naive_text = decode(naive_output)
cached_text = decode(cached_output)

print(f"  {BOLD}Correctness check:{RESET}")
print(f"    Seed: \"{seed}\"")
print(f"    Naive:  \"{naive_text}\"")
print(f"    Cached: \"{cached_text}\"")
match = naive_text == cached_text
print(f"    Match: {GREEN + 'YES ✓' + RESET if match else RED + 'NO ✗' + RESET}")
if not match:
    # Find first mismatch
    for i, (a, b) in enumerate(zip(naive_text, cached_text)):
        if a != b:
            print(f"    First mismatch at position {i}: naive='{a}' cached='{b}'")
            break
print()

# Show what the cache does at each step
print(f"  {BOLD}What happens at each step:{RESET}")
print()
print(f"  {'Step':<6} {'Action':<45} {'Cache size':>12} {'Attn ops':>10}")
print(f"  {'-'*6} {'-'*45} {'-'*12} {'-'*10}")

seed_len = len(seed_tokens)
total_cached_ops = 0
total_naive_ops = 0

# Prefill
prefill_ops = seed_len * seed_len * NUM_HEADS * 4  # full attention for seed
total_cached_ops += prefill_ops
total_naive_ops += prefill_ops
print(f"  {'pre':<6} {'Prefill: process all seed tokens at once':<45} "
      f"{'K,V: '+str(seed_len)+'×'+str(EMBED_DIM):>12} {prefill_ops:>10,}")

# Each generation step
for step in range(15):
    seq_len = seed_len + step + 1  # total sequence length

    # KV-cached: new token attends to all cached K's
    cached_ops = seq_len * NUM_HEADS * 4  # Q_new (1) @ K_all (seq_len), per head per layer
    total_cached_ops += cached_ops

    # Naive: recompute full attention matrix
    naive_ops = seq_len * seq_len * NUM_HEADS * 4
    total_naive_ops += naive_ops

    action = f"New token → Q_new @ K_cache[:{seq_len}]"
    cache_str = f"K,V: {seq_len}×{EMBED_DIM}"
    print(f"  {step+1:<6} {action:<45} {cache_str:>12} {cached_ops:>10,}")

print(f"  {'':>6} {'':>45} {'Total:':>12} {total_cached_ops:>10,}")
print()
print(f"  Naive total for same generation: {total_naive_ops:>10,}")
print(f"  Speedup: {total_naive_ops / total_cached_ops:.1f}×")
print()


# ===================================================================
# 2. Wall-clock comparison
# ===================================================================
print(f"{BOLD}2. WALL-CLOCK COMPARISON{RESET}")
print(f"{'='*65}")
print()

gen_lengths = [5, 10, 15, 20, 25]
n_trials = 10

# Warmup
with torch.no_grad():
    _ = naive_generate(model, seed_tokens, 5)
    _ = kv_cached_generate(model, seed_tokens, 5)

naive_times = []
cached_times = []

print(f"  {'Gen len':<9} {'Naive (ms)':>11} {'Cached (ms)':>12} {'Speedup':>9} {'Same output?':>13}")
print(f"  {'-'*9} {'-'*11} {'-'*12} {'-'*9} {'-'*13}")

for gen_len in gen_lengths:
    # Time naive
    t_naive = []
    for _ in range(n_trials):
        with torch.no_grad():
            start = time.perf_counter()
            naive_out = naive_generate(model, seed_tokens, gen_len)
            elapsed = time.perf_counter() - start
            t_naive.append(elapsed)

    # Time cached
    t_cached = []
    for _ in range(n_trials):
        with torch.no_grad():
            start = time.perf_counter()
            cached_out = kv_cached_generate(model, seed_tokens, gen_len)
            elapsed = time.perf_counter() - start
            t_cached.append(elapsed)

    avg_naive = np.mean(t_naive) * 1000
    avg_cached = np.mean(t_cached) * 1000
    speedup = avg_naive / avg_cached if avg_cached > 0 else float('inf')
    same = decode(naive_out) == decode(cached_out)

    naive_times.append(avg_naive)
    cached_times.append(avg_cached)

    print(f"  {gen_len:<9} {avg_naive:>10.1f} {avg_cached:>11.1f} {speedup:>8.2f}× "
          f"{'  ' + GREEN + 'YES' + RESET if same else '  ' + RED + 'NO' + RESET}")

print()

# Why KV-cache may not show speedup on tiny models
if all(n/c < 1.5 for n, c in zip(naive_times, cached_times)):
    print(f"  {YELLOW}Note: KV-cache may be SLOWER on our tiny model!{RESET}")
    print(f"  Our sequences are short (≤32) and model is small (64-dim).")
    print(f"  The overhead of Python-level cache management (torch.cat, manual")
    print(f"  LayerNorm, separate forward calls) dominates at this scale.")
    print(f"  The real win shows at seq_len > 1000 with GPU-optimized code.")
elif all(n/c >= 1.5 for n, c in zip(naive_times, cached_times)):
    print(f"  {GREEN}KV-cache is faster even on our tiny model!{RESET}")
else:
    print(f"  {YELLOW}Mixed results — overhead and savings are similar at this scale.{RESET}")
print()


# ===================================================================
# 3. FLOPs estimate
# ===================================================================
print(f"{BOLD}3. FLOPs ESTIMATE{RESET}")
print(f"{'='*65}")
print()

print(f"  {BOLD}Per-step breakdown (generating token t, 4 layers, 4 heads):{RESET}")
print()
print(f"  Component              Naive                  KV-cached")
print(f"  {'─'*22} {'─'*22} {'─'*22}")

d = EMBED_DIM       # 64
h = NUM_HEADS        # 4
head_d = d // h      # 16
ff = FF_DIM          # 128
L = 4                # layers

# For a single step at position t:
# Q,K,V projection: t tokens × d × d × 3 (naive) vs 1 × d × d × 3 (cached)
# Attention: t × t × head_d × h (naive) vs 1 × t × head_d × h (cached)
# O projection: t × d × d (naive) vs 1 × d × d (cached)
# FFN: t × d × ff × 2 (naive) vs 1 × d × ff × 2 (cached)

print(f"  Q,K,V projection       t × 3 × {d}²             1 × 3 × {d}²")
print(f"                         = {3*d*d}t                 = {3*d*d}")
print()
print(f"  Attention (Q @ K^T)    t × t × {head_d} × {h}    1 × t × {head_d} × {h}")
print(f"                         = {head_d*h}t²             = {head_d*h}t")
print()
print(f"  Attn @ V               t × t × {head_d} × {h}    1 × t × {head_d} × {h}")
print(f"                         = {head_d*h}t²             = {head_d*h}t")
print()
print(f"  O projection           t × {d}²                  1 × {d}²")
print(f"                         = {d*d}t                   = {d*d}")
print()
print(f"  FFN (2 layers)         t × 2 × {d} × {ff}        1 × 2 × {d} × {ff}")
print(f"                         = {2*d*ff}t                = {2*d*ff}")
print()
print(f"  Per step, per layer:   ~{3*d*d + 2*head_d*h + d*d + 2*d*ff}t + {2*head_d*h}t²")
print(f"  (naive)                ~({3*d*d + d*d + 2*d*ff})t + {2*head_d*h}t²")
print(f"  (cached)               ~{3*d*d + d*d + 2*d*ff + 2*head_d*h}t + {3*d*d + d*d + 2*d*ff}")
print()

# Total FLOPs for generating n tokens
print(f"  {BOLD}Total FLOPs to generate n tokens (attention-dominated):{RESET}")
print()
print(f"  {'n tokens':>10} {'Naive FLOPs':>14} {'Cached FLOPs':>14} {'Savings':>9}")
print(f"  {'-'*10} {'-'*14} {'-'*14} {'-'*9}")

for n in [10, 32, 100, 512, 1024, 4096]:
    # Attention FLOPs (dominating term): 2 * d * t² (naive) vs 2 * d * t (cached)
    # Summed over t=1..n, times L layers
    naive_flops = L * 2 * d * sum(t*t for t in range(1, n+1))
    cached_flops_attn = L * 2 * d * sum(t for t in range(1, n+1))
    # Projection FLOPs: same for cached per new token
    proj_per_token = L * (3 * d * d + d * d + 2 * d * ff)
    cached_flops = cached_flops_attn + n * proj_per_token
    # Naive also has projection but for all t tokens each step
    naive_flops += L * proj_per_token * sum(range(1, n+1))

    savings = 1 - cached_flops / naive_flops

    if naive_flops > 1e9:
        naive_str = f"{naive_flops/1e9:.1f}G"
        cached_str = f"{cached_flops/1e9:.1f}G"
    elif naive_flops > 1e6:
        naive_str = f"{naive_flops/1e6:.1f}M"
        cached_str = f"{cached_flops/1e6:.1f}M"
    else:
        naive_str = f"{naive_flops/1e3:.0f}K"
        cached_str = f"{cached_flops/1e3:.0f}K"

    print(f"  {n:>10,} {naive_str:>14} {cached_str:>14} {savings:>8.1%}")

print()


# ===================================================================
# 4. Memory cost — the tradeoff
# ===================================================================
print(f"{BOLD}4. MEMORY COST — THE TRADEOFF{RESET}")
print(f"{'='*65}")
print()
print(f"  KV-cache trades memory for speed. How much memory?")
print()
print(f"  Per layer: K cache = (seq_len × {EMBED_DIM}) floats")
print(f"             V cache = (seq_len × {EMBED_DIM}) floats")
print(f"             Total   = 2 × seq_len × {EMBED_DIM} × 4 bytes (float32)")
print()
print(f"  All {L} layers: 2 × {L} × seq_len × {EMBED_DIM} × 4 bytes")
print()

print(f"  {'Seq length':>12} {'KV cache (ours)':>16} {'KV cache (GPT-2)':>18} {'KV cache (LLaMA-70B)':>22}")
print(f"  {'-'*12} {'-'*16} {'-'*18} {'-'*22}")

# Our model: d=64, L=4
# GPT-2 Small: d=768, L=12
# LLaMA-70B: d=8192, L=80 (GQA reduces this, but let's show the full version)
for n in [32, 128, 512, 2048, 8192, 32768, 131072]:
    ours = 2 * L * n * EMBED_DIM * 4
    gpt2 = 2 * 12 * n * 768 * 4
    llama = 2 * 80 * n * 8192 * 4  # without GQA

    def fmt_bytes(b):
        if b >= 1e9:
            return f"{b/1e9:.1f} GB"
        elif b >= 1e6:
            return f"{b/1e6:.1f} MB"
        elif b >= 1e3:
            return f"{b/1e3:.1f} KB"
        return f"{b} B"

    print(f"  {n:>12,} {fmt_bytes(ours):>16} {fmt_bytes(gpt2):>18} {fmt_bytes(llama):>22}")

print()
print(f"  {RED}At 128K context, LLaMA-70B KV-cache = 160 GB — more than the model itself!{RESET}")
print(f"  This is why techniques like GQA, MQA, and KV-cache quantization matter.")
print()
print(f"  {BOLD}The memory-speed tradeoff:{RESET}")
print(f"    Naive:  O(1) extra memory,  O(n³) compute")
print(f"    Cached: O(n·d·L) memory,    O(n²) compute")
print(f"    We pay O(n·d·L) memory to save O(n) compute per step.")
print()


# ===================================================================
# 5. Visualize the KV-cache mechanism
# ===================================================================
print(f"{BOLD}5. GENERATING VISUALIZATION...{RESET}")
print(f"{'='*65}")
print()

fig = plt.figure(figsize=(20, 14))
fig.suptitle("KV-Cache: From O(n³) to O(n²)", fontsize=15, fontweight='bold')

# --- Panel 1: Naive vs Cached — what gets computed ---
# Show 4 generation steps, comparing naive (full matrix) vs cached (single row)
ax1 = fig.add_subplot(2, 3, 1)

# Create a visual comparing naive vs cached at step 8
step_size = 12  # show a 12×12 matrix
naive_grid = np.zeros((step_size, step_size, 3))
cached_grid = np.zeros((step_size, step_size, 3))

for i in range(step_size):
    for j in range(i + 1):
        if i == step_size - 1:
            naive_grid[i, j] = [0.2, 0.7, 0.3]   # green = used
            cached_grid[i, j] = [0.2, 0.7, 0.3]   # green = computed & used
        else:
            naive_grid[i, j] = [0.9, 0.3, 0.3]    # red = wasted
            cached_grid[i, j] = [0.85, 0.85, 0.85] # gray = from cache (not recomputed)

# Draw naive on the left half, cached on the right half
combined = np.ones((step_size, step_size * 2 + 2, 3))
combined[:, :step_size] = naive_grid
combined[:, step_size+2:] = cached_grid

ax1.imshow(combined, aspect='auto')
ax1.set_title(f"Step {step_size}: Naive (left) vs Cached (right)", fontsize=10)
ax1.axvline(x=step_size + 0.5, color='black', linewidth=2)
ax1.set_xticks([step_size//2, step_size + 2 + step_size//2])
ax1.set_xticklabels(['Naive\n(compute all)', 'Cached\n(compute last row)'])
ax1.set_yticks([])

# Legend patches
from matplotlib.patches import Patch
ax1.legend(handles=[
    Patch(facecolor=(0.2, 0.7, 0.3), label='Computed & used'),
    Patch(facecolor=(0.9, 0.3, 0.3), label='Computed & wasted'),
    Patch(facecolor=(0.85, 0.85, 0.85), label='From cache'),
], loc='lower right', fontsize=7)


# --- Panel 2: Per-step computation ---
ax2 = fig.add_subplot(2, 3, 2)
n_range = np.arange(1, CONTEXT_LEN + 1)

ax2.fill_between(n_range, n_range**2, alpha=0.3, color='red', label='Naive: t² per step')
ax2.fill_between(n_range, n_range, alpha=0.3, color='green', label='Cached: t per step')
ax2.plot(n_range, n_range**2, 'r-', linewidth=2)
ax2.plot(n_range, n_range, 'g-', linewidth=2)
ax2.set_xlabel("Step t")
ax2.set_ylabel("Attention operations")
ax2.set_title("Per-Step Cost", fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)


# --- Panel 3: Cumulative cost ---
ax3 = fig.add_subplot(2, 3, 3)
n_range_long = np.arange(1, 501)
naive_cumul = np.cumsum(n_range_long**2)
cached_cumul = np.cumsum(n_range_long)

ax3.plot(n_range_long, naive_cumul, 'r-', linewidth=2, label='Naive: Σt² ≈ n³/3')
ax3.plot(n_range_long, cached_cumul, 'g-', linewidth=2, label='Cached: Σt ≈ n²/2')
ax3.set_xlabel("Tokens generated (n)")
ax3.set_ylabel("Total attention ops")
ax3.set_title("Cumulative Cost", fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')


# --- Panel 4: Wall-clock comparison ---
ax4 = fig.add_subplot(2, 3, 4)
x_pos = np.arange(len(gen_lengths))
width = 0.35

bars1 = ax4.bar(x_pos - width/2, naive_times, width, color='red', alpha=0.7, label='Naive')
bars2 = ax4.bar(x_pos + width/2, cached_times, width, color='green', alpha=0.7, label='Cached')

ax4.set_xlabel("Generation length")
ax4.set_ylabel("Time (ms)")
ax4.set_title("Wall-Clock Time", fontsize=10)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(gen_lengths)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, axis='y')


# --- Panel 5: Memory growth ---
ax5 = fig.add_subplot(2, 3, 5)
n_range_mem = np.arange(1, 4097)
# Our model memory per token cached: 2 * L * d * 4 bytes
bytes_per_token_ours = 2 * L * EMBED_DIM * 4
bytes_per_token_gpt2 = 2 * 12 * 768 * 4
bytes_per_token_llama = 2 * 80 * 8192 * 4

ax5.plot(n_range_mem, n_range_mem * bytes_per_token_ours / 1e6,
         'b-', linewidth=2, label=f'Ours (d={EMBED_DIM}, L={L})')
ax5.plot(n_range_mem, n_range_mem * bytes_per_token_gpt2 / 1e6,
         'orange', linewidth=2, label='GPT-2 (d=768, L=12)')
ax5.plot(n_range_mem, n_range_mem * bytes_per_token_llama / 1e6,
         'r-', linewidth=2, label='LLaMA-70B (d=8192, L=80)')
ax5.set_xlabel("Sequence length")
ax5.set_ylabel("KV-cache size (MB)")
ax5.set_title("Memory Cost of KV-Cache", fontsize=10)
ax5.legend(fontsize=7)
ax5.grid(True, alpha=0.3)
ax5.set_yscale('log')


# --- Panel 6: Speedup ratio ---
ax6 = fig.add_subplot(2, 3, 6)
n_range_sp = np.arange(1, 4097)
# Theoretical speedup: Σt² / Σt = (n(n+1)(2n+1)/6) / (n(n+1)/2) = (2n+1)/3
theoretical_speedup = (2 * n_range_sp + 1) / 3

ax6.plot(n_range_sp, theoretical_speedup, 'g-', linewidth=2)
ax6.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax6.set_xlabel("Sequence length n")
ax6.set_ylabel("Speedup (naive / cached)")
ax6.set_title("Theoretical Speedup from KV-Cache", fontsize=10)
ax6.grid(True, alpha=0.3)

# Annotate key points
for n_val in [100, 1000, 4096]:
    sp = (2*n_val + 1) / 3
    ax6.annotate(f'n={n_val}: {sp:.0f}×',
                xy=(n_val, sp), fontsize=8,
                xytext=(n_val+200, sp-100),
                arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), 'kv_cache.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"  Saved visualization → {save_path}")
print()


# ===================================================================
# Summary
# ===================================================================
print(f"""
{'='*65}
{BOLD}STEP 2 SUMMARY: KV-CACHE — FROM O(n³) TO O(n²){RESET}
{'='*65}

{BOLD}The mechanism:{RESET}

  Phase 1 — Prefill (process prompt):
    Feed all seed tokens at once (normal forward pass).
    Save K,V for each layer → KV-cache initialized.

  Phase 2 — Decode (generate tokens):
    For each new token:
      1. Embed ONLY the new token          → (1, 1, d)
      2. Compute Q_new, K_new, V_new       → one token's projections
      3. Append K_new, V_new to cache      → cache grows by 1
      4. Attend: Q_new @ K_cache^T         → (1, 1, seq_len) scores
      5. Output: weights @ V_cache         → (1, 1, d)
      6. Through FFN, predict next token

{BOLD}Cost comparison:{RESET}

  Operation          Naive (step t)      KV-cached (step t)
  ─────────────      ──────────────      ──────────────────
  Q,K,V projection   t × 3d²            1 × 3d²
  Attention           t²d                 td
  FFN                t × 2d·ff           1 × 2d·ff
  ─────────────      ──────────────      ──────────────────
  Per step           O(t²d + td²)        O(td + d²)
  Total (n steps)    {RED}O(n³d){RESET}              {GREEN}O(n²d){RESET}

{BOLD}Memory cost:{RESET}

  Cache stores: 2 × L × seq_len × d floats (K and V, all layers)

  At our scale (d=64, L=4):    32 × seq_len × 4 bytes  → negligible
  At GPT-2 (d=768, L=12):     18K × seq_len bytes      → ~37 MB at 2K context
  At LLaMA-70B (d=8192, L=80): ~5M × seq_len bytes     → ~160 GB at 128K context!

  This is why memory-efficient techniques matter:
    • GQA (Grouped-Query Attention): share K,V across head groups
    • MQA (Multi-Query Attention): one K,V for all heads
    • KV-cache quantization: store cache in int8/int4
    • Sliding window attention: cap cache size

{BOLD}Critical insight:{RESET}

  {CYAN}Autoregressive ≠ recomputation.{RESET}

  Generating one-token-at-a-time does NOT mean recomputing everything.
  K and V for old tokens don't change — cache them.
  Only Q for the new token matters (it's the only new query).

  This becomes essential for:
    • {GREEN}Streaming{RESET}: first-token latency vs throughput
    • {GREEN}Agents{RESET}: long multi-turn conversations with tool use
    • {GREEN}Tool use{RESET}: model generates, pauses, gets result, continues
    • {GREEN}Long context{RESET}: 128K+ context windows need efficient KV management
""")
