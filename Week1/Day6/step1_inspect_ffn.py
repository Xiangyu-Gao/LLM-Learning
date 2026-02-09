"""
Step 1: Inspect FFN dimensions.

Examine a transformer block's parameter budget:
  - Attention: W_q, W_k, W_v, W_o  →  4 × d² parameters
  - FFN:      W_up(d→4d), W_down(4d→d)  →  8 × d² parameters

The FFN has 2× more parameters than attention!
Most model capacity is in per-token transformation, not inter-token mixing.
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Day3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Day4'))

from train_tiny_char_level_lm import vocab_size, encode, decode, data, CONTEXT_LEN
from step1_scaled_dot_product_attention import make_causal_mask
from step1_multihead_attention import MultiHeadAttention
from step1_inspect_representations import TransformerBlock, TransformerLM, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

print(f"""
{'='*65}
{BOLD}INSPECT FFN DIMENSIONS{RESET}
{'='*65}

  Every transformer block has two main components:
    1. Multi-Head Attention  — mixes information ACROSS tokens
    2. Feed-Forward Network  — transforms each token INDEPENDENTLY

  Question: where do most of the parameters live?
""")

# ===================================================================
# 1. Anatomy of a single TransformerBlock
# ===================================================================
print(f"{BOLD}1. ANATOMY OF A TRANSFORMER BLOCK{RESET}")
print(f"{'='*65}")
print()

block = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM)
d = EMBED_DIM
ff = FF_DIM
h = NUM_HEADS

print(f"  Config: d_model={d}, num_heads={h}, ff_dim={ff}")
print(f"  FFN expansion ratio: {ff/d:.1f}×")
print()

# Count parameters in each sub-component
attn_params = {}
ffn_params = {}
ln_params = {}

for name, param in block.named_parameters():
    count = param.numel()
    if name.startswith('attn.'):
        attn_params[name] = count
    elif name.startswith('ffn.'):
        ffn_params[name] = count
    elif name.startswith('ln'):
        ln_params[name] = count

total_attn = sum(attn_params.values())
total_ffn = sum(ffn_params.values())
total_ln = sum(ln_params.values())
total_block = total_attn + total_ffn + total_ln

print(f"  {BOLD}Attention parameters:{RESET}")
for name, count in attn_params.items():
    shape = dict(block.named_parameters())[name].shape
    print(f"    {name:<25} {str(tuple(shape)):>15}  = {count:>8,}")
print(f"    {'':>25} {'Total':>15}  = {total_attn:>8,}")
print()

print(f"  {BOLD}FFN parameters:{RESET}")
for name, count in ffn_params.items():
    shape = dict(block.named_parameters())[name].shape
    print(f"    {name:<25} {str(tuple(shape)):>15}  = {count:>8,}")
print(f"    {'':>25} {'Total':>15}  = {total_ffn:>8,}")
print()

print(f"  {BOLD}LayerNorm parameters:{RESET}")
for name, count in ln_params.items():
    shape = dict(block.named_parameters())[name].shape
    print(f"    {name:<25} {str(tuple(shape)):>15}  = {count:>8,}")
print(f"    {'':>25} {'Total':>15}  = {total_ln:>8,}")
print()

# Ratios
print(f"  {BOLD}Parameter budget breakdown:{RESET}")
print()
attn_pct = 100 * total_attn / total_block
ffn_pct = 100 * total_ffn / total_block
ln_pct = 100 * total_ln / total_block

bar_attn = "█" * int(40 * total_attn / total_block)
bar_ffn = "█" * int(40 * total_ffn / total_block)
bar_ln = "█" * int(40 * total_ln / total_block)

print(f"    Attention:  {total_attn:>8,} ({attn_pct:>5.1f}%)  {bar_attn}")
print(f"    FFN:        {total_ffn:>8,} ({ffn_pct:>5.1f}%)  {bar_ffn}")
print(f"    LayerNorm:  {total_ln:>8,} ({ln_pct:>5.1f}%)  {bar_ln}")
print(f"    {'─'*50}")
print(f"    Total:      {total_block:>8,}")
print()

print(f"  {GREEN}FFN/Attention ratio: {total_ffn/total_attn:.2f}×{RESET}")
print(f"  The FFN has {total_ffn/total_attn:.1f}× more parameters than attention!")
print()

# ===================================================================
# 2. The math: why 2:1 ratio?
# ===================================================================
print(f"{BOLD}2. WHY THE 2:1 RATIO (with 4× expansion){RESET}")
print(f"{'='*65}")
print()
print(f"  {BOLD}Attention:{RESET}  4 weight matrices, each d×d")
print(f"    W_q: ({d} × {d}) = {d*d:>8,}")
print(f"    W_k: ({d} × {d}) = {d*d:>8,}")
print(f"    W_v: ({d} × {d}) = {d*d:>8,}")
print(f"    W_o: ({d} × {d}) = {d*d:>8,}")
print(f"    Total: 4d²  = 4 × {d}² = {4*d*d:>8,}")
print()
print(f"  {BOLD}FFN:{RESET}  2 weight matrices + 2 bias vectors")
print(f"    W_up:   ({d} × {ff}) = {d*ff:>8,}   (expand)")
print(f"    b_up:   ({ff},)      = {ff:>8,}")
print(f"    W_down: ({ff} × {d}) = {ff*d:>8,}   (compress)")
print(f"    b_down: ({d},)       = {d:>8,}")
print(f"    Total: 2×d×4d + d + 4d  = 8d² + 5d = {2*d*ff + ff + d:>8,}")
print()
print(f"  {CYAN}With standard 4× expansion:{RESET}")
print(f"    FFN params ≈ 8d²")
print(f"    Attn params = 4d²  (no bias in our attention)")
print(f"    Ratio ≈ 8d²/4d² = {BOLD}2:1{RESET}")
print()

# ===================================================================
# 3. Scale to real models
# ===================================================================
print(f"{BOLD}3. SCALING TO REAL MODELS{RESET}")
print(f"{'='*65}")
print()
print(f"  {'Model':<20} {'d_model':>8} {'ff_dim':>8} {'Ratio':>6} {'Attn params':>14} {'FFN params':>14} {'FFN %':>6}")
print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*6} {'-'*14} {'-'*14} {'-'*6}")

real_models = [
    ("Ours (tiny)",        64,   128, 4),
    ("GPT-2 Small",       768,  3072, 4),
    ("GPT-2 Large",      1280,  5120, 4),
    ("LLaMA-7B",         4096, 11008, 4),
    ("LLaMA-70B",        8192, 28672, 4),
    ("GPT-4 (est.)",    12288, 49152, 4),
]

for name, dm, ff_d, heads in real_models:
    # Attention: 4 × d² (no bias)
    attn_p = 4 * dm * dm
    # FFN: 2 × d × ff (weights only, ignoring small bias terms)
    ffn_p = 2 * dm * ff_d
    ratio = ff_d / dm
    pct = 100 * ffn_p / (attn_p + ffn_p)
    print(f"  {name:<20} {dm:>8,} {ff_d:>8,} {ratio:>5.1f}× {attn_p:>14,} {ffn_p:>14,} {pct:>5.1f}%")

print()
print(f"  {GREEN}At every scale, FFN dominates the parameter count.{RESET}")
print(f"  In LLaMA-7B with ~2.7× expansion: FFN is ~57% of each block.")
print(f"  With 4× expansion: FFN is always ~67% of each block.")
print()

# ===================================================================
# 4. What does the FFN actually compute?
# ===================================================================
print(f"{BOLD}4. WHAT DOES THE FFN COMPUTE?{RESET}")
print(f"{'='*65}")
print()

print(f"  FFN(x) = W_down · GELU( W_up · x + b_up ) + b_down")
print()
print(f"  Step by step for a single token:")
print(f"    x: ({d},)  →  one token's hidden state")
print()

torch.manual_seed(42)
model = TransformerLM(vocab_size, CONTEXT_LEN, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS)

# Get a sample hidden state
tokens = encode("To be, or not to be")
x = torch.tensor([tokens], dtype=torch.long)
with torch.no_grad():
    emb = model.embed(x) + model.pos_embed(torch.arange(x.shape[1]))
    h = model.blocks[0].ln2(emb)  # pre-FFN input after LN
    token_h = h[0, 0]  # first token's hidden state

ffn = model.blocks[0].ffn
W_up = ffn[0]    # Linear(d→ff)
gelu = ffn[1]    # GELU
W_down = ffn[2]  # Linear(ff→d)

with torch.no_grad():
    # Step 1: expand
    expanded = W_up(token_h)  # (ff_dim,)
    print(f"    1. W_up · x + b:  ({d},) → ({ff},)")
    print(f"       Expand to {ff/d:.0f}× wider space")
    print(f"       Values range: [{expanded.min():.3f}, {expanded.max():.3f}]")
    print()

    # Step 2: GELU activation
    activated = gelu(expanded)
    alive = (activated.abs() > 0.01).float().mean().item()
    print(f"    2. GELU(·):  ({ff},) → ({ff},)")
    print(f"       Non-linear gate: smoothly zeroes out negative values")
    print(f"       Active neurons: {alive:.1%} ({int(alive*ff)}/{ff})")
    print()

    # Step 3: compress
    output = W_down(activated)  # (d,)
    print(f"    3. W_down · h + b:  ({ff},) → ({d},)")
    print(f"       Compress back to original dimension")
    print(f"       Output norm: {output.norm():.3f}")
    print()

# ===================================================================
# 5. FFN neuron activation patterns
# ===================================================================
print(f"{BOLD}5. FFN NEURON ACTIVATION PATTERNS{RESET}")
print(f"{'='*65}")
print()
print(f"  Which FFN neurons fire for different tokens?")
print()

# Train a model first
print(f"  Training 4-layer model ({NUM_LAYERS}L, {EMBED_DIM}d, {FF_DIM}ff, 1200 steps)...")
torch.manual_seed(42)
model = TransformerLM(vocab_size, CONTEXT_LEN, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS)
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

# Probe different tokens through the FFN
model.eval()
probe_text = "To be, or not to be, that is the"
tokens_list = encode(probe_text)
x = torch.tensor([tokens_list], dtype=torch.long)

# Collect FFN activations at each layer
all_activations = []  # [layer][position] = (ff_dim,) post-GELU
layer_names = [f"Layer {i+1}" for i in range(NUM_LAYERS)]

with torch.no_grad():
    emb = model.embed(x) + model.pos_embed(torch.arange(x.shape[1]))
    mask = make_causal_mask(x.shape[1])
    h = emb
    for layer_idx, block in enumerate(model.blocks):
        # Attention
        attn_out, _ = block.attn(block.ln1(h), mask)
        h = h + attn_out
        # FFN — capture intermediate
        ffn_input = block.ln2(h)
        expanded = block.ffn[0](ffn_input)    # W_up
        activated = block.ffn[1](expanded)     # GELU
        all_activations.append(activated[0].numpy())  # (C, ff_dim)
        compressed = block.ffn[2](activated)   # W_down
        h = h + compressed

# Analyze activation patterns
print(f"  Probe: \"{probe_text}\"")
print()

for layer_idx in range(NUM_LAYERS):
    acts = all_activations[layer_idx]  # (C, ff_dim)
    active_mask = np.abs(acts) > 0.01

    avg_active = active_mask.mean(axis=1)  # per position
    total_active = active_mask.any(axis=0).sum()  # neurons used by ANY token
    shared = active_mask.all(axis=0).sum()  # neurons active for ALL tokens

    print(f"  {layer_names[layer_idx]}:")
    print(f"    Avg active neurons per token: {active_mask.mean()*ff:.0f}/{ff} ({active_mask.mean():.1%})")
    print(f"    Neurons used by ANY token:    {total_active}/{ff}")
    print(f"    Neurons active for ALL tokens: {shared}/{ff} (shared)")
    print()

# Show per-token activation count for first layer
print(f"  {BOLD}Per-token activation count (Layer 1):{RESET}")
acts_l1 = all_activations[0]  # (C, ff_dim)
active_l1 = np.abs(acts_l1) > 0.01
for pos in range(len(probe_text)):
    char = probe_text[pos]
    n_active = active_l1[pos].sum()
    bar = "█" * int(40 * n_active / ff)
    print(f"    pos {pos:>2} '{char}': {n_active:>3}/{ff}  {bar}")
print()

# ===================================================================
# 6. Neuron specialization: do specific neurons respond to specific tokens?
# ===================================================================
print(f"{BOLD}6. NEURON SPECIALIZATION{RESET}")
print(f"{'='*65}")
print()
print(f"  Feed many different contexts through the model.")
print(f"  For each FFN neuron, find which input tokens activate it most.")
print()

# Collect activations over many samples
n_samples = 200
all_acts_l1 = []
all_input_chars = []

with torch.no_grad():
    for _ in range(n_samples):
        start = torch.randint(0, len(data) - CONTEXT_LEN - 1, (1,)).item()
        sample = data[start : start + CONTEXT_LEN].unsqueeze(0)
        emb = model.embed(sample) + model.pos_embed(torch.arange(CONTEXT_LEN))
        mask = make_causal_mask(CONTEXT_LEN)
        h = emb
        # Get layer 1 FFN activations
        attn_out, _ = model.blocks[0].attn(model.blocks[0].ln1(h), mask)
        h = h + attn_out
        ffn_input = model.blocks[0].ln2(h)
        expanded = model.blocks[0].ffn[0](ffn_input)
        activated = model.blocks[0].ffn[1](expanded)
        all_acts_l1.append(activated[0].numpy())  # (C, ff_dim)
        all_input_chars.append(sample[0].numpy())

all_acts_l1 = np.concatenate(all_acts_l1, axis=0)  # (n_samples*C, ff_dim)
all_input_chars = np.concatenate(all_input_chars, axis=0)  # (n_samples*C,)

# For top neurons by variance, show which characters activate them most
neuron_variance = all_acts_l1.var(axis=0)
top_neurons = np.argsort(neuron_variance)[-8:][::-1]  # top 8 by variance

print(f"  Top 8 most selective neurons (Layer 1, by activation variance):")
print()

from train_tiny_char_level_lm import idx_to_char

for neuron_idx in top_neurons:
    acts = all_acts_l1[:, neuron_idx]
    # Find top activating characters
    char_avg_act = {}
    for char_id in range(vocab_size):
        mask = all_input_chars == char_id
        if mask.sum() > 0:
            char_avg_act[char_id] = acts[mask].mean()

    sorted_chars = sorted(char_avg_act.items(), key=lambda x: x[1], reverse=True)
    top3 = sorted_chars[:3]
    bot3 = sorted_chars[-3:]

    top_str = ", ".join(f"'{idx_to_char[c]}' ({v:+.2f})" for c, v in top3)
    bot_str = ", ".join(f"'{idx_to_char[c]}' ({v:+.2f})" for c, v in bot3)
    print(f"    Neuron {neuron_idx:>3} (var={neuron_variance[neuron_idx]:.3f}):")
    print(f"      Most activated by:  {top_str}")
    print(f"      Least activated by: {bot_str}")

print()

# ===================================================================
# Visualization
# ===================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("FFN Inspection: Where the Parameters Live", fontsize=14, fontweight='bold')

# Panel 1: Parameter budget pie chart
ax = axes[0, 0]
sizes = [total_attn, total_ffn, total_ln]
labels = [f"Attention\n{total_attn:,} ({attn_pct:.0f}%)",
          f"FFN\n{total_ffn:,} ({ffn_pct:.0f}%)",
          f"LayerNorm\n{total_ln:,} ({ln_pct:.0f}%)"]
colors_pie = ['#4ECDC4', '#FF6B6B', '#95E1D3']
ax.pie(sizes, labels=labels, colors=colors_pie, startangle=90,
       textprops={'fontsize': 9}, autopct='')
ax.set_title(f"Parameter Budget per Block\n(d={d}, ff={ff})")

# Panel 2: Activation heatmap (positions × neurons) for layer 1
ax = axes[0, 1]
acts_display = all_activations[0][:, :64]  # first 64 neurons
im = ax.imshow(acts_display.T, aspect='auto', cmap='RdBu_r',
               vmin=-acts_display.max(), vmax=acts_display.max())
ax.set_xlabel("Token Position")
ax.set_ylabel("FFN Neuron Index")
ax.set_title("FFN Activations (Layer 1, first 64 neurons)")
ax.set_xticks(range(len(probe_text)))
ax.set_xticklabels(list(probe_text), fontsize=6)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Panel 3: Active neuron count per layer
ax = axes[1, 0]
for layer_idx in range(NUM_LAYERS):
    acts = all_activations[layer_idx]
    active_counts = (np.abs(acts) > 0.01).sum(axis=1)  # per position
    ax.plot(active_counts, 'o-', label=f'Layer {layer_idx+1}',
            markersize=4, linewidth=1.5)
ax.set_xlabel("Token Position")
ax.set_ylabel("Active Neurons")
ax.set_title(f"Active FFN Neurons per Token (of {ff})")
ax.set_xticks(range(len(probe_text)))
ax.set_xticklabels(list(probe_text), fontsize=6, rotation=0)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 4: Scaling — FFN vs Attn params at different d_model sizes
ax = axes[1, 1]
d_range = np.array([64, 128, 256, 512, 768, 1024, 2048, 4096, 8192])
attn_curve = 4 * d_range**2
ffn_curve_4x = 2 * d_range * (4 * d_range)  # 4× expansion
ffn_curve_2_7x = 2 * d_range * (2.7 * d_range)  # LLaMA-style

ax.plot(d_range, attn_curve / 1e6, 's-', label='Attention (4d²)',
        color='#4ECDC4', linewidth=2, markersize=5)
ax.plot(d_range, ffn_curve_4x / 1e6, 'o-', label='FFN 4× (8d²)',
        color='#FF6B6B', linewidth=2, markersize=5)
ax.plot(d_range, ffn_curve_2_7x / 1e6, '^--', label='FFN 2.7× (5.4d²)',
        color='#FF9999', linewidth=2, markersize=5)
ax.set_xlabel("d_model")
ax.set_ylabel("Parameters (millions)")
ax.set_title("Parameter Scaling: Attention vs FFN")
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), 'inspect_ffn.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved visualization → {save_path}")
print()

# ===================================================================
# Summary
# ===================================================================
print(f"""
{'='*65}
{BOLD}STEP 1 SUMMARY: FFN DIMENSIONS{RESET}
{'='*65}

{BOLD}Architecture:{RESET}

  Each transformer block:
    stream → LN → Attention → + residual → LN → FFN → + residual
                  (4d² params)                  (8d² params)

  FFN structure:
    x → Linear({d}→{ff}) → GELU → Linear({ff}→{d}) → output
        (expand {ff/d:.0f}×)           (compress back)

{BOLD}Key findings:{RESET}

  1. {CYAN}FFN dominates the parameter budget{RESET}
     Our tiny model (2× expansion):
       Attention: {total_attn:,} params ({attn_pct:.0f}%) ≈ FFN: {total_ffn:,} ({ffn_pct:.0f}%)

     Real models use 4× expansion, making FFN ~67% of each block:
       FFN params ≈ 8d², Attention params = 4d² → 2:1 ratio.
       This holds at EVERY scale (GPT-2 to LLaMA-70B).

  2. {CYAN}FFN operates per-token independently{RESET}
     Each token position goes through the FFN separately.
     No information flows between positions in the FFN.
     This is why FFN is sometimes called a "pointwise MLP".

  3. {CYAN}FFN neurons show specialization{RESET}
     Different neurons activate for different input characters.
     Only ~50-70% of neurons are active for any given token.
     This sparsity is why Mixture-of-Experts (MoE) works:
     if most neurons are dormant, only route to the active ones.

  4. {CYAN}The expand-compress pattern is key{RESET}
     Expand: project into high-dimensional space ({d}→{ff})
     Activate: GELU zeroes out irrelevant dimensions
     Compress: project back to original size ({ff}→{d})

     The high-dimensional intermediate space lets the FFN
     represent complex per-token transformations as a
     composition of simple linear operations + gating.

{BOLD}Why this matters:{RESET}

  {GREEN}Attention decides WHAT information to gather from context.
  FFN decides HOW to transform that gathered information.{RESET}

  Most of the model's "knowledge" (facts, patterns, rules)
  is stored in FFN weights, not attention weights.
  This is why FFN parameters dominate the parameter count:
  storing knowledge requires capacity.
""")
