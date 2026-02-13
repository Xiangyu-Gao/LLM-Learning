"""
Step 3: Attention vs FFN roles.

Zero out attention or FFN contributions at each layer to isolate
what each sub-layer actually does.

  Attention → mixes information ACROSS tokens (inter-token)
  FFN       → transforms information PER token (intra-token)
  Residual  → preserves stability

Together they form a refinement loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Day1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Day2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Day3'))
sys.path.insert(0, os.path.dirname(__file__))

from train_tiny_char_level_lm import vocab_size, encode, decode, data, CONTEXT_LEN
from step1_scaled_dot_product_attention import make_causal_mask
from step1_multihead_attention import MultiHeadAttention
from step1_inspect_representations import EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Transformer block with ablation switches
# ---------------------------------------------------------------------------
class AblateBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.disable_attn = False
        self.disable_ffn = False

    def forward(self, x, mask=None):
        attn_out, attn_w = self.attn(self.ln1(x), mask)
        if self.disable_attn:
            attn_out = torch.zeros_like(attn_out)
        x = x + attn_out

        ffn_out = self.ffn(self.ln2(x))
        if self.disable_ffn:
            ffn_out = torch.zeros_like(ffn_out)
        x = x + ffn_out
        return x, attn_w

class AblateTransformerLM(nn.Module):
    def __init__(self, vocab_size, context_len, embed_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        self.context_len = context_len
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(context_len, embed_dim)
        self.blocks = nn.ModuleList([
            AblateBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, C = x.shape
        emb = self.embed(x) + self.pos_embed(torch.arange(C, device=x.device))
        mask = make_causal_mask(C).to(x.device)
        h = emb
        for block in self.blocks:
            h, _ = block(h, mask)
        return self.out_proj(self.ln_final(h))

    def set_ablation(self, disable_attn=False, disable_ffn=False):
        for block in self.blocks:
            block.disable_attn = disable_attn
            block.disable_ffn = disable_ffn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_loss(model, num_batches=20):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            ix = torch.randint(0, len(data) - CONTEXT_LEN - 1, (128,))
            x = torch.stack([data[i : i + CONTEXT_LEN] for i in ix])
            y = torch.stack([data[i + 1 : i + CONTEXT_LEN + 1] for i in ix])
            total += F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1)).item()
    return total / num_batches

def generate(model, seed_text, length=120):
    model.eval()
    tokens = encode(seed_text)
    generated = list(tokens)
    with torch.no_grad():
        for _ in range(length):
            inp = torch.tensor([generated[-CONTEXT_LEN:]], dtype=torch.long)
            logits = model(inp)
            probs = F.softmax(logits[0, -1], dim=0)
            generated.append(torch.multinomial(probs, 1).item())
    return decode(generated[len(tokens):])

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
print(f"{BOLD}Training {NUM_LAYERS}-layer transformer...{RESET}")
torch.manual_seed(42)
model = AblateTransformerLM(vocab_size, CONTEXT_LEN, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

for step in range(1, 1201):
    ix = torch.randint(0, len(data) - CONTEXT_LEN - 1, (128,))
    x = torch.stack([data[i : i + CONTEXT_LEN] for i in ix])
    y = torch.stack([data[i + 1 : i + CONTEXT_LEN + 1] for i in ix])
    loss = F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 300 == 0:
        print(f"  Step {step:>4d} | Loss: {loss.item():.4f}")

print()

seed = "To be, or not to be, that is the"

# ===========================================================================
# EXPERIMENT 1: Global ablation — zero ALL attn or ALL FFN
# ===========================================================================
print(f"{'='*65}")
print(f"{BOLD}EXPERIMENT 1: GLOBAL ABLATION{RESET}")
print(f"{'='*65}")
print(f"  Disable ALL attention or ALL FFN across every layer.")
print()

conditions = [
    ("Baseline (all active)", False, False),
    ("No Attention (FFN only)", True, False),
    ("No FFN (Attention only)", False, True),
    ("No Attn + No FFN (embed only)", True, True),
]

results = {}
for label, no_attn, no_ffn in conditions:
    model.set_ablation(disable_attn=no_attn, disable_ffn=no_ffn)
    torch.manual_seed(99)
    loss_val = compute_loss(model)
    torch.manual_seed(99)
    gen = generate(model, seed, length=80)
    results[label] = {'loss': loss_val, 'gen': gen}

    if no_attn and no_ffn:
        color = DIM
    elif no_attn:
        color = RED
    elif no_ffn:
        color = YELLOW
    else:
        color = GREEN

    print(f"  {color}{BOLD}{label}{RESET}")
    print(f"    Loss: {loss_val:.4f}")
    print(f"    Gen:  \"{gen[:80]}\"")
    print()

# Reset
model.set_ablation(False, False)

baseline_loss = results["Baseline (all active)"]['loss']
no_attn_loss = results["No Attention (FFN only)"]['loss']
no_ffn_loss = results["No FFN (Attention only)"]['loss']

print(f"  {CYAN}Loss impact:{RESET}")
print(f"    Removing Attention: +{no_attn_loss - baseline_loss:.4f} ({(no_attn_loss - baseline_loss) / baseline_loss * 100:.0f}% worse)")
print(f"    Removing FFN:       +{no_ffn_loss - baseline_loss:.4f} ({(no_ffn_loss - baseline_loss) / baseline_loss * 100:.0f}% worse)")
more_important = "Attention" if no_attn_loss > no_ffn_loss else "FFN"
print(f"    {BOLD}{more_important} removal hurts more.{RESET}")
print()

# ===========================================================================
# EXPERIMENT 2: Per-layer ablation — which layers' attn/ffn matter most?
# ===========================================================================
print(f"{'='*65}")
print(f"{BOLD}EXPERIMENT 2: PER-LAYER ABLATION{RESET}")
print(f"{'='*65}")
print(f"  Disable attn or FFN in ONE layer at a time.")
print()

per_layer_attn = []
per_layer_ffn = []

print(f"  {'Layer':<10} {'No Attn (loss)':<18} {'Δ':>8}   {'No FFN (loss)':<18} {'Δ':>8}")
print(f"  {'-'*10} {'-'*18} {'-'*8}   {'-'*18} {'-'*8}")

for layer_idx in range(NUM_LAYERS):
    # Zero out attention in this layer only
    for i, block in enumerate(model.blocks):
        block.disable_attn = (i == layer_idx)
        block.disable_ffn = False
    loss_no_attn = compute_loss(model)
    delta_attn = loss_no_attn - baseline_loss
    per_layer_attn.append(delta_attn)

    # Zero out FFN in this layer only
    for i, block in enumerate(model.blocks):
        block.disable_attn = False
        block.disable_ffn = (i == layer_idx)
    loss_no_ffn = compute_loss(model)
    delta_ffn = loss_no_ffn - baseline_loss
    per_layer_ffn.append(delta_ffn)

    attn_color = RED if delta_attn > 0.5 else (YELLOW if delta_attn > 0.1 else GREEN)
    ffn_color = RED if delta_ffn > 0.5 else (YELLOW if delta_ffn > 0.1 else GREEN)

    print(f"  Layer {layer_idx+1:<4} {loss_no_attn:<18.4f} {attn_color}+{delta_attn:<7.4f}{RESET}   {loss_no_ffn:<18.4f} {ffn_color}+{delta_ffn:<7.4f}{RESET}")

# Reset
model.set_ablation(False, False)

print()
most_critical_attn = np.argmax(per_layer_attn) + 1
most_critical_ffn = np.argmax(per_layer_ffn) + 1
print(f"  Most critical attention: Layer {most_critical_attn} (Δ = +{max(per_layer_attn):.4f})")
print(f"  Most critical FFN:       Layer {most_critical_ffn} (Δ = +{max(per_layer_ffn):.4f})")
print()

# ===========================================================================
# EXPERIMENT 3: Attention = cross-token mixing, FFN = per-token transform
# ===========================================================================
print(f"{'='*65}")
print(f"{BOLD}EXPERIMENT 3: WHAT DOES EACH SUB-LAYER ACTUALLY DO?{RESET}")
print(f"{'='*65}")
print()

# Demonstrate that attention MIXES tokens but FFN doesn't
# by checking: does position i's output depend on position j's input?

print(f"  {BOLD}Test: Does changing one token affect OTHER tokens' representations?{RESET}")
print()

tokens_a = encode("To be, or not to be, that is the")
tokens_b = list(tokens_a)
tokens_b[0] = encode("a")[0]  # change first character from 'T' to 'a'

x_a = torch.tensor([tokens_a], dtype=torch.long)
x_b = torch.tensor([tokens_b], dtype=torch.long)

model.eval()
model.set_ablation(False, False)

# Get outputs with all active
with torch.no_grad():
    logits_a = model(x_a)
    logits_b = model(x_b)

# How much do OTHER positions change?
diff_all = (logits_a[0, 1:] - logits_b[0, 1:]).abs().mean().item()

# Now with no attention (FFN only) — does the change propagate?
model.set_ablation(disable_attn=True, disable_ffn=False)
with torch.no_grad():
    logits_a_no_attn = model(x_a)
    logits_b_no_attn = model(x_b)

diff_no_attn = (logits_a_no_attn[0, 1:] - logits_b_no_attn[0, 1:]).abs().mean().item()

# Now with no FFN (attention only) — does the change propagate?
model.set_ablation(disable_attn=False, disable_ffn=True)
with torch.no_grad():
    logits_a_no_ffn = model(x_a)
    logits_b_no_ffn = model(x_b)

diff_no_ffn = (logits_a_no_ffn[0, 1:] - logits_b_no_ffn[0, 1:]).abs().mean().item()

model.set_ablation(False, False)

print(f"  Changed token 0 from 'T' to 'a'. Measured effect on OTHER positions:")
print()
print(f"    {'Condition':<30} {'Avg change at other positions':>30}")
print(f"    {'-'*30} {'-'*30}")
print(f"    {'All active':<30} {diff_all:>30.4f}")
print(f"    {'No Attention (FFN only)':<30} {diff_no_attn:>30.4f}")
print(f"    {'No FFN (Attention only)':<30} {diff_no_ffn:>30.4f}")
print()

if diff_no_attn < diff_all * 0.1:
    print(f"  {GREEN}Without attention, the change does NOT propagate.{RESET}")
    print(f"  FFN is position-independent — it processes each token in isolation.")
else:
    print(f"  {YELLOW}Some propagation remains (through embedding/positional structure).{RESET}")

if diff_no_ffn > diff_all * 0.3:
    print(f"  {GREEN}Without FFN, the change STILL propagates via attention.{RESET}")
    print(f"  Attention is the mechanism for cross-token communication.")
print()

# ===========================================================================
# Visualization
# ===========================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Attention vs FFN: What Does Each Sub-layer Do?", fontsize=14, fontweight='bold')

# Panel 1: Global ablation losses
ax = axes[0]
labels = ['Baseline', 'No Attn\n(FFN only)', 'No FFN\n(Attn only)', 'Neither']
losses = [results[c[0]]['loss'] for c in conditions]
colors = ['green', 'red', 'orange', 'gray']
bars = ax.bar(labels, losses, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_ylabel("Loss")
ax.set_title("Global Ablation")
for bar, val in zip(bars, losses):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}', ha='center', va='bottom', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Panel 2: Per-layer ablation (grouped bar chart)
ax = axes[1]
x_pos = np.arange(NUM_LAYERS)
width = 0.35
ax.bar(x_pos - width/2, per_layer_attn, width, label='Remove Attn', color='steelblue')
ax.bar(x_pos + width/2, per_layer_ffn, width, label='Remove FFN', color='darkorange')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'Layer {i+1}' for i in range(NUM_LAYERS)])
ax.set_ylabel("Loss Increase (Δ)")
ax.set_title("Per-Layer Ablation")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Panel 3: Cross-token propagation
ax = axes[2]
labels = ['All active', 'No Attn\n(FFN only)', 'No FFN\n(Attn only)']
vals = [diff_all, diff_no_attn, diff_no_ffn]
colors = ['green', 'red', 'orange']
bars = ax.bar(labels, vals, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_ylabel("Avg Logit Change at Other Positions")
ax.set_title("Cross-Token Information Flow")
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), 'attn_vs_ffn.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved visualization → {save_path}")
print()

# ===========================================================================
# Summary
# ===========================================================================
print(f"""
{'='*65}
{BOLD}STEP 3 SUMMARY: ATTENTION vs FFN ROLES{RESET}
{'='*65}

{BOLD}The transformer refinement loop:{RESET}

  for each layer:
      stream += Attention(stream)   ← mix across tokens
      stream += FFN(stream)         ← transform per token

{BOLD}What we proved:{RESET}

  1. {CYAN}Attention = inter-token communication{RESET}
     When we changed token 0 from 'T' to 'a':
       - With attention:  other tokens' outputs changed by {diff_all:.3f}
       - Without attention: change was only {diff_no_attn:.3f} (no propagation)
     Attention is HOW tokens learn about each other.

  2. {CYAN}FFN = per-token transformation{RESET}
     FFN processes each position independently (like a pointwise MLP).
     It doesn't mix information across positions — it TRANSFORMS
     the information that attention has already gathered.
     Think of it as: attention COLLECTS evidence, FFN PROCESSES it.

  3. {CYAN}Both are necessary{RESET}
     No attention → loss {no_attn_loss:.2f} (can't see context, each token alone)
     No FFN       → loss {no_ffn_loss:.2f} (can see context, can't process it)
     Baseline     → loss {baseline_loss:.2f}

  4. {CYAN}Per-layer importance varies{RESET}
     Layer {most_critical_attn} attention matters most (early layers do heavy context mixing).
     Layer {most_critical_ffn} FFN matters most (transforms the gathered information).

{BOLD}The mental model:{RESET}

  Attention is like a {GREEN}meeting{RESET} — tokens share information.
  FFN is like {GREEN}individual thinking{RESET} — each token processes what it heard.
  Residual is like {GREEN}notes{RESET} — nothing discussed is forgotten.

  One cycle: meet → think → meet → think → ... → predict.
""")
