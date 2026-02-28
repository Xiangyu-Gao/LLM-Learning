"""
Step 2: Visualize Distributions — See decoding as a policy.

Plot for multiple prediction positions:
  1. Full softmax curve (sorted by probability)
  2. Cumulative probability curve
  3. Highlight tokens chosen by greedy, top-k, top-p

Key observations:
  - The long tail is massive (most tokens have near-zero probability)
  - Top 5 tokens often contain >80% of the mass
  - Top-p dynamically adapts to confidence, top-k doesn't

Key insight: Decoding is a POLICY applied after training.
  Greedy = deterministic policy
  Top-p  = stochastic exploration
  Temperature = entropy scaling
  You are now in RL territory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math
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
# Train model (same 1-layer, 100 steps as step1)
# ===================================================================
print(f"""
{'='*65}
{BOLD}VISUALIZE DISTRIBUTIONS: DECODING AS A POLICY{RESET}
{'='*65}
""")

print(f"{BOLD}Training 1-layer model (100 steps)...{RESET}")
torch.manual_seed(42)
model = TransformerLM(vocab_size, CONTEXT_LEN, EMBED_DIM, NUM_HEADS, FF_DIM, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

for step in range(1, 101):
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
# Get predictions at multiple positions
# ===================================================================
seed = "To be, or not to be, that is"
tokens = encode(seed)
x = torch.tensor([tokens], dtype=torch.long)

with torch.no_grad():
    logits = model(x)[0]  # (C, V)


def get_topk_set(probs, k):
    """Return indices of top-k tokens."""
    _, idx = torch.topk(probs, k)
    return set(idx.tolist())


def get_topp_set(probs, p):
    """Return indices of tokens in the top-p nucleus."""
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    mask = (cumsum - sorted_probs) < p
    return set(sorted_idx[mask].tolist())


# ===================================================================
# 1. Analyze the long tail
# ===================================================================
print(f"{BOLD}1. THE LONG TAIL{RESET}")
print(f"{'='*65}")
print()
print(f"  Seed: \"{seed}\"")
print()

# Pick a few interesting positions: confident, uncertain, and medium
def compute_entropy(p):
    return -(p * torch.log(p + 1e-10)).sum().item()

positions_info = []
for pos in range(len(seed) - 1):
    p = F.softmax(logits[pos], dim=0)
    ent = compute_entropy(p)
    positions_info.append((pos, ent, p))

# Sort by entropy to find confident/uncertain positions
sorted_by_entropy = sorted(positions_info, key=lambda x: x[1])
confident_pos = sorted_by_entropy[0][0]
medium_pos = sorted_by_entropy[len(sorted_by_entropy)//2][0]
uncertain_pos = sorted_by_entropy[-1][0]

analysis_positions = [
    (confident_pos, "CONFIDENT"),
    (medium_pos, "MEDIUM"),
    (uncertain_pos, "UNCERTAIN"),
]

for pos, label in analysis_positions:
    p = F.softmax(logits[pos], dim=0)
    sorted_p, sorted_i = torch.sort(p, descending=True)
    cumsum = torch.cumsum(sorted_p, dim=0)

    char_in = seed[pos]
    char_true = seed[pos + 1] if pos + 1 < len(seed) else "?"
    ent = compute_entropy(p)

    # Mass concentration
    top1_mass = sorted_p[0].item()
    top3_mass = sorted_p[:3].sum().item()
    top5_mass = sorted_p[:5].sum().item()
    top10_mass = sorted_p[:10].sum().item()

    # How many tokens to reach 90%?
    n_for_90 = (cumsum < 0.9).sum().item() + 1
    n_for_99 = (cumsum < 0.99).sum().item() + 1

    # Tail: how many tokens have prob < 1%?
    tail_count = (p < 0.01).sum().item()

    print(f"  Position {pos} ('{char_in}' → '{char_true}') — {label} (entropy={ent:.2f})")
    print()
    print(f"    Mass concentration:")
    print(f"      Top  1 token:  {top1_mass:>6.1%} {'█' * int(40 * top1_mass)}")
    print(f"      Top  3 tokens: {top3_mass:>6.1%} {'█' * int(40 * top3_mass)}")
    print(f"      Top  5 tokens: {top5_mass:>6.1%} {'█' * int(40 * top5_mass)}")
    print(f"      Top 10 tokens: {top10_mass:>6.1%} {'█' * int(40 * top10_mass)}")
    print()
    print(f"    Tokens needed for 90% mass: {n_for_90} of {vocab_size}")
    print(f"    Tokens needed for 99% mass: {n_for_99} of {vocab_size}")
    print(f"    Tokens with prob < 1%:      {tail_count} of {vocab_size} ({100*tail_count/vocab_size:.0f}%)")
    print()

# ===================================================================
# 2. Decoding strategy comparison per position
# ===================================================================
print(f"{BOLD}2. DECODING STRATEGIES: WHICH TOKENS GET SELECTED?{RESET}")
print(f"{'='*65}")
print()

K = 5
P = 0.9

for pos, label in analysis_positions:
    p = F.softmax(logits[pos], dim=0)
    sorted_p, sorted_i = torch.sort(p, descending=True)

    greedy_set = get_topk_set(p, 1)
    topk_set = get_topk_set(p, K)
    topp_set = get_topp_set(p, P)

    char_in = seed[pos]
    print(f"  Position {pos} ('{char_in}') — {label}")
    print(f"    Greedy: 1 token  |  Top-k(k={K}): {K} tokens  |  Top-p(p={P}): {len(topp_set)} tokens")
    print()
    print(f"    {'Rank':<6} {'Char':<6} {'Prob':>7} {'Cum':>7} {'Greedy':>7} {'Top-k':>6} {'Top-p':>6}")
    print(f"    {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*6}")

    cum = 0.0
    for rank in range(min(15, vocab_size)):
        idx = sorted_i[rank].item()
        char = repr(idx_to_char[idx])
        prob = sorted_p[rank].item()
        cum += prob

        in_greedy = "●" if idx in greedy_set else "·"
        in_topk = "●" if idx in topk_set else "·"
        in_topp = "●" if idx in topp_set else "·"

        # Color the row based on selection
        if idx in greedy_set:
            color = GREEN
        elif idx in topp_set:
            color = CYAN
        elif idx in topk_set:
            color = YELLOW
        else:
            color = DIM

        print(f"    {color}{rank+1:<6} {char:<6} {prob:>6.3f} {cum:>6.3f} {in_greedy:>7} {in_topk:>6} {in_topp:>6}{RESET}")

    remaining = vocab_size - min(15, vocab_size)
    if remaining > 0:
        print(f"    {DIM}... {remaining} more tokens in the long tail{RESET}")
    print()

print(f"  {GREEN}● = selected by greedy{RESET}")
print(f"  {CYAN}● = in top-p nucleus{RESET}")
print(f"  {YELLOW}● = in top-k set{RESET}")
print(f"  {DIM}· = excluded{RESET}")
print()

# ===================================================================
# 3. Top-p adapts, top-k doesn't
# ===================================================================
print(f"{BOLD}3. TOP-P ADAPTS, TOP-K DOESN'T{RESET}")
print(f"{'='*65}")
print()

print(f"  Tokens selected at every position in \"{seed[:20]}...\"")
print()
print(f"  {'Pos':<5} {'Char':<6} {'Entropy':>8} {'Top-k(5)':>9} {'Top-p(.9)':>10} {'Ratio':>6}")
print(f"  {'-'*5} {'-'*6} {'-'*8} {'-'*9} {'-'*10} {'-'*6}")

topk_counts = []
topp_counts = []

for pos in range(len(seed) - 1):
    p = F.softmax(logits[pos], dim=0)
    ent = compute_entropy(p)

    n_topk = K  # always K
    n_topp = len(get_topp_set(p, P))

    topk_counts.append(n_topk)
    topp_counts.append(n_topp)

    ratio = n_topp / n_topk
    char = seed[pos]

    # Visual indicator
    if n_topp < K:
        indicator = f"{GREEN}↓ fewer{RESET}"
    elif n_topp > K:
        indicator = f"{RED}↑ more{RESET}"
    else:
        indicator = f"  equal"

    print(f"  {pos:<5} '{char}'    {ent:>7.2f} {n_topk:>9} {n_topp:>10} {ratio:>5.1f}× {indicator}")

avg_topp = np.mean(topp_counts)
print()
print(f"  Top-k always selects {K} tokens (by definition).")
print(f"  Top-p selects {min(topp_counts)}-{max(topp_counts)} tokens (avg {avg_topp:.1f}).")
print()
print(f"  {CYAN}When confident:{RESET} top-p uses fewer tokens → more focused (like greedy).")
print(f"  {CYAN}When uncertain:{RESET} top-p uses more tokens → more diverse (explores).")
print(f"  {YELLOW}Top-k is blind to confidence — always the same fixed window.{RESET}")
print()

# ===================================================================
# 4. The RL perspective
# ===================================================================
print(f"{BOLD}4. DECODING AS REINFORCEMENT LEARNING{RESET}")
print(f"{'='*65}")
print()
print(f"  The model is a POLICY NETWORK. At each step:")
print(f"    State  = token sequence so far")
print(f"    Action = next token to emit")
print(f"    Policy = softmax(logits / T)")
print()
print(f"  Decoding strategy = policy type:")
print()
print(f"    {BOLD}Greedy{RESET}     = deterministic policy  (exploit only)")
print(f"    {BOLD}Top-k{RESET}      = ε-greedy with fixed k (rigid exploration)")
print(f"    {BOLD}Top-p{RESET}      = adaptive exploration  (confidence-aware)")
print(f"    {BOLD}Temperature{RESET} = entropy regularizer   (softens/sharpens policy)")
print()
print(f"  This is why decoding is called 'post-training alignment':")
print(f"    The base model (weights) is FROZEN after training.")
print(f"    The decoding strategy is the ONLY knob you can turn.")
print(f"    Different strategies → different behavior from the same model.")
print()
print(f"  In RL terms:")
print(f"    {GREEN}RLHF{RESET} changes the model's WEIGHTS to shift the distribution.")
print(f"    {GREEN}Decoding{RESET} changes which SAMPLES you draw from the distribution.")
print(f"    Both affect output quality. They're complementary.")
print()

# Show the explore-exploit tradeoff
print(f"  The explore/exploit tradeoff at each position:")
print()

for pos, label in analysis_positions:
    p = F.softmax(logits[pos], dim=0)
    ent = compute_entropy(p)
    max_ent = math.log(vocab_size)
    exploit_score = 1 - ent / max_ent

    bar_exploit = "█" * int(30 * exploit_score)
    bar_explore = "░" * (30 - int(30 * exploit_score))

    print(f"  pos {pos:>2} ('{seed[pos]}') {label:>10}: [{GREEN}{bar_exploit}{CYAN}{bar_explore}{RESET}]")
    print(f"  {'':>26}  {'exploit':>9} ← {exploit_score:.0%} → {'explore':>7}")

print()

# ===================================================================
# Visualization: 6-panel figure
# ===================================================================
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
fig.suptitle("Decoding as a Policy: Visualizing Token Distributions",
             fontsize=16, fontweight='bold', y=0.98)

for panel_idx, (pos, label) in enumerate(analysis_positions):
    p = F.softmax(logits[pos], dim=0)
    sorted_p, sorted_i = torch.sort(p, descending=True)
    cumsum = torch.cumsum(sorted_p, dim=0)

    sorted_p_np = sorted_p.numpy()
    cumsum_np = cumsum.numpy()

    greedy_set = get_topk_set(p, 1)
    topk_set = get_topk_set(p, K)
    topp_set = get_topp_set(p, P)

    char_in = seed[pos]
    char_true = seed[pos + 1] if pos + 1 < len(seed) else "?"
    ent = compute_entropy(p)

    # --- Left panel: probability distribution ---
    ax = fig.add_subplot(gs[panel_idx, 0])
    x_range = np.arange(vocab_size)

    # Bar colors based on decoding strategy
    bar_colors = []
    for rank in range(vocab_size):
        idx = sorted_i[rank].item()
        if idx in greedy_set:
            bar_colors.append('#2ecc71')  # green
        elif idx in topp_set and idx in topk_set:
            bar_colors.append('#3498db')  # blue (both)
        elif idx in topp_set:
            bar_colors.append('#9b59b6')  # purple (top-p only)
        elif idx in topk_set:
            bar_colors.append('#e67e22')  # orange (top-k only)
        else:
            bar_colors.append('#bdc3c7')  # gray (tail)

    ax.bar(x_range, sorted_p_np, color=bar_colors, width=1.0, edgecolor='none')
    ax.set_xlabel("Token rank (sorted by probability)")
    ax.set_ylabel("Probability")
    ax.set_title(f"Pos {pos} '{char_in}'→'{char_true}' — {label} (H={ent:.2f})")

    # Add token labels for top tokens
    for rank in range(min(6, vocab_size)):
        idx = sorted_i[rank].item()
        char = idx_to_char[idx]
        if char == '\n':
            char = '\\n'
        ax.annotate(f"'{char}'", (rank, sorted_p_np[rank]),
                    ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_xlim(-0.5, vocab_size - 0.5)

    # Legend
    patches = [
        mpatches.Patch(color='#2ecc71', label='Greedy'),
        mpatches.Patch(color='#3498db', label=f'Top-k({K}) ∩ Top-p({P})'),
        mpatches.Patch(color='#9b59b6', label=f'Top-p({P}) only'),
        mpatches.Patch(color='#e67e22', label=f'Top-k({K}) only'),
        mpatches.Patch(color='#bdc3c7', label='Tail (excluded)'),
    ]
    ax.legend(handles=patches, fontsize=6, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')

    # --- Right panel: cumulative distribution ---
    ax2 = fig.add_subplot(gs[panel_idx, 1])
    ax2.plot(x_range, cumsum_np, 'b-', linewidth=2.5, label='Cumulative prob')

    # Mark where top-k and top-p cut off
    topk_cum = sorted_p_np[:K].sum()
    ax2.axhline(y=topk_cum, color='#e67e22', linestyle='--', alpha=0.7,
                label=f'Top-k({K}) covers {topk_cum:.1%}')
    ax2.axvline(x=K-0.5, color='#e67e22', linestyle=':', alpha=0.5)

    n_topp = len(topp_set)
    ax2.axhline(y=P, color='#9b59b6', linestyle='--', alpha=0.7,
                label=f'Top-p({P}) threshold')
    ax2.axvline(x=n_topp-0.5, color='#9b59b6', linestyle=':', alpha=0.5)

    # Shade the top-k region
    ax2.fill_between(x_range[:K], 0, cumsum_np[:K], alpha=0.15, color='#e67e22')
    # Shade the top-p region
    ax2.fill_between(x_range[:n_topp], 0, cumsum_np[:n_topp], alpha=0.1, color='#9b59b6')

    ax2.axhline(y=0.99, color='gray', linestyle=':', alpha=0.3)
    ax2.text(vocab_size * 0.6, 0.99, '99%', fontsize=8, color='gray', va='bottom')

    ax2.set_xlabel("Number of tokens (sorted by probability)")
    ax2.set_ylabel("Cumulative probability")
    ax2.set_title(f"Cumulative: top-k={K} tokens vs top-p={n_topp} tokens")
    ax2.legend(fontsize=7, loc='lower right')
    ax2.set_xlim(-0.5, vocab_size - 0.5)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

plt.savefig(os.path.join(os.path.dirname(__file__), 'distributions.png'),
            dpi=150, bbox_inches='tight')
print(f"Saved distribution plots → distributions.png")

# Second figure: adaptive top-p vs fixed top-k across positions
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))
fig2.suptitle("Top-p Adapts to Confidence, Top-k Doesn't",
              fontsize=14, fontweight='bold')

# Panel 1: token count per position
ax = axes2[0]
positions = range(len(seed) - 1)
ax.bar(positions, topp_counts, alpha=0.7, color='#9b59b6', label=f'Top-p (p={P})')
ax.axhline(y=K, color='#e67e22', linewidth=2.5, linestyle='--', label=f'Top-k (k={K})')
ax.set_xlabel("Position in sequence")
ax.set_ylabel("Tokens in candidate set")
ax.set_title("Candidate Set Size")
ax.set_xticks(list(positions))
ax.set_xticklabels(list(seed[:-1]), fontsize=6)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Panel 2: entropy vs top-p count
ax = axes2[1]
ents = [compute_entropy(F.softmax(logits[pos], dim=0)) for pos in range(len(seed) - 1)]
ax.scatter(ents, topp_counts, c='#9b59b6', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
ax.axhline(y=K, color='#e67e22', linewidth=2, linestyle='--', label=f'Top-k always = {K}')

# Trend line
z = np.polyfit(ents, topp_counts, 1)
x_fit = np.linspace(min(ents), max(ents), 100)
ax.plot(x_fit, np.polyval(z, x_fit), '--', color='#9b59b6', alpha=0.5, label='Top-p trend')

ax.set_xlabel("Entropy (model uncertainty)")
ax.set_ylabel("Top-p candidate set size")
ax.set_title("Top-p Scales with Uncertainty")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig(os.path.join(os.path.dirname(__file__), 'topk_vs_topp.png'),
             dpi=150, bbox_inches='tight')
print(f"Saved top-k vs top-p comparison → topk_vs_topp.png")
print()

# ===================================================================
# Summary
# ===================================================================
print(f"""
{'='*65}
{BOLD}STEP 2 SUMMARY: VISUALIZE DISTRIBUTIONS{RESET}
{'='*65}

{BOLD}What we plotted:{RESET}

  1. Full softmax curve — sorted probability for all {vocab_size} tokens
  2. Cumulative probability curve — how fast mass accumulates
  3. Which tokens each strategy selects (greedy, top-k, top-p)

{BOLD}What we observed:{RESET}

  1. {CYAN}The long tail is massive{RESET}
     Most tokens have near-zero probability.
     Even at uncertain positions, 90% of mass is in ~{int(avg_topp)}-10 tokens.
     The remaining {vocab_size - 10}+ tokens share <10% of the mass.

  2. {CYAN}Top-5 tokens often contain most mass{RESET}
     At confident positions: top-1 alone has >90%.
     At uncertain positions: top-5 still has >60-80%.
     The distribution is ALWAYS heavy-headed.

  3. {CYAN}Top-p adapts, top-k doesn't{RESET}
     Top-k always picks exactly {K} tokens — even when
     the model is 95% sure (wasting 4 slots on junk)
     or only 30% sure (might need more options).

     Top-p picks {min(topp_counts)}-{max(topp_counts)} tokens depending on confidence.
     Confident → fewer tokens (exploit).
     Uncertain → more tokens (explore).

{BOLD}The RL connection:{RESET}

  {GREEN}The trained model = frozen policy network.{RESET}
  {GREEN}Decoding strategy = how you sample from the policy.{RESET}

  ┌──────────────┬──────────────────────────────────────┐
  │ Strategy     │ RL analogy                           │
  ├──────────────┼──────────────────────────────────────┤
  │ Greedy       │ Pure exploitation (argmax)           │
  │ Top-k        │ ε-greedy with fixed exploration      │
  │ Top-p        │ Adaptive exploration (entropy-aware) │
  │ Temperature  │ Entropy regularization of the policy │
  └──────────────┴──────────────────────────────────────┘

  RLHF changes the model WEIGHTS → shifts the distribution.
  Decoding changes the SAMPLING → selects from the distribution.
  Both control output quality. They are complementary tools.
""")
