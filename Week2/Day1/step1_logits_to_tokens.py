"""
Step 1: Logits → Tokens — Decoding as a Policy.

The model does NOT output words.
It outputs a probability distribution over vocabulary.
Decoding = choosing an action from that distribution.

Three strategies:
  1. Greedy    — always pick the most probable token
  2. Top-k     — sample from the k most probable tokens
  3. Top-p     — sample from the smallest set covering probability p
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
# Train a model (reuse Week1 architecture)
# ===================================================================
print(f"""
{'='*65}
{BOLD}LOGITS → TOKENS: DECODING AS A POLICY{RESET}
{'='*65}

  The model outputs logits z ∈ ℝ^V (one score per vocab token).
  probs = softmax(z) gives a probability distribution.
  Decoding = choosing which token to emit from that distribution.

  This is a POLICY DECISION, not a deterministic function.
  Different strategies produce wildly different text.
""")

print(f"{BOLD}Training 1-layer model (100 steps — intentionally undertrained)...{RESET}")
print(f"  (Undertrained so the model is uncertain → decoding differences visible)")
print()
TRAIN_LAYERS = 1
TRAIN_STEPS = 100
torch.manual_seed(42)
model = TransformerLM(vocab_size, CONTEXT_LEN, EMBED_DIM, NUM_HEADS, FF_DIM, TRAIN_LAYERS)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

for step in range(1, TRAIN_STEPS + 1):
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
# Part 0: What do logits look like?
# ===================================================================
print(f"{BOLD}0. WHAT ARE LOGITS?{RESET}")
print(f"{'='*65}")
print()

seed = "To be, or not to be"
tokens = encode(seed)
x = torch.tensor([tokens], dtype=torch.long)

with torch.no_grad():
    logits = model(x)  # (1, C, V)

# Look at the last position's logits (predicting next char after "be")
last_logits = logits[0, -1]  # (V,)
probs = F.softmax(last_logits, dim=0)

print(f"  Input: \"{seed}\"")
print(f"  Predicting next character after '{seed[-1]}'")
print()
print(f"  Raw logits z ∈ ℝ^{vocab_size}:")
print(f"    min={last_logits.min():.2f}, max={last_logits.max():.2f}, mean={last_logits.mean():.2f}")
print()

# Show top 10 predictions
sorted_probs, sorted_idx = probs.sort(descending=True)
print(f"  After softmax → probability distribution:")
print(f"  {'Rank':<6} {'Char':<8} {'Logit':>8} {'Prob':>8} {'Cumulative':>10}")
print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

cumulative = 0.0
for rank in range(min(15, vocab_size)):
    idx = sorted_idx[rank].item()
    char = repr(idx_to_char[idx])
    logit = last_logits[idx].item()
    prob = sorted_probs[rank].item()
    cumulative += prob
    bar = "█" * int(40 * prob)
    print(f"  {rank+1:<6} {char:<8} {logit:>8.2f} {prob:>8.4f} {cumulative:>9.4f}  {bar}")

print(f"  ... ({vocab_size - 15} more tokens with tiny probabilities)")
print()
print(f"  {GREEN}The model is ~{sorted_probs[0].item():.0%} confident the next char is '{idx_to_char[sorted_idx[0].item()]}'.{RESET}")
print(f"  But {sorted_probs[1].item():.0%} chance of '{idx_to_char[sorted_idx[1].item()]}', {sorted_probs[2].item():.0%} of '{idx_to_char[sorted_idx[2].item()]}', etc.")
print(f"  {CYAN}Decoding decides which of these possibilities to follow.{RESET}")
print()

# ===================================================================
# Part 1: Implement decoders
# ===================================================================

def greedy_decode(logits):
    """Always pick the highest-probability token."""
    return torch.argmax(logits, dim=-1)


def topk_decode(logits, k=5, temperature=1.0):
    """Sample from the top-k most probable tokens."""
    logits = logits / temperature
    topk_vals, topk_idx = torch.topk(logits, k)
    topk_probs = F.softmax(topk_vals, dim=-1)
    sampled = torch.multinomial(topk_probs, 1)
    return topk_idx[sampled].squeeze(-1)


def topp_decode(logits, p=0.9, temperature=1.0):
    """Sample from the smallest set of tokens whose cumulative probability ≥ p."""
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    # Sort by probability (descending)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)

    # Cumulative sum
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    # Find where cumulative sum first exceeds p
    # Keep all tokens up to and including the one that crosses p
    mask = cumsum - sorted_probs < p  # shift: include the crossing token
    sorted_probs[~mask] = 0.0

    # Renormalize
    sorted_probs = sorted_probs / sorted_probs.sum()

    # Sample
    sampled = torch.multinomial(sorted_probs, 1)
    return sorted_idx[sampled].squeeze(-1)


# ===================================================================
# Part 1.5: Show how each decoder works step by step
# ===================================================================
print(f"{BOLD}1. GREEDY DECODING{RESET}")
print(f"{'='*65}")
print()
print(f"  Rule: token = argmax(probs)")
print(f"  Always pick the single most likely token.")
print()

chosen = greedy_decode(last_logits)
print(f"  For \"{seed}\" → next char: '{idx_to_char[chosen.item()]}'")
print(f"  (probability: {probs[chosen].item():.4f})")
print()
print(f"  {YELLOW}Problem:{RESET} Always the same output. No diversity.")
print(f"  Every run produces identical text. Boring and repetitive.")
print()

print(f"{BOLD}2. TOP-K DECODING{RESET}")
print(f"{'='*65}")
print()
print(f"  Rule: keep only the k most probable tokens, sample from them.")
print()

for k in [1, 3, 5, 10]:
    print(f"  k={k}:")
    # Show which tokens are kept
    topk_vals, topk_idx = torch.topk(probs, k)
    kept = [(idx_to_char[topk_idx[i].item()], topk_vals[i].item()) for i in range(k)]
    kept_str = ", ".join(f"'{c}' ({p:.3f})" for c, p in kept)
    renorm = topk_vals / topk_vals.sum()
    renorm_str = ", ".join(f"{p:.3f}" for p in renorm.tolist())
    print(f"    Kept: {kept_str}")
    print(f"    Renormalized: {renorm_str}")
    # Sample 20 times to show distribution
    samples = []
    for _ in range(20):
        tok = topk_decode(last_logits, k=k)
        samples.append(idx_to_char[tok.item()])
    print(f"    20 samples: {''.join(samples)}")
    print()

print(f"  {YELLOW}Problem:{RESET} k is a fixed number regardless of distribution shape.")
print(f"  If the model is very confident, k=10 includes junk tokens.")
print(f"  If the model is uncertain, k=3 might cut off good options.")
print()

print(f"{BOLD}3. TOP-P (NUCLEUS) DECODING{RESET}")
print(f"{'='*65}")
print()
print(f"  Rule: keep the smallest set of tokens whose cumulative prob ≥ p.")
print(f"  Adaptive: keeps fewer tokens when confident, more when uncertain.")
print()

for p in [0.5, 0.8, 0.9, 0.95]:
    sorted_p, sorted_i = probs.sort(descending=True)
    cumsum = torch.cumsum(sorted_p, dim=0)
    mask = (cumsum - sorted_p) < p
    n_kept = mask.sum().item()

    kept_chars = [idx_to_char[sorted_i[j].item()] for j in range(n_kept)]
    kept_probs = [sorted_p[j].item() for j in range(n_kept)]
    kept_str = ", ".join(f"'{c}' ({pr:.3f})" for c, pr in zip(kept_chars[:6], kept_probs[:6]))
    if n_kept > 6:
        kept_str += f", ... ({n_kept - 6} more)"

    print(f"  p={p}:")
    print(f"    Tokens kept: {n_kept}")
    print(f"    {kept_str}")
    # Sample 20 times
    samples = []
    for _ in range(20):
        tok = topp_decode(last_logits, p=p)
        samples.append(idx_to_char[tok.item()])
    print(f"    20 samples: {''.join(samples)}")
    print()

print(f"  {GREEN}Advantage:{RESET} Adapts to the model's confidence automatically.")
print(f"  Confident → few tokens kept (like greedy).")
print(f"  Uncertain → many tokens kept (more diversity).")
print()

# ===================================================================
# Part 2: Generate full sequences with each strategy
# ===================================================================
print(f"{BOLD}4. FULL GENERATION COMPARISON{RESET}")
print(f"{'='*65}")
print()

def generate(model, seed_text, decode_fn, length=100):
    """Generate text using a specific decoding strategy."""
    tokens = encode(seed_text)
    generated = list(tokens)
    with torch.no_grad():
        for _ in range(length):
            inp = torch.tensor([generated[-CONTEXT_LEN:]], dtype=torch.long)
            logits = model(inp)
            next_logits = logits[0, -1]  # last position
            next_token = decode_fn(next_logits)
            generated.append(next_token.item())
    return decode(generated[len(tokens):])


seed_text = "To be, or not to be"

# Greedy — deterministic
print(f"  Seed: \"{seed_text}\"")
print()

print(f"  {BOLD}Greedy:{RESET}")
for trial in range(3):
    torch.manual_seed(trial)
    text = generate(model, seed_text, greedy_decode, length=80)
    print(f"    Run {trial+1}: \"{text}\"")
print(f"  → {DIM}All identical — greedy is deterministic.{RESET}")
print()

# Top-k with different k values
for k in [3, 10]:
    print(f"  {BOLD}Top-k (k={k}):{RESET}")
    for trial in range(3):
        torch.manual_seed(trial)
        text = generate(model, seed_text, lambda z: topk_decode(z, k=k), length=80)
        print(f"    Run {trial+1}: \"{text}\"")
    print()

# Top-p with different p values
for p in [0.5, 0.9]:
    print(f"  {BOLD}Top-p (p={p}):{RESET}")
    for trial in range(3):
        torch.manual_seed(trial)
        text = generate(model, seed_text, lambda z, _p=p: topp_decode(z, p=_p), length=80)
        print(f"    Run {trial+1}: \"{text}\"")
    print()

# ===================================================================
# Part 3: Temperature
# ===================================================================
print(f"{BOLD}5. TEMPERATURE SCALING{RESET}")
print(f"{'='*65}")
print()
print(f"  Before softmax, divide logits by temperature T:")
print(f"    probs = softmax(z / T)")
print()
print(f"  T < 1.0 → sharper distribution (more confident)")
print(f"  T = 1.0 → original distribution")
print(f"  T > 1.0 → flatter distribution (more random)")
print(f"  T → 0   → greedy (argmax)")
print(f"  T → ∞   → uniform random")
print()

# Show temperature effect on the distribution
temps = [0.3, 0.5, 1.0, 1.5, 2.0]
print(f"  Distribution after \"{seed}\" at different temperatures:")
print()
print(f"  {'Char':<6}", end="")
for t in temps:
    print(f"  {'T='+str(t):>8}", end="")
print()
print(f"  {'-'*6}", end="")
for _ in temps:
    print(f"  {'-'*8}", end="")
print()

for rank in range(8):
    idx = sorted_idx[rank].item()
    char = repr(idx_to_char[idx])
    print(f"  {char:<6}", end="")
    for t in temps:
        p = F.softmax(last_logits / t, dim=0)[idx].item()
        print(f"  {p:>8.4f}", end="")
    print()

print()

# Generate at different temperatures
print(f"  Generation with top-p (p=0.9) at different temperatures:")
print()
for t in [0.3, 0.7, 1.0, 1.5, 2.0]:
    torch.manual_seed(42)
    text = generate(
        model, seed_text,
        lambda z, _t=t: topp_decode(z, p=0.9, temperature=_t),
        length=80,
    )
    print(f"  T={t:<4} \"{text}\"")

print()

# ===================================================================
# Part 4: Entropy analysis — when to be greedy vs creative
# ===================================================================
print(f"{BOLD}6. ENTROPY: WHEN IS THE MODEL CONFIDENT?{RESET}")
print(f"{'='*65}")
print()

probe = "To be, or not to be, that is"
probe_tokens = encode(probe)
x = torch.tensor([probe_tokens], dtype=torch.long)

with torch.no_grad():
    logits = model(x)

# Compute entropy at each position
entropies = []
top1_probs = []
for pos in range(len(probe) - 1):
    p = F.softmax(logits[0, pos], dim=0)
    entropy = -(p * torch.log(p + 1e-10)).sum().item()
    entropies.append(entropy)
    top1_probs.append(p.max().item())

max_entropy = math.log(vocab_size)

print(f"  Probe: \"{probe}\"")
print(f"  Max possible entropy: {max_entropy:.2f} (uniform over {vocab_size} tokens)")
print()
print(f"  {'Pos':<5} {'Input':<7} {'→Next':<7} {'Entropy':>8} {'Top-1 %':>8} {'Confidence'}")
print(f"  {'-'*5} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*20}")

for pos in range(len(probe) - 1):
    char_in = probe[pos]
    char_next = probe[pos + 1]
    ent = entropies[pos]
    p1 = top1_probs[pos]
    conf_bar = "█" * int(20 * (1 - ent / max_entropy))
    print(f"  {pos:<5} '{char_in}'     '{char_next}'    {ent:>7.3f} {p1:>7.1%}  {conf_bar}")

print()
print(f"  {GREEN}Low entropy{RESET} → model is confident → greedy is fine")
print(f"  {RED}High entropy{RESET} → model is uncertain → sampling adds variety")
print()
print(f"  {CYAN}Smart decoding adapts: be greedy when sure, creative when unsure.{RESET}")
print()

# ===================================================================
# Visualization
# ===================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Logits → Tokens: Decoding Strategies", fontsize=14, fontweight='bold')

# Panel 1: Probability distribution with different temperatures
ax = axes[0, 0]
x_labels = [repr(idx_to_char[sorted_idx[i].item()])[1:-1] for i in range(12)]
x_pos = np.arange(12)
width = 0.15
for i, t in enumerate([0.3, 0.7, 1.0, 1.5, 2.0]):
    t_probs = F.softmax(last_logits / t, dim=0)
    bars = [t_probs[sorted_idx[j]].item() for j in range(12)]
    ax.bar(x_pos + i * width, bars, width, label=f'T={t}', alpha=0.8)
ax.set_xticks(x_pos + 2 * width)
ax.set_xticklabels(x_labels, fontsize=8)
ax.set_ylabel("Probability")
ax.set_title(f"Temperature Effect on Distribution\n(after \"{seed[-10:]}\")")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2, axis='y')

# Panel 2: Top-k vs Top-p token count across positions
ax = axes[0, 1]
probe_short = "To be, or not to be, that is t"
probe_tokens_short = encode(probe_short)
x_in = torch.tensor([probe_tokens_short], dtype=torch.long)
with torch.no_grad():
    logits_all = model(x_in)

topk_counts = {k: [] for k in [3, 5, 10]}
topp_counts = {p: [] for p in [0.5, 0.9, 0.95]}

for pos in range(len(probe_short) - 1):
    p = F.softmax(logits_all[0, pos], dim=0)
    for k in topk_counts:
        topk_counts[k].append(k)  # always k tokens
    for p_val in topp_counts:
        sorted_p, _ = p.sort(descending=True)
        cumsum = torch.cumsum(sorted_p, dim=0)
        n = ((cumsum - sorted_p) < p_val).sum().item()
        topp_counts[p_val].append(n)

positions = range(len(probe_short) - 1)
for p_val, counts in topp_counts.items():
    ax.plot(positions, counts, 'o-', label=f'top-p={p_val}', markersize=3, linewidth=1.5)
ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='top-k=3')
ax.axhline(y=10, color='gray', linestyle=':', alpha=0.5, label='top-k=10')
ax.set_xlabel("Position")
ax.set_ylabel("Tokens in Candidate Set")
ax.set_title("Top-p Adapts, Top-k Doesn't")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(0, len(probe_short) - 1, 4))

# Panel 3: Entropy per position
ax = axes[1, 0]
ax.bar(range(len(entropies)), entropies, color='steelblue', alpha=0.7)
ax.axhline(y=max_entropy, color='red', linestyle='--', alpha=0.5, label=f'Max entropy ({max_entropy:.1f})')
ax.set_xlabel("Position")
ax.set_ylabel("Entropy (nats)")
ax.set_title(f"Model Confidence Per Position\n\"{probe[:30]}...\"")
ax.set_xticks(range(len(entropies)))
ax.set_xticklabels(list(probe[:-1]), fontsize=6, rotation=0)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Panel 4: Top-1 probability per position
ax = axes[1, 1]
colors = ['#2ecc71' if p > 0.7 else '#e74c3c' if p < 0.3 else '#f39c12' for p in top1_probs]
ax.bar(range(len(top1_probs)), top1_probs, color=colors, alpha=0.8)
ax.axhline(y=1/vocab_size, color='gray', linestyle='--', alpha=0.5, label=f'Uniform ({1/vocab_size:.3f})')
ax.set_xlabel("Position")
ax.set_ylabel("Top-1 Probability")
ax.set_title("When to Be Greedy vs Creative")
ax.set_xticks(range(len(top1_probs)))
ax.set_xticklabels(list(probe[:-1]), fontsize=6, rotation=0)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), 'logits_to_tokens.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved visualization → {save_path}")
print()

# ===================================================================
# Summary
# ===================================================================
print(f"""
{'='*65}
{BOLD}STEP 1 SUMMARY: LOGITS → TOKENS{RESET}
{'='*65}

{BOLD}The core insight:{RESET}

  The model outputs a probability distribution, not a token.
  Decoding is the POLICY that turns distributions into text.
  Different policies → different text from the SAME model.

{BOLD}Three decoding strategies:{RESET}

  1. {CYAN}Greedy{RESET} — argmax(probs)
     Always pick the most likely token.
     Deterministic. Repetitive. Safe.
     Good for: factual answers, code completion.

  2. {CYAN}Top-k{RESET} — sample from the k most probable
     Fixed candidate set size.
     k=1 is greedy, k=V is pure sampling.
     Problem: k doesn't adapt to confidence.

  3. {CYAN}Top-p (nucleus){RESET} — sample from smallest set ≥ p
     Adaptive candidate set size.
     Confident → few candidates (like greedy).
     Uncertain → many candidates (more diversity).
     This is what ChatGPT/Claude use in practice.

{BOLD}Temperature T:{RESET}

  probs = softmax(logits / T)

  T < 1  → sharper  (more deterministic)
  T = 1  → original
  T > 1  → flatter  (more random)
  T → 0  → greedy
  T → ∞  → uniform

{BOLD}The analogy:{RESET}

  {GREEN}The model is a weather forecaster.
  It gives: "70% sun, 20% clouds, 10% rain."
  Decoding is: do you bring an umbrella?{RESET}

  Greedy: "It'll be sunny" (most likely, but sometimes wrong).
  Top-p:  "Probably sunny, maybe clouds" (hedges appropriately).
  High-T: "Could be anything!" (too much uncertainty).
""")
