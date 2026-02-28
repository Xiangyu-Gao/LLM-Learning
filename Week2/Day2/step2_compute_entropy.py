"""
Step 2: Compute Entropy — H(p) = -Σ pᵢ log pᵢ

Entropy measures how "spread out" a probability distribution is.
  - Low entropy  → model is confident (peaked distribution)
  - High entropy → model is uncertain (flat distribution)

Key observation:
  - Entropy increases smoothly with temperature.
  - Correctness does NOT increase monotonically.
  There's a sweet spot where temperature helps, then it hurts.
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
# Train model (same as step1 — 1-layer, 100 steps, undertrained)
# ===================================================================
print(f"""
{'='*65}
{BOLD}ENTROPY AND TEMPERATURE{RESET}
{'='*65}

  H(p) = -Σ pᵢ log pᵢ

  Entropy = expected surprise. How many bits of information
  does each sample from the distribution carry?
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
# 1. Entropy basics
# ===================================================================
print(f"{BOLD}1. WHAT IS ENTROPY?{RESET}")
print(f"{'='*65}")
print()


def compute_entropy(probs):
    """H(p) = -Σ pᵢ log pᵢ  (in nats, using natural log)."""
    # Avoid log(0) by adding tiny epsilon
    return -(probs * torch.log(probs + 1e-10)).sum(dim=-1)


# Extreme distributions
print(f"  Examples of entropy for different distributions:")
print()

examples = [
    ("Perfectly certain", torch.tensor([1.0] + [0.0] * 37)),
    ("Very confident",    F.softmax(torch.tensor([10.0] + [0.0] * 37), dim=0)),
    ("Somewhat confident", F.softmax(torch.tensor([3.0, 2.0, 1.0] + [0.0] * 35), dim=0)),
    ("Spread out",        F.softmax(torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6] + [0.0] * 33), dim=0)),
    ("Nearly uniform",    F.softmax(torch.ones(38) * 0.1, dim=0)),
    ("Perfectly uniform",  torch.ones(38) / 38),
]

max_entropy = math.log(vocab_size)
print(f"  {'Distribution':<22} {'Entropy':>8} {'% of max':>9} {'Effective choices':>18}")
print(f"  {'-'*22} {'-'*8} {'-'*9} {'-'*18}")

for name, dist in examples:
    ent = compute_entropy(dist).item()
    pct = 100 * ent / max_entropy
    # Perplexity = e^H = effective number of choices
    perplexity = math.exp(ent)
    print(f"  {name:<22} {ent:>8.3f} {pct:>8.1f}% {perplexity:>17.1f}")

print()
print(f"  Max entropy = log({vocab_size}) = {max_entropy:.3f} nats")
print(f"  Perplexity = e^H = effective number of equally-likely choices")
print(f"  {CYAN}Entropy 0 = 1 choice (certain). Entropy max = {vocab_size} choices (uniform).{RESET}")
print()

# ===================================================================
# 2. Temperature vs entropy (the smooth curve)
# ===================================================================
print(f"{BOLD}2. ENTROPY vs TEMPERATURE{RESET}")
print(f"{'='*65}")
print()

# Collect logits over many positions
n_samples = 100
all_logits = []

with torch.no_grad():
    for _ in range(n_samples):
        start = torch.randint(0, len(data) - CONTEXT_LEN - 1, (1,)).item()
        sample = data[start : start + CONTEXT_LEN].unsqueeze(0)
        logits = model(sample)  # (1, C, V)
        all_logits.append(logits[0])  # (C, V)

all_logits = torch.cat(all_logits, dim=0)  # (n_samples*C, V)
n_positions = all_logits.shape[0]

temperatures = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 3.0, 5.0, 10.0]

avg_entropies = []
med_entropies = []

print(f"  {'Temp':>6} {'Avg Entropy':>12} {'% of max':>9} {'Perplexity':>11}")
print(f"  {'-'*6} {'-'*12} {'-'*9} {'-'*11}")

for T in temperatures:
    probs = F.softmax(all_logits / T, dim=-1)
    ents = compute_entropy(probs)  # (n_positions,)
    avg_ent = ents.mean().item()
    med_ent = ents.median().item()
    avg_entropies.append(avg_ent)
    med_entropies.append(med_ent)
    pct = 100 * avg_ent / max_entropy
    ppx = math.exp(avg_ent)
    print(f"  {T:>6.1f} {avg_ent:>12.4f} {pct:>8.1f}% {ppx:>10.1f}")

print()
print(f"  {GREEN}Observation:{RESET} Entropy increases smoothly and monotonically with T.")
print(f"  T=0.1 → almost greedy (entropy ≈ 0)")
print(f"  T=10  → almost uniform (entropy ≈ max)")
print(f"  This is a smooth, predictable curve.")
print()

# ===================================================================
# 3. Correctness vs temperature (the non-monotonic curve)
# ===================================================================
print(f"{BOLD}3. CORRECTNESS vs TEMPERATURE{RESET}")
print(f"{'='*65}")
print()
print(f"  Now: does higher entropy = better predictions?")
print(f"  Measure: cross-entropy loss and top-1 accuracy at each temperature.")
print()

# Prepare evaluation data
n_eval = 200
eval_logits = []
eval_targets = []

with torch.no_grad():
    for _ in range(n_eval):
        start = torch.randint(0, len(data) - CONTEXT_LEN - 1, (1,)).item()
        x = data[start : start + CONTEXT_LEN].unsqueeze(0)
        y = data[start + 1 : start + CONTEXT_LEN + 1]
        logits = model(x)  # (1, C, V)
        eval_logits.append(logits[0])
        eval_targets.append(y)

eval_logits = torch.cat(eval_logits, dim=0)   # (n_eval*C, V)
eval_targets = torch.cat(eval_targets, dim=0)  # (n_eval*C,)

temps_fine = np.arange(0.1, 5.01, 0.1)
ce_losses = []
top1_accs = []
top3_accs = []
entropies_fine = []

for T in temps_fine:
    scaled_logits = eval_logits / T
    probs = F.softmax(scaled_logits, dim=-1)

    # Cross-entropy loss
    ce = F.cross_entropy(scaled_logits, eval_targets).item()
    ce_losses.append(ce)

    # Top-1 accuracy (greedy at this temperature)
    preds = scaled_logits.argmax(dim=-1)
    acc1 = (preds == eval_targets).float().mean().item()
    top1_accs.append(acc1)

    # Top-3 accuracy
    top3 = scaled_logits.topk(3, dim=-1).indices
    acc3 = (top3 == eval_targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
    top3_accs.append(acc3)

    # Average entropy
    ent = compute_entropy(probs).mean().item()
    entropies_fine.append(ent)

# Find optimal temperature
best_ce_idx = np.argmin(ce_losses)
best_ce_temp = temps_fine[best_ce_idx]
best_acc_idx = np.argmax(top1_accs)
best_acc_temp = temps_fine[best_acc_idx]

print(f"  {'Temp':>6} {'CE Loss':>8} {'Top-1 Acc':>10} {'Top-3 Acc':>10} {'Entropy':>8}")
print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")

display_temps = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
for T in display_temps:
    idx = int(round((T - 0.1) / 0.1))
    if idx >= len(ce_losses):
        idx = len(ce_losses) - 1
    marker = ""
    if abs(T - best_ce_temp) < 0.05:
        marker = f"  ← {GREEN}best CE{RESET}"
    elif abs(T - best_acc_temp) < 0.05:
        marker = f"  ← {GREEN}best acc{RESET}"
    print(f"  {T:>6.1f} {ce_losses[idx]:>8.4f} {top1_accs[idx]:>9.1%} {top3_accs[idx]:>9.1%} {entropies_fine[idx]:>8.3f}{marker}")

print()
print(f"  {BOLD}Best cross-entropy loss at T = {best_ce_temp:.1f}{RESET}")
print(f"  {BOLD}Best top-1 accuracy at T = {best_acc_temp:.1f}{RESET}")
print()
print(f"  {RED}Key observation:{RESET}")
print(f"    Entropy increases SMOOTHLY with temperature.")
print(f"    But correctness does NOT increase monotonically!")
print()
print(f"    T too low → distribution too peaked → misses correct alternatives")
print(f"    T ≈ 1.0   → original model calibration → often best")
print(f"    T too high → distribution too flat → dilutes correct answer")
print()

# ===================================================================
# 4. Per-position entropy analysis
# ===================================================================
print(f"{BOLD}4. ENTROPY VARIES BY POSITION{RESET}")
print(f"{'='*65}")
print()

probe = "To be, or not to be, that is"
probe_tokens = encode(probe)
x = torch.tensor([probe_tokens], dtype=torch.long)
targets = encode(probe[1:] + " ")  # shifted by 1

with torch.no_grad():
    logits = model(x)[0]  # (C, V)

print(f"  Probe: \"{probe}\"")
print()
print(f"  {'Pos':<5} {'In':<5} {'→True':<7} {'→Pred':<7} {'Entropy':>8} {'Correct':>8} {'Top-1 prob':>10}")
print(f"  {'-'*5} {'-'*5} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*10}")

pos_entropies = []
for pos in range(len(probe) - 1):
    p = F.softmax(logits[pos], dim=0)
    ent = compute_entropy(p).item()
    pos_entropies.append(ent)

    pred_idx = p.argmax().item()
    true_idx = targets[pos]
    pred_char = idx_to_char[pred_idx]
    true_char = probe[pos + 1]
    correct = "✓" if pred_idx == true_idx else "✗"
    p_top1 = p.max().item()

    # Color coding
    if ent < 0.5:
        ent_color = GREEN
    elif ent < 1.5:
        ent_color = YELLOW
    else:
        ent_color = RED

    print(f"  {pos:<5} '{probe[pos]}'   '{true_char}'     '{pred_char}'     {ent_color}{ent:>7.3f}{RESET} {correct:>8} {p_top1:>9.1%}")

print()

# High vs low entropy positions
high_ent_pos = [i for i, e in enumerate(pos_entropies) if e > 1.5]
low_ent_pos = [i for i, e in enumerate(pos_entropies) if e < 0.5]

if high_ent_pos:
    high_chars = [f"'{probe[i]}'" for i in high_ent_pos]
    print(f"  {RED}High entropy positions:{RESET} {', '.join(high_chars)}")
    print(f"    These are where the model is UNCERTAIN.")
    print(f"    Sampling here creates diversity; greedy might be wrong.")
    print()

if low_ent_pos:
    low_chars = [f"'{probe[i]}'" for i in low_ent_pos]
    print(f"  {GREEN}Low entropy positions:{RESET} {', '.join(low_chars)}")
    print(f"    These are where the model is CONFIDENT.")
    print(f"    Greedy is nearly always correct here.")
    print()

# ===================================================================
# 5. Entropy at each temperature for a specific position
# ===================================================================
print(f"{BOLD}5. TEMPERATURE EFFECT ON A SINGLE POSITION{RESET}")
print(f"{'='*65}")
print()

# Pick a position where the model is uncertain
uncertain_pos = max(range(len(pos_entropies)), key=lambda i: pos_entropies[i])
certain_pos = min(range(len(pos_entropies)), key=lambda i: pos_entropies[i])

for pos, label in [(certain_pos, "CONFIDENT"), (uncertain_pos, "UNCERTAIN")]:
    pos_logits = logits[pos]
    true_char = probe[pos + 1]
    true_idx = targets[pos]
    print(f"  Position {pos} ('{probe[pos]}' → '{true_char}') — {label}")
    print(f"  {'Temp':>6} {'Entropy':>8} {'P(correct)':>11} {'Top-1':>8} {'Top-3 chars':<20}")
    print(f"  {'-'*6} {'-'*8} {'-'*11} {'-'*8} {'-'*20}")

    for T in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        p = F.softmax(pos_logits / T, dim=0)
        ent = compute_entropy(p).item()
        p_correct = p[true_idx].item()
        top3_idx = p.topk(3).indices.tolist()
        top3_str = ", ".join(f"'{idx_to_char[i]}'({p[i].item():.2f})" for i in top3_idx)
        print(f"  {T:>6.1f} {ent:>8.3f} {p_correct:>10.4f} {p.max().item():>7.1%} {top3_str}")

    print()

# ===================================================================
# Visualization
# ===================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Entropy and Temperature: Smooth Curve, Non-Monotonic Correctness",
             fontsize=14, fontweight='bold')

# Panel 1: Entropy vs Temperature
ax = axes[0, 0]
ax.plot(temps_fine, entropies_fine, 'b-', linewidth=2.5, label='Avg entropy')
ax.axhline(y=max_entropy, color='red', linestyle='--', alpha=0.5,
           label=f'Max entropy ({max_entropy:.2f})')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel("Temperature", fontsize=12)
ax.set_ylabel("Entropy (nats)", fontsize=12)
ax.set_title("Entropy vs Temperature\n(smooth, monotonically increasing)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: Correctness vs Temperature
ax = axes[0, 1]
ax.plot(temps_fine, ce_losses, 'r-', linewidth=2.5, label='Cross-entropy loss')
ax.axvline(x=best_ce_temp, color='green', linestyle='--', alpha=0.7,
           label=f'Best T={best_ce_temp:.1f}')
ax.set_xlabel("Temperature", fontsize=12)
ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
ax.set_title("Loss vs Temperature\n(NOT monotonic — there's a sweet spot)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: Accuracy vs Temperature
ax = axes[1, 0]
ax.plot(temps_fine, top1_accs, 'o-', color='steelblue', linewidth=2,
        markersize=2, label='Top-1 accuracy')
ax.plot(temps_fine, top3_accs, 's-', color='darkorange', linewidth=2,
        markersize=2, label='Top-3 accuracy')
ax.axvline(x=best_acc_temp, color='green', linestyle='--', alpha=0.7,
           label=f'Best acc T={best_acc_temp:.1f}')
ax.set_xlabel("Temperature", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Accuracy vs Temperature")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 4: Entropy vs Correctness (the decoupling)
ax = axes[1, 1]
scatter = ax.scatter(entropies_fine, ce_losses, c=temps_fine, cmap='coolwarm',
                     s=30, alpha=0.8, edgecolors='black', linewidth=0.3)
ax.set_xlabel("Average Entropy", fontsize=12)
ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
ax.set_title("Entropy vs Loss\n(higher entropy ≠ better predictions)")
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Temperature")
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), 'entropy_vs_temperature.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved visualization → {save_path}")
print()

# ===================================================================
# Summary
# ===================================================================
print(f"""
{'='*65}
{BOLD}STEP 2 SUMMARY: ENTROPY AND TEMPERATURE{RESET}
{'='*65}

{BOLD}Entropy:{RESET}

  H(p) = -Σ pᵢ log pᵢ

  Measures the "spread" of a distribution:
    H = 0       → one token has probability 1 (certain)
    H = log(V)  → all tokens equally likely (uniform)

  Perplexity = e^H = effective number of choices.
  A model with perplexity 10 is "choosing between 10 options."

{BOLD}Temperature scaling:{RESET}

  probs = softmax(logits / T)

  T controls entropy:
    T → 0:   entropy → 0 (greedy)
    T = 1:   entropy = model's learned calibration
    T → ∞:   entropy → log(V) (uniform)

  {GREEN}Entropy increases SMOOTHLY with temperature.{RESET}
  This is mathematically guaranteed — softmax with higher T
  always produces a flatter (higher entropy) distribution.

{BOLD}The key insight: correctness is NOT monotonic:{RESET}

  {RED}More entropy does NOT mean better predictions.{RESET}

  Best CE loss at T = {best_ce_temp:.1f}
  Best top-1 acc at T = {best_acc_temp:.1f}

  Why?
    T too low  → over-confident. Commits to one answer even
                 when the model should hedge between options.
    T ≈ 1      → the model's own learned calibration.
    T too high → under-confident. Spreads probability to
                 tokens that make no sense, diluting the
                 correct answer's probability.

{BOLD}Practical implication:{RESET}

  {CYAN}Temperature is not "creativity knob" — it's a calibration tool.{RESET}

  For each task there's an optimal temperature:
    - Factual Q&A:    T ≈ 0.0-0.3 (be confident)
    - Code generation: T ≈ 0.2-0.5 (mostly greedy, some variety)
    - Creative writing: T ≈ 0.7-1.0 (allow exploration)
    - Brainstorming:   T ≈ 1.0-1.5 (maximize diversity)

  Going above T ≈ 2.0 almost always hurts — it doesn't make
  the model "more creative," it makes it randomly wrong.
""")
