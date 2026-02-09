"""
Step 3: Head ablation experiment — which heads actually matter?

Zero out one head at a time by nullifying its columns in W_o,
then measure loss change and generation quality.

Key insight: W_o maps each head's output to the final embedding.
Columns h*D:(h+1)*D of W_o carry head h's contribution.
Zeroing those columns = removing that head entirely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Day1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Day2'))
sys.path.insert(0, os.path.dirname(__file__))

from train_tiny_char_level_lm import vocab_size, encode, decode, data, CONTEXT_LEN
from step1_scaled_dot_product_attention import make_causal_mask
from step1_multihead_attention import MultiHeadAttention

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Multi-head attention LM (same as step2, redefined to avoid triggering
# step2's module-level training code on import)
# ---------------------------------------------------------------------------
class MultiHeadAttentionLM(nn.Module):
    def __init__(self, vocab_size, context_len, embed_dim, num_heads):
        super().__init__()
        self.context_len = context_len
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(context_len, embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.out_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, C = x.shape
        emb = self.embed(x) + self.pos_embed(torch.arange(C, device=x.device))
        mask = make_causal_mask(C).to(x.device)
        attn_out, self._attn_weights = self.attn(emb, mask)
        return self.out_proj(attn_out)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
EMBED_DIM = 64
NUM_HEADS = 4
HEAD_DIM = EMBED_DIM // NUM_HEADS
STEPS = 800

print(f"{BOLD}Training multi-head attention LM ({NUM_HEADS} heads)...{RESET}")
torch.manual_seed(42)
model = MultiHeadAttentionLM(vocab_size, CONTEXT_LEN, EMBED_DIM, NUM_HEADS)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

for step in range(1, STEPS + 1):
    ix = torch.randint(0, len(data) - CONTEXT_LEN - 1, (128,))
    x = torch.stack([data[i : i + CONTEXT_LEN] for i in ix])
    y = torch.stack([data[i + 1 : i + CONTEXT_LEN + 1] for i in ix])
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 200 == 0:
        print(f"  Step {step:>4d} | Loss: {loss.item():.4f}")

print()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_loss(model, num_batches=10):
    """Average loss over multiple batches for stable measurement."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            ix = torch.randint(0, len(data) - CONTEXT_LEN - 1, (128,))
            x = torch.stack([data[i : i + CONTEXT_LEN] for i in ix])
            y = torch.stack([data[i + 1 : i + CONTEXT_LEN + 1] for i in ix])
            logits = model(x)
            total_loss += F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
    return total_loss / num_batches

def generate(model, seed_text, length=100):
    """Generate text from a seed."""
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
# Baseline measurements
# ---------------------------------------------------------------------------
torch.manual_seed(99)  # fixed seed for reproducible generation
baseline_loss = compute_loss(model)
seed_text = "To be, or not to be"
baseline_gen = generate(model, seed_text)

print(f"{'='*65}")
print(f"{BOLD}BASELINE (all heads active){RESET}")
print(f"{'='*65}")
print(f"  Loss: {baseline_loss:.4f}")
print(f"  Generation: \"{baseline_gen[:80]}\"")
print()

# ---------------------------------------------------------------------------
# Ablation: zero out one head at a time via W_o columns
# ---------------------------------------------------------------------------
# W_o.weight shape: (E, E) = (E, H*D)
# Head h's contribution flows through columns h*D:(h+1)*D
# Zeroing those columns removes head h from the output.

print(f"{'='*65}")
print(f"{BOLD}HEAD ABLATION: Zeroing out one head at a time{RESET}")
print(f"{'='*65}")
print(f"  Method: zero out columns [h*D : (h+1)*D] of W_o")
print(f"  This removes head h's contribution to the final output.")
print()

W_o = model.attn.W_o  # the output projection
original_weight = W_o.weight.data.clone()  # save original

ablation_results = []

for h in range(NUM_HEADS):
    # Zero out head h's columns in W_o
    col_start = h * HEAD_DIM
    col_end = (h + 1) * HEAD_DIM
    W_o.weight.data[:, col_start:col_end] = 0.0

    # Measure loss
    torch.manual_seed(99)
    ablated_loss = compute_loss(model)
    loss_delta = ablated_loss - baseline_loss
    loss_pct = loss_delta / baseline_loss * 100

    # Generate
    torch.manual_seed(99)
    ablated_gen = generate(model, seed_text)

    ablation_results.append({
        'head': h,
        'loss': ablated_loss,
        'loss_delta': loss_delta,
        'loss_pct': loss_pct,
        'generation': ablated_gen,
    })

    # Color based on impact
    if loss_pct > 20:
        color = RED
        impact = "HIGH IMPACT"
    elif loss_pct > 5:
        color = YELLOW
        impact = "MODERATE"
    else:
        color = GREEN
        impact = "LOW IMPACT"

    print(f"  {BOLD}Head {h} removed:{RESET}  {color}{impact}{RESET}")
    print(f"    Loss: {ablated_loss:.4f}  (Δ = +{loss_delta:.4f}, +{loss_pct:.1f}%)")
    print(f"    Generation: \"{ablated_gen[:80]}\"")
    print()

    # Restore W_o
    W_o.weight.data.copy_(original_weight)

# ---------------------------------------------------------------------------
# Rank heads by importance
# ---------------------------------------------------------------------------
print(f"{'='*65}")
print(f"{BOLD}HEAD IMPORTANCE RANKING{RESET}")
print(f"{'='*65}")

ranked = sorted(ablation_results, key=lambda r: r['loss_delta'], reverse=True)
print(f"  {'Rank':<6} {'Head':<6} {'Loss Δ':<10} {'Loss %':<10} {'Verdict'}")
print(f"  {'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*20}")

for rank, r in enumerate(ranked):
    if r['loss_pct'] > 20:
        verdict = f"{RED}Critical — big loss spike{RESET}"
    elif r['loss_pct'] > 5:
        verdict = f"{YELLOW}Important — noticeable{RESET}"
    else:
        verdict = f"{GREEN}Redundant — safely removable{RESET}"
    print(f"  {rank+1:<6} {r['head']:<6} +{r['loss_delta']:<9.4f} +{r['loss_pct']:<9.1f}% {verdict}")

print()

# ---------------------------------------------------------------------------
# Ablate ALL heads one by one and sum deltas vs ablate-all
# ---------------------------------------------------------------------------
print(f"{'='*65}")
print(f"{BOLD}REDUNDANCY TEST: Sum of individual Δs vs removing ALL heads{RESET}")
print(f"{'='*65}")

sum_individual = sum(r['loss_delta'] for r in ablation_results)

# Zero out ALL of W_o (= removing all heads)
W_o.weight.data.zero_()
torch.manual_seed(99)
all_ablated_loss = compute_loss(model)
all_delta = all_ablated_loss - baseline_loss
W_o.weight.data.copy_(original_weight)

print(f"  Sum of individual head Δs:  +{sum_individual:.4f}")
print(f"  Actual all-heads-removed Δ: +{all_delta:.4f}")

if all_delta > sum_individual * 1.2:
    print(f"  {RED}All-removed >> sum of individuals → heads COMPENSATE for each other{RESET}")
    print(f"  Removing one head is tolerable because others pick up the slack.")
elif all_delta < sum_individual * 0.8:
    print(f"  {YELLOW}All-removed << sum of individuals → heads are REDUNDANT{RESET}")
    print(f"  Individual losses overlap — heads duplicate each other's work.")
else:
    print(f"  {GREEN}Roughly additive → heads contribute INDEPENDENTLY{RESET}")
    print(f"  Each head covers a different aspect, minimal overlap.")

print()

# ---------------------------------------------------------------------------
# Pairwise ablation (remove two heads at once)
# ---------------------------------------------------------------------------
print(f"{'='*65}")
print(f"{BOLD}PAIRWISE ABLATION: Which pair hurts most?{RESET}")
print(f"{'='*65}")

pair_results = []
for h1 in range(NUM_HEADS):
    for h2 in range(h1 + 1, NUM_HEADS):
        # Zero out both heads
        W_o.weight.data[:, h1*HEAD_DIM:(h1+1)*HEAD_DIM] = 0.0
        W_o.weight.data[:, h2*HEAD_DIM:(h2+1)*HEAD_DIM] = 0.0

        torch.manual_seed(99)
        pair_loss = compute_loss(model)
        pair_delta = pair_loss - baseline_loss

        # Expected delta if independent
        expected = ablation_results[h1]['loss_delta'] + ablation_results[h2]['loss_delta']
        interaction = pair_delta - expected

        pair_results.append((h1, h2, pair_delta, expected, interaction))

        # Restore
        W_o.weight.data.copy_(original_weight)

print(f"  {'Pair':<10} {'Actual Δ':<12} {'Expected Δ':<12} {'Interaction':<12} {'Meaning'}")
print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*25}")
for h1, h2, actual, expected, interaction in pair_results:
    if interaction > 0.02:
        meaning = f"{RED}synergy (compensate){RESET}"
    elif interaction < -0.02:
        meaning = f"{YELLOW}redundancy (overlap){RESET}"
    else:
        meaning = f"{GREEN}independent{RESET}"
    print(f"  ({h1}, {h2})     +{actual:<11.4f} +{expected:<11.4f} {interaction:+.4f}      {meaning}")

print()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
most_important = ranked[0]['head']
least_important = ranked[-1]['head']

print(f"""
{'='*65}
{BOLD}STEP 3 SUMMARY: HEAD ABLATION{RESET}
{'='*65}

{BOLD}What we found:{RESET}
  - Most important: Head {most_important} (removing it causes +{ranked[0]['loss_pct']:.1f}% loss)
  - Least important: Head {least_important} (removing it causes +{ranked[-1]['loss_pct']:.1f}% loss)
  - Not all heads are equal — some carry more of the model's capability.

{BOLD}Redundancy:{RESET}
  Heads are partially redundant. Removing any single head doesn't
  catastrophically break the model (loss increases, but generation
  still works). This is because:
  - W_o learns to combine head outputs, creating some overlap
  - Other heads can partially compensate for a missing head
  - But the compensation isn't perfect — loss still goes up

{BOLD}Why this matters for real models:{RESET}
  1. {GREEN}Model compression{RESET}: "unimportant" heads can be pruned
     to make models smaller/faster with minimal quality loss.

  2. {GREEN}Interpretability{RESET}: the most important heads often
     correspond to the most meaningful linguistic patterns
     (syntax, coreference, positional tracking).

  3. {GREEN}Training dynamics{RESET}: if heads are too redundant,
     capacity is wasted. Techniques like head dropout or
     auxiliary diversity losses can push heads to specialize.

{BOLD}The key insight:{RESET}
  Multi-head attention provides redundancy AND specialization.
  Not complete collapse (heads aren't identical) and not
  complete independence (removing one doesn't zero out performance).
  The sweet spot is in between — and that's by design.
""")
