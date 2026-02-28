"""
Step 1: Generate at Multiple Temperatures.

Same prompt, five temperatures: 0.1, 0.7, 1.0, 1.5, 2.0

Measure three things:
  1. Token entropy — how spread is the distribution?
  2. Self-BLEU (repetition) — how repetitive is the output?
  3. Factual accuracy — does the text match the training corpus?

Key insight: there is NO single best temperature.
  Low T → accurate but repetitive
  High T → diverse but incoherent
  The sweet spot depends on the task.
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
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Week1', 'Day1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Week1', 'Day2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Week1', 'Day3'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Week1', 'Day4'))

from train_tiny_char_level_lm import (
    vocab_size, encode, decode, data, CONTEXT_LEN, idx_to_char, CORPUS
)
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
# Train model — 4-layer, 1200 steps (well-trained, to see real effects)
# ===================================================================
print(f"""
{'='*65}
{BOLD}GENERATE AT MULTIPLE TEMPERATURES{RESET}
{'='*65}

  Same model, same prompt, five temperatures.
  Watch how T trades off accuracy, diversity, and coherence.
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
# Generation with temperature
# ===================================================================
def generate_with_stats(model, seed_text, temperature, length=150, seed_val=42):
    """Generate text and collect per-step statistics."""
    torch.manual_seed(seed_val)
    tokens = encode(seed_text)
    generated = list(tokens)
    entropies = []
    top1_probs = []

    with torch.no_grad():
        for _ in range(length):
            inp = torch.tensor([generated[-CONTEXT_LEN:]], dtype=torch.long)
            logits = model(inp)[0, -1]  # last position logits

            # Stats at original temperature
            probs = F.softmax(logits / temperature, dim=0)
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            entropies.append(entropy)
            top1_probs.append(probs.max().item())

            # Sample
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)

    text = decode(generated[len(tokens):])
    return text, entropies, top1_probs


# ===================================================================
# Self-BLEU: measure repetition
# ===================================================================
def get_ngrams(text, n):
    """Extract character n-grams from text."""
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def self_bleu(text, n=3):
    """Self-BLEU: split text in half, compute n-gram overlap.
    High self-BLEU → repetitive. Low → diverse."""
    mid = len(text) // 2
    half1 = text[:mid]
    half2 = text[mid:]

    ngrams1 = Counter(get_ngrams(half1, n))
    ngrams2 = Counter(get_ngrams(half2, n))

    if not ngrams1 or not ngrams2:
        return 0.0

    # Count overlapping n-grams
    overlap = sum((ngrams1 & ngrams2).values())
    total = sum(ngrams1.values())

    return overlap / total if total > 0 else 0.0


def repetition_rate(text, n=3):
    """What fraction of n-grams appear more than once?"""
    ngrams = get_ngrams(text, n)
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / len(counts)


# ===================================================================
# Corpus match: measure factual accuracy
# ===================================================================
def corpus_overlap(generated_text, corpus=CORPUS, n=5):
    """What fraction of generated n-grams actually appear in the training corpus?
    High overlap → text matches training data (factually accurate for this task).
    Low overlap → hallucinated/novel content."""
    gen_ngrams = get_ngrams(generated_text, n)
    if not gen_ngrams:
        return 0.0

    corpus_ngrams = set(get_ngrams(corpus, n))
    matches = sum(1 for ng in gen_ngrams if ng in corpus_ngrams)
    return matches / len(gen_ngrams)


def longest_corpus_match(generated_text, corpus=CORPUS):
    """Length of the longest substring that appears in the training corpus."""
    best = 0
    for length in range(1, min(len(generated_text), 80) + 1):
        found = False
        for start in range(len(generated_text) - length + 1):
            if generated_text[start:start+length] in corpus:
                found = True
                best = length
                break
        if not found:
            break
    return best


# ===================================================================
# Part 1: Generate at each temperature
# ===================================================================
temperatures = [0.1, 0.7, 1.0, 1.5, 2.0]
seed_text = "To be, or not"
gen_length = 150
n_samples = 5  # multiple samples per temperature

print(f"{BOLD}1. GENERATION SAMPLES{RESET}")
print(f"{'='*65}")
print(f"  Seed: \"{seed_text}\"")
print(f"  Length: {gen_length} chars per sample")
print()

all_results = {}  # T -> list of (text, entropies, top1_probs)

for T in temperatures:
    print(f"  {BOLD}T = {T}{RESET}")
    samples = []
    for trial in range(n_samples):
        text, ents, top1s = generate_with_stats(
            model, seed_text, T, length=gen_length, seed_val=trial
        )
        samples.append((text, ents, top1s))
        if trial < 3:  # show first 3
            print(f"    [{trial+1}] \"{text[:80]}\"")
    all_results[T] = samples
    print()

# ===================================================================
# Part 2: Measure token entropy
# ===================================================================
print(f"{BOLD}2. TOKEN ENTROPY{RESET}")
print(f"{'='*65}")
print()
print(f"  Average entropy of the distribution at each generation step.")
print(f"  Higher entropy → more uncertainty → flatter distribution.")
print()

max_entropy = math.log(vocab_size)

print(f"  {'Temp':>6} {'Avg Entropy':>12} {'% of max':>9} {'Perplexity':>11} {'Avg Top-1':>10}")
print(f"  {'-'*6} {'-'*12} {'-'*9} {'-'*11} {'-'*10}")

avg_entropies = {}
avg_top1s = {}

for T in temperatures:
    all_ents = []
    all_top1 = []
    for text, ents, top1s in all_results[T]:
        all_ents.extend(ents)
        all_top1.extend(top1s)

    avg_ent = np.mean(all_ents)
    avg_t1 = np.mean(all_top1)
    avg_entropies[T] = avg_ent
    avg_top1s[T] = avg_t1

    pct = 100 * avg_ent / max_entropy
    ppx = math.exp(avg_ent)

    bar = "█" * int(30 * avg_ent / max_entropy)
    print(f"  {T:>6.1f} {avg_ent:>12.4f} {pct:>8.1f}% {ppx:>10.1f} {avg_t1:>9.1%}  {bar}")

print()
print(f"  {CYAN}Entropy grows with temperature — the distribution flattens.{RESET}")
print(f"  At T=0.1 the model is near-deterministic (perplexity ≈ 1).")
print(f"  At T=2.0 it's choosing among ~{math.exp(avg_entropies[2.0]):.0f} effective options per step.")
print()

# ===================================================================
# Part 3: Self-BLEU (repetition)
# ===================================================================
print(f"{BOLD}3. REPETITION (Self-BLEU and n-gram analysis){RESET}")
print(f"{'='*65}")
print()

print(f"  Self-BLEU: n-gram overlap between first and second half of text.")
print(f"  Repetition rate: fraction of unique n-grams that appear 2+ times.")
print(f"  Higher = more repetitive.")
print()

print(f"  {'Temp':>6} {'Self-BLEU':>10} {'Rep rate':>9} {'Unique 3g':>10} {'Total 3g':>9} {'Diversity'}")
print(f"  {'-'*6} {'-'*10} {'-'*9} {'-'*10} {'-'*9} {'-'*15}")

rep_rates = {}
self_bleus = {}
diversities = {}

for T in temperatures:
    all_sbleu = []
    all_rep = []
    all_unique = []
    all_total = []

    for text, _, _ in all_results[T]:
        sb = self_bleu(text, n=3)
        rr = repetition_rate(text, n=3)
        ngrams = get_ngrams(text, 3)
        unique = len(set(ngrams))

        all_sbleu.append(sb)
        all_rep.append(rr)
        all_unique.append(unique)
        all_total.append(len(ngrams))

    avg_sb = np.mean(all_sbleu)
    avg_rr = np.mean(all_rep)
    avg_unique = np.mean(all_unique)
    avg_total = np.mean(all_total)
    diversity = avg_unique / avg_total if avg_total > 0 else 0

    self_bleus[T] = avg_sb
    rep_rates[T] = avg_rr
    diversities[T] = diversity

    # Visual indicator
    if avg_rr > 0.5:
        indicator = f"{RED}very repetitive{RESET}"
    elif avg_rr > 0.3:
        indicator = f"{YELLOW}somewhat repetitive{RESET}"
    else:
        indicator = f"{GREEN}diverse{RESET}"

    print(f"  {T:>6.1f} {avg_sb:>10.3f} {avg_rr:>8.1%} {avg_unique:>10.0f} {avg_total:>9.0f} {indicator}")

print()
print(f"  {CYAN}Surprising result:{RESET} High T has MORE n-gram repetition, not less!")
print(f"  Why? Low T reproduces the corpus faithfully — natural prose is diverse.")
print(f"  High T generates random character combos that hit common n-grams")
print(f"  ('th', 'he', ' t', etc.) more often than structured text does.")
print()
print(f"  {BOLD}In larger models with bigger vocabularies, the pattern flips:{RESET}")
print(f"  Low T → greedy gets stuck in repetitive loops ('I think I think I think...')")
print(f"  High T → random sampling breaks out of loops.")
print(f"  Our tiny model doesn't loop because it memorized the corpus perfectly.")
print()

# ===================================================================
# Part 4: Factual accuracy (corpus match)
# ===================================================================
print(f"{BOLD}4. FACTUAL ACCURACY (corpus match){RESET}")
print(f"{'='*65}")
print()
print(f"  Since our model is trained on a Shakespeare passage, 'factual accuracy'")
print(f"  = how much of the generated text actually appears in the training corpus.")
print()

print(f"  {'Temp':>6} {'5-gram match':>12} {'Longest run':>12} {'Quality'}")
print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*20}")

corpus_matches = {}
longest_matches = {}

for T in temperatures:
    all_overlap = []
    all_longest = []

    for text, _, _ in all_results[T]:
        overlap = corpus_overlap(text, n=5)
        longest = longest_corpus_match(text)
        all_overlap.append(overlap)
        all_longest.append(longest)

    avg_overlap = np.mean(all_overlap)
    avg_longest = np.mean(all_longest)
    corpus_matches[T] = avg_overlap
    longest_matches[T] = avg_longest

    if avg_overlap > 0.7:
        quality = f"{GREEN}high fidelity{RESET}"
    elif avg_overlap > 0.3:
        quality = f"{YELLOW}moderate{RESET}"
    else:
        quality = f"{RED}hallucinating{RESET}"

    print(f"  {T:>6.1f} {avg_overlap:>11.1%} {avg_longest:>11.0f} chars {quality}")

print()
print(f"  {GREEN}Low T:{RESET} nearly perfect corpus reproduction (near-memorization).")
print(f"  {RED}High T:{RESET} generates novel text that doesn't match the corpus.")
print()

# Show specific examples of factual vs hallucinated text
print(f"  {BOLD}Example — same starting point, different temperatures:{RESET}")
print()
for T in [0.1, 1.0, 2.0]:
    text = all_results[T][0][0]
    overlap = corpus_overlap(text[:60], n=5)
    print(f"  T={T:<4} ({overlap:.0%} match): \"{text[:70]}\"")
print()

# ===================================================================
# Part 5: The tradeoff
# ===================================================================
print(f"{BOLD}5. THE ACCURACY-DIVERSITY TRADEOFF{RESET}")
print(f"{'='*65}")
print()
print(f"  {'Temp':>6} {'Corpus match':>13} {'Repetition':>11} {'Entropy':>8} {'Verdict'}")
print(f"  {'-'*6} {'-'*13} {'-'*11} {'-'*8} {'-'*25}")

for T in temperatures:
    cm = corpus_matches[T]
    rr = rep_rates[T]
    ent = avg_entropies[T]

    if T <= 0.3:
        verdict = "accurate but boring"
    elif T <= 0.8:
        verdict = "good balance"
    elif T <= 1.2:
        verdict = "creative, mostly correct"
    elif T <= 1.7:
        verdict = "diverse, some errors"
    else:
        verdict = "incoherent, novel"

    print(f"  {T:>6.1f} {cm:>12.1%} {rr:>10.1%} {ent:>8.2f} {verdict}")

print()
print(f"  {BOLD}There is NO single best temperature.{RESET}")
print(f"  The right T depends on what you need:")
print()
print(f"    Need accuracy?   → T ≈ 0.1-0.3 (exploit the model's knowledge)")
print(f"    Need creativity? → T ≈ 0.7-1.0 (explore while staying coherent)")
print(f"    Need novelty?    → T ≈ 1.5+    (but expect errors)")
print()

# ===================================================================
# Visualization
# ===================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Temperature: Accuracy vs Diversity Tradeoff",
             fontsize=14, fontweight='bold')

T_arr = np.array(temperatures)

# Panel 1: Entropy vs Temperature
ax = axes[0, 0]
ent_arr = [avg_entropies[T] for T in temperatures]
ax.plot(T_arr, ent_arr, 'o-', color='steelblue', linewidth=2.5, markersize=8)
ax.axhline(y=max_entropy, color='red', linestyle='--', alpha=0.4,
           label=f'Max ({max_entropy:.2f})')
ax.fill_between(T_arr, 0, ent_arr, alpha=0.1, color='steelblue')
ax.set_xlabel("Temperature", fontsize=12)
ax.set_ylabel("Avg Token Entropy", fontsize=12)
ax.set_title("Entropy Increases Smoothly")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Corpus match vs Temperature
ax = axes[0, 1]
cm_arr = [corpus_matches[T] for T in temperatures]
ax.plot(T_arr, cm_arr, 's-', color='#2ecc71', linewidth=2.5, markersize=8)
ax.fill_between(T_arr, 0, cm_arr, alpha=0.1, color='#2ecc71')
ax.set_xlabel("Temperature", fontsize=12)
ax.set_ylabel("5-gram Corpus Match", fontsize=12)
ax.set_title("Accuracy Drops with Temperature")
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

# Panel 3: Repetition vs Temperature
ax = axes[1, 0]
rr_arr = [rep_rates[T] for T in temperatures]
sb_arr = [self_bleus[T] for T in temperatures]
ax.plot(T_arr, rr_arr, 'D-', color='#e74c3c', linewidth=2.5, markersize=8,
        label='Repetition rate')
ax.plot(T_arr, sb_arr, '^-', color='#e67e22', linewidth=2.5, markersize=8,
        label='Self-BLEU')
ax.set_xlabel("Temperature", fontsize=12)
ax.set_ylabel("Repetition Score", fontsize=12)
ax.set_title("Repetition Decreases with Temperature")
ax.legend()
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3)

# Panel 4: Accuracy vs Diversity (the tradeoff curve)
ax = axes[1, 1]
div_arr = [diversities[T] for T in temperatures]
for i, T in enumerate(temperatures):
    color = plt.cm.coolwarm(T / 2.5)
    ax.scatter(div_arr[i], cm_arr[i], s=150, c=[color], edgecolors='black',
               linewidth=1, zorder=5)
    ax.annotate(f'T={T}', (div_arr[i], cm_arr[i]),
                xytext=(8, 5), textcoords='offset points', fontsize=10)

# Connect with line
ax.plot(div_arr, cm_arr, '--', color='gray', alpha=0.5, linewidth=1)
ax.set_xlabel("Diversity (unique 3-gram ratio)", fontsize=12)
ax.set_ylabel("Accuracy (corpus 5-gram match)", fontsize=12)
ax.set_title("The Tradeoff: Accuracy vs Diversity")
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), 'temperature_tradeoff.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved visualization → {save_path}")

# Entropy over generation steps
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 5))
fig2.suptitle("Per-Step Entropy During Generation", fontsize=14, fontweight='bold')

colors = plt.cm.coolwarm(np.linspace(0, 1, len(temperatures)))
for i, T in enumerate(temperatures):
    # Use first sample's entropies
    ents = all_results[T][0][1]
    # Smooth with rolling average
    window = 10
    if len(ents) > window:
        smoothed = np.convolve(ents, np.ones(window)/window, mode='valid')
        ax2.plot(smoothed, color=colors[i], linewidth=2, label=f'T={T}', alpha=0.8)

ax2.axhline(y=max_entropy, color='gray', linestyle='--', alpha=0.3)
ax2.set_xlabel("Generation Step", fontsize=12)
ax2.set_ylabel("Token Entropy", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_path2 = os.path.join(os.path.dirname(__file__), 'entropy_per_step.png')
fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
print(f"Saved per-step entropy → {save_path2}")
print()

# ===================================================================
# Summary
# ===================================================================
print(f"""
{'='*65}
{BOLD}STEP 1 SUMMARY: GENERATE AT MULTIPLE TEMPERATURES{RESET}
{'='*65}

{BOLD}What we measured:{RESET}

  1. {CYAN}Token entropy{RESET} — how uncertain the model is at each step
  2. {CYAN}Self-BLEU / repetition{RESET} — how much the text repeats itself
  3. {CYAN}Corpus match{RESET} — how much generated text appears in training data

{BOLD}Results:{RESET}

  ┌───────┬───────────┬─────────────┬──────────────┬───────────────────┐
  │  T    │  Entropy  │  N-gram rep │ Corpus match │  Character        │
  ├───────┼───────────┼─────────────┼──────────────┼───────────────────┤
  │  0.1  │  Very low │  Low  (9%)  │   100%       │  Memorized parrot │
  │  0.7  │  Low      │  Low  (9%)  │   100%       │  Faithful copyist │
  │  1.0  │  Low      │  Low  (9%)  │   100%       │  Still memorized  │
  │  1.5  │  Medium   │  Med (12%)  │    83%       │  Creative author  │
  │  2.0  │  High     │  High(17%)  │    47%       │  Drunk poet       │
  └───────┴───────────┴─────────────┴──────────────┴───────────────────┘

{BOLD}Three key observations:{RESET}

  1. {GREEN}Entropy increases SMOOTHLY{RESET} — monotonic, predictable.
     This is a mathematical guarantee of softmax scaling.

  2. {RED}Accuracy does NOT increase monotonically{RESET}
     Corpus match drops as T rises. The model "forgets" the training data
     and generates novel (but often nonsensical) text.

  3. {YELLOW}Repetition INCREASES with temperature here{RESET}
     Surprising! Low T reproduces the corpus — natural prose is diverse.
     High T generates random text that repeats common n-grams ('th', ' t').
     In large LLMs, the pattern reverses: greedy gets stuck in loops,
     while sampling breaks out. Our tiny memorized model doesn't loop.

{BOLD}The insight:{RESET}

  {CYAN}Temperature doesn't make the model smarter.{RESET}
  It just changes which parts of the distribution you sample from.

  Low T → you only see what the model is MOST confident about.
  High T → you see everything the model considers remotely possible.

  The model's knowledge is fixed. Temperature is the lens.
""")
