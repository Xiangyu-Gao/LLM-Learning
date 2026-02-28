"""
Step 1: Failure Modes — Build Debugging Instincts.

For each failure mode:
  1. Explain what it looks like and the root cause
  2. Trigger it deliberately on our tiny model
  3. Reduce it using decoding tweaks

Failure modes:
  1. Hallucination      — confident wrong facts (likelihood objective)
  2. Exposure Bias      — divergence after long generation (teacher forcing mismatch)
  3. Repetition Loops   — "the the the" (decoding entropy collapse)
  4. Prompt Sensitivity  — minor prompt changes, huge shifts (latent space curvature)
  5. Overconfidence     — high logprob, wrong answer (miscalibration)
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Week1', 'Day4'))

from train_tiny_char_level_lm import vocab_size, encode, decode, data, CONTEXT_LEN, idx_to_char, char_to_idx, CORPUS
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
MAGENTA = "\033[95m"
RESET = "\033[0m"


# ===================================================================
# Train model
# ===================================================================
print(f"""
{'='*65}
{BOLD}FAILURE MODES — BUILD DEBUGGING INSTINCTS{RESET}
{'='*65}
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
    optimizer.zero_grad(); loss.backward(); optimizer.step()

print(f"  Final loss: {loss.item():.4f}")
model.eval()
print()


# ===================================================================
# Helper functions
# ===================================================================
def greedy_generate(model, seed_tokens, length, temperature=1.0):
    """Greedy generation with temperature."""
    generated = list(seed_tokens)
    with torch.no_grad():
        for _ in range(length):
            inp = torch.tensor([generated[-CONTEXT_LEN:]], dtype=torch.long)
            logits = model(inp)[0, -1] / temperature
            next_token = logits.argmax().item()
            generated.append(next_token)
    return generated[len(seed_tokens):]


def sample_generate(model, seed_tokens, length, temperature=1.0, top_k=0, top_p=1.0):
    """Sampling with temperature, top-k, top-p."""
    generated = list(seed_tokens)
    with torch.no_grad():
        for _ in range(length):
            inp = torch.tensor([generated[-CONTEXT_LEN:]], dtype=torch.long)
            logits = model(inp)[0, -1] / temperature
            probs = F.softmax(logits, dim=-1)

            if top_k > 0:
                topk_probs, topk_idx = torch.topk(probs, top_k)
                probs = torch.zeros_like(probs)
                probs.scatter_(0, topk_idx, topk_probs)
                probs = probs / probs.sum()

            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=0)
                mask = cumsum - sorted_probs < top_p
                sorted_probs[~mask] = 0.0
                probs = torch.zeros_like(probs)
                probs.scatter_(0, sorted_idx, sorted_probs)
                probs = probs / probs.sum()

            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
    return generated[len(seed_tokens):]


def get_per_position_stats(model, input_tokens, target_tokens):
    """Get loss, probability, and entropy at each position."""
    inp = torch.tensor([input_tokens], dtype=torch.long)
    with torch.no_grad():
        logits = model(inp)
    probs = F.softmax(logits[0], dim=-1)

    losses = []
    correct_probs = []
    entropies = []
    predictions = []

    for pos in range(len(target_tokens)):
        target = target_tokens[pos]
        p = probs[pos]
        loss = -torch.log(p[target] + 1e-10).item()
        losses.append(loss)
        correct_probs.append(p[target].item())
        entropy = -(p * torch.log(p + 1e-10)).sum().item()
        entropies.append(entropy)
        predictions.append(p.argmax().item())

    return losses, correct_probs, entropies, predictions


# ===================================================================
# FAILURE MODE TABLE
# ===================================================================
print(f"{BOLD}FAILURE MODE REFERENCE TABLE{RESET}")
print(f"{'='*65}")
print()
print(f"  {'Failure':<20} {'What It Looks Like':<25} {'Root Cause'}")
print(f"  {'─'*20} {'─'*25} {'─'*30}")
print(f"  {'Hallucination':<20} {'Confident wrong facts':<25} {'Likelihood objective'}")
print(f"  {'Exposure Bias':<20} {'Diverges after long gen':<25} {'Teacher forcing mismatch'}")
print(f"  {'Repetition Loops':<20} {'\"the the the\"':<25} {'Decoding entropy collapse'}")
print(f"  {'Prompt Sensitivity':<20} {'Minor change → big shift':<25} {'High latent curvature'}")
print(f"  {'Overconfidence':<20} {'High logprob, wrong':<25} {'Miscalibration'}")
print()
print()


# ===================================================================
# FAILURE 1: HALLUCINATION
# ===================================================================
print(f"""
{'='*65}
{BOLD}FAILURE 1: HALLUCINATION{RESET}
{'='*65}

  {BOLD}What it looks like:{RESET}  The model produces confident but wrong output.
  {BOLD}Root cause:{RESET}         The training objective is P(next_token | context),
                       NOT "is this factually correct?"
                       The model maximizes PLAUSIBILITY, not TRUTH.
""")

print(f"  {BOLD}Trigger: prompt with text NOT in the training corpus{RESET}")
print()

# Our model memorized Shakespeare. Feed it non-Shakespeare.
non_corpus_prompts = [
    "The weather today is",
    "Machine learning is",
    "Once upon a time in",
    "2 + 2 = ",
    "Hello world",
]

# Also test a corpus prompt for comparison
corpus_prompt = "To be, or not to be"

print(f"  {GREEN}Corpus prompt (model has seen this):{RESET}")
tokens = encode(corpus_prompt[:CONTEXT_LEN])
gen = greedy_generate(model, tokens, 30)
print(f"    \"{corpus_prompt}\" → \"{decode(gen)}\"")

# Check: is the continuation in the corpus?
full_text = corpus_prompt + decode(gen)
in_corpus = full_text[:50] in CORPUS
print(f"    In corpus: {GREEN + 'YES' + RESET if in_corpus else RED + 'NO' + RESET}")
print()

print(f"  {RED}Non-corpus prompts (model has NEVER seen these):{RESET}")
for prompt in non_corpus_prompts:
    # Some chars might not be in our vocab — filter
    safe_prompt = ''.join(c for c in prompt if c in char_to_idx)
    if len(safe_prompt) < 3:
        continue
    tokens = encode(safe_prompt)
    gen_len = min(30, CONTEXT_LEN - len(tokens))
    if gen_len < 5:
        continue
    gen = greedy_generate(model, tokens, gen_len)
    gen_text = decode(gen)
    print(f"    \"{safe_prompt}\" → \"{gen_text}\"")

    # Measure confidence — stay within CONTEXT_LEN
    total_len = min(len(tokens) + len(gen), CONTEXT_LEN)
    inp = torch.tensor([tokens[:total_len - len(gen)] + gen[:total_len - len(tokens)]], dtype=torch.long)
    with torch.no_grad():
        logits = model(inp)
    probs = F.softmax(logits[0], dim=-1)
    start_pos = min(len(tokens), inp.shape[1])
    max_probs = probs[start_pos:].max(dim=-1).values
    avg_conf = max_probs.mean().item() if max_probs.numel() > 0 else 0
    print(f"    Average confidence: {avg_conf:.1%} {RED}(high confidence on garbage!){RESET}")
    print()

print(f"  {BOLD}Key insight:{RESET}")
print(f"  The model is equally confident on in-corpus and out-of-corpus text.")
print(f"  It has NO mechanism to say \"I don't know\" — it always produces")
print(f"  the most likely next token, even when the input is nonsensical.")
print()

print(f"  {BOLD}Mitigation — entropy-based uncertainty detection:{RESET}")
print()

# Show that entropy is slightly higher on non-corpus inputs
print(f"  {'Input type':<25} {'Avg entropy':>12} {'Avg max prob':>13}")
print(f"  {'-'*25} {'-'*12} {'-'*13}")

# Corpus samples
corpus_entropies = []
corpus_max_probs = []
for start in range(0, len(data) - CONTEXT_LEN - 1, 20):
    inp = data[start:start + CONTEXT_LEN].unsqueeze(0)
    with torch.no_grad():
        logits = model(inp)
    probs = F.softmax(logits[0], dim=-1)
    ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
    maxp = probs.max(dim=-1).values.mean().item()
    corpus_entropies.append(ent)
    corpus_max_probs.append(maxp)

print(f"  {'Corpus (seen)':>25} {np.mean(corpus_entropies):>11.3f} {np.mean(corpus_max_probs):>12.1%}")

# Non-corpus (generated text feeding back)
non_corpus_ents = []
non_corpus_maxps = []
for prompt in ["The weather today is", "Hello world"]:
    safe = ''.join(c for c in prompt if c in char_to_idx)
    if len(safe) < 5:
        continue
    tokens = encode(safe)
    gen = greedy_generate(model, tokens, CONTEXT_LEN - len(tokens))
    inp = torch.tensor([tokens + gen], dtype=torch.long)
    with torch.no_grad():
        logits = model(inp)
    probs = F.softmax(logits[0], dim=-1)
    ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
    maxp = probs.max(dim=-1).values.mean().item()
    non_corpus_ents.append(ent)
    non_corpus_maxps.append(maxp)

if non_corpus_ents:
    print(f"  {'Non-corpus (hallucinated)':>25} {np.mean(non_corpus_ents):>11.3f} {np.mean(non_corpus_maxps):>12.1%}")
print()
print(f"  {YELLOW}In a tiny memorized model, entropy may not differ much.{RESET}")
print(f"  In large LLMs, calibration techniques (temperature scaling,")
print(f"  conformal prediction) can help detect hallucinations.")
print()


# ===================================================================
# FAILURE 2: EXPOSURE BIAS
# ===================================================================
print(f"""
{'='*65}
{BOLD}FAILURE 2: EXPOSURE BIAS{RESET}
{'='*65}

  {BOLD}What it looks like:{RESET}  Quality degrades over longer generations.
                       First 10 tokens are great, last 10 are garbage.
  {BOLD}Root cause:{RESET}         During training, the model always sees GROUND TRUTH
                       as input (teacher forcing). During generation, it
                       sees ITS OWN outputs — including mistakes.
                       Errors compound: one wrong token shifts everything.
""")

print(f"  {BOLD}Trigger: compare teacher-forced vs autoregressive accuracy{RESET}")
print()

# Take a long corpus passage
passage = CORPUS[:128]
passage_tokens = encode(passage)

# Teacher forced: model sees ground truth at each step
inp_tf = torch.tensor([passage_tokens[:CONTEXT_LEN]], dtype=torch.long)
tgt_tf = passage_tokens[1:CONTEXT_LEN + 1]

losses_tf, probs_tf, _, preds_tf = get_per_position_stats(
    model, passage_tokens[:CONTEXT_LEN], tgt_tf
)

# Autoregressive: model sees its own predictions
seed_len = 5
seed_tokens = passage_tokens[:seed_len]
gen_ar = greedy_generate(model, seed_tokens, CONTEXT_LEN - seed_len)

# Now measure: how much does autoregressive output match the ground truth?
truth = passage_tokens[seed_len:CONTEXT_LEN]
ar_output = gen_ar[:len(truth)]

# Per-position match rate
matches = [1 if a == b else 0 for a, b in zip(ar_output, truth)]

# Show the divergence point
print(f"  Ground truth: \"{passage[:CONTEXT_LEN]}\"")
print(f"  AR generated: \"{decode(seed_tokens)}{decode(ar_output)}\"")
print()

# Find first error
first_error = next((i for i, m in enumerate(matches) if m == 0), len(matches))
print(f"  First error at position {seed_len + first_error} "
      f"(after {first_error} correct tokens)")
print()

# Show accuracy in windows
window = 5
print(f"  {BOLD}Accuracy over time (teacher-forced vs autoregressive):{RESET}")
print()
print(f"  {'Position':>10} {'Teacher-forced':>16} {'Autoregressive':>16}")
print(f"  {'-'*10} {'-'*16} {'-'*16}")

for start in range(0, min(len(matches), 25), window):
    end = min(start + window, len(matches))
    tf_acc = sum(1 for p, t in zip(preds_tf[start:end], tgt_tf[start:end]) if p == t) / (end - start)
    ar_acc = sum(matches[start:end]) / (end - start)

    tf_color = GREEN if tf_acc > 0.8 else (YELLOW if tf_acc > 0.5 else RED)
    ar_color = GREEN if ar_acc > 0.8 else (YELLOW if ar_acc > 0.5 else RED)

    print(f"  {seed_len + start:>3}-{seed_len + end:>3}     "
          f"{tf_color}{tf_acc:>12.0%}{RESET}     {ar_color}{ar_acc:>12.0%}{RESET}")

print()
print(f"  {BOLD}Key insight:{RESET}")
print(f"  Teacher-forced accuracy stays high (model sees true context).")
print(f"  Autoregressive accuracy drops after errors compound.")
print(f"  Each mistake shifts the distribution, causing more mistakes.")
print()

print(f"  {BOLD}Mitigation — scheduled sampling and nucleus sampling:{RESET}")
print(f"  During training: occasionally feed model's own predictions (scheduled sampling)")
print(f"  During generation: use sampling (top-p) to explore, reducing error lockstep")
print()

# Show that sampling helps
torch.manual_seed(0)
n_trials = 5
greedy_matches = []
sampled_matches = []

for _ in range(n_trials):
    gen_g = greedy_generate(model, seed_tokens, CONTEXT_LEN - seed_len)
    gen_s = sample_generate(model, seed_tokens, CONTEXT_LEN - seed_len,
                           temperature=0.8, top_p=0.9)
    g_match = sum(1 for a, b in zip(gen_g[:len(truth)], truth) if a == b) / len(truth)
    s_match = sum(1 for a, b in zip(gen_s[:len(truth)], truth) if a == b) / len(truth)
    greedy_matches.append(g_match)
    sampled_matches.append(s_match)

print(f"  Match with ground truth ({n_trials} trials):")
print(f"    Greedy:                {np.mean(greedy_matches):.1%}")
print(f"    Sampling (T=0.8, p=0.9): {np.mean(sampled_matches):.1%}")
print(f"  {YELLOW}(In a memorized model, greedy may match better since it")
print(f"   reproduces the corpus exactly. Sampling helps in GENERAL models.){RESET}")
print()


# ===================================================================
# FAILURE 3: REPETITION LOOPS
# ===================================================================
print(f"""
{'='*65}
{BOLD}FAILURE 3: REPETITION LOOPS{RESET}
{'='*65}

  {BOLD}What it looks like:{RESET}  "the the the" or repeated phrases/sentences.
  {BOLD}Root cause:{RESET}         Greedy decoding + high-probability tokens =
                       deterministic loops. Once a pattern starts,
                       it reinforces itself (attention to recent tokens
                       produces the same prediction).
""")

print(f"  {BOLD}Trigger: greedy decoding at low temperature{RESET}")
print()

# Try different seeds and find ones that loop
seeds_to_try = [
    "To",
    "The",
    "And",
    "Or ",
    "No ",
    "For",
    "'ti",
    "Mus",
    "Whe",
]

print(f"  {'Seed':<8} {'Temperature':<13} {'Generated (60 chars)':<45} {'Loops?'}")
print(f"  {'-'*8} {'-'*13} {'-'*45} {'-'*8}")

loop_examples = []

for seed in seeds_to_try:
    safe_seed = ''.join(c for c in seed if c in char_to_idx)
    if len(safe_seed) < 2:
        continue
    tokens = encode(safe_seed)

    for temp in [0.3, 1.0]:
        gen = greedy_generate(model, tokens, 60, temperature=temp)
        text = decode(gen)

        # Detect loops: repeated 3-grams
        ngrams = [text[i:i+4] for i in range(len(text) - 3)]
        counts = Counter(ngrams)
        max_repeat = max(counts.values()) if counts else 0
        has_loop = max_repeat >= 5

        display = text[:45]
        loop_str = f"{RED}YES (×{max_repeat}){RESET}" if has_loop else f"{GREEN}no{RESET}"

        if has_loop:
            loop_examples.append((safe_seed, temp, text))

        print(f"  \"{safe_seed}\"  T={temp:<11} \"{display}\" {loop_str}")

print()

print(f"  {BOLD}Mitigation — temperature and repetition penalty:{RESET}")
print()

if loop_examples:
    seed_text, _, _ = loop_examples[0]
    tokens = encode(seed_text)
    print(f"  Using seed \"{seed_text}\" which loops at T=0.3:")
    print()

    # Show different mitigations
    mitigations = [
        ("Greedy T=0.3 (loops)", lambda: greedy_generate(model, tokens, 60, temperature=0.3)),
        ("Greedy T=1.0", lambda: greedy_generate(model, tokens, 60, temperature=1.0)),
        ("Sample T=1.0, p=0.9", lambda: sample_generate(model, tokens, 60, temperature=1.0, top_p=0.9)),
        ("Sample T=1.2, k=10", lambda: sample_generate(model, tokens, 60, temperature=1.2, top_k=10)),
    ]

    for name, gen_fn in mitigations:
        torch.manual_seed(42)
        gen = gen_fn()
        text = decode(gen)[:50]

        ngrams = [text[i:i+4] for i in range(len(text) - 3)]
        counts = Counter(ngrams)
        max_repeat = max(counts.values()) if counts else 0

        loop_str = f"{RED}loops ×{max_repeat}{RESET}" if max_repeat >= 4 else f"{GREEN}ok{RESET}"
        print(f"    {name:<30} \"{text}\" {loop_str}")

    print()
    print(f"  {GREEN}Higher temperature and sampling break repetition cycles.{RESET}")
    print(f"  Production systems also use frequency/presence penalties:")
    print(f"    logit[token] -= penalty × count(token in generated)")
else:
    print(f"  {GREEN}This well-memorized model doesn't loop easily — it reproduces{RESET}")
    print(f"  {GREEN}the corpus faithfully. Repetition is more common in larger models{RESET}")
    print(f"  {GREEN}on open-ended generation tasks.{RESET}")
print()


# ===================================================================
# FAILURE 4: PROMPT SENSITIVITY
# ===================================================================
print(f"""
{'='*65}
{BOLD}FAILURE 4: PROMPT SENSITIVITY{RESET}
{'='*65}

  {BOLD}What it looks like:{RESET}  Minor changes to the prompt produce
                       completely different outputs.
  {BOLD}Root cause:{RESET}         The model's latent space has high curvature.
                       Nearby inputs (in text space) may be far apart
                       in embedding space, leading to different attention
                       patterns and predictions.
""")

print(f"  {BOLD}Trigger: make tiny changes to a prompt and compare outputs{RESET}")
print()

# Base prompt and variations
base = "To be, or"
variations = [
    ("To be, or", "exact original"),
    ("to be, or", "lowercase 't'"),
    ("To be  or", "double space, no comma"),
    ("To be, Or", "capitalize 'or'"),
    ("To be, or ", "trailing space"),
    (" To be, or", "leading space"),
]

print(f"  {'Prompt':<20} {'Change':<25} {'Generated (30 chars)'}")
print(f"  {'-'*20} {'-'*25} {'-'*35}")

base_gen = None
for prompt, change in variations:
    safe = ''.join(c for c in prompt if c in char_to_idx)
    tokens = encode(safe)
    gen = greedy_generate(model, tokens, 30)
    text = decode(gen)[:30]

    if base_gen is None:
        base_gen = gen

    # How different from base?
    overlap = sum(1 for a, b in zip(gen, base_gen) if a == b) / max(len(gen), 1)

    if prompt == base:
        print(f"  \"{safe}\"  {'(baseline)':<25} \"{text}\"")
    else:
        color = GREEN if overlap > 0.8 else (YELLOW if overlap > 0.5 else RED)
        print(f"  \"{safe}\"  {change:<25} \"{text}\"  {color}({overlap:.0%} match){RESET}")

print()

# Measure in embedding space
print(f"  {BOLD}Why this happens — embedding space distances:{RESET}")
print()

with torch.no_grad():
    base_tokens = encode(''.join(c for c in base if c in char_to_idx))
    base_inp = torch.tensor([base_tokens], dtype=torch.long)
    base_emb = model.embed(base_inp) + model.pos_embed(torch.arange(len(base_tokens)))
    base_emb_flat = base_emb[0, -1]  # last position embedding

    for prompt, change in variations[1:]:
        safe = ''.join(c for c in prompt if c in char_to_idx)
        tokens = encode(safe)
        if len(tokens) == 0:
            continue
        inp = torch.tensor([tokens], dtype=torch.long)
        emb = model.embed(inp) + model.pos_embed(torch.arange(len(tokens)))
        emb_flat = emb[0, -1]

        # Compute distances
        l2_dist = torch.norm(base_emb_flat - emb_flat).item()
        cos_sim = F.cosine_similarity(base_emb_flat.unsqueeze(0),
                                       emb_flat.unsqueeze(0)).item()

        print(f"    \"{safe}\" vs base: L2={l2_dist:.3f}, cosine={cos_sim:.4f}  ({change})")

print()
print(f"  {BOLD}Mitigation:{RESET}")
print(f"  • Prompt engineering: standardize formatting (whitespace, casing)")
print(f"  • Ensemble: generate with multiple prompt variants, aggregate")
print(f"  • In large LLMs: instruction tuning reduces sensitivity")
print()


# ===================================================================
# FAILURE 5: OVERCONFIDENCE
# ===================================================================
print(f"""
{'='*65}
{BOLD}FAILURE 5: OVERCONFIDENCE{RESET}
{'='*65}

  {BOLD}What it looks like:{RESET}  The model assigns high probability to wrong tokens.
                       It's "certain" but incorrect.
  {BOLD}Root cause:{RESET}         Cross-entropy training pushes the model to be
                       maximally confident on training data.
                       But this confidence doesn't transfer to
                       inputs outside the training distribution.
""")

print(f"  {BOLD}Trigger: feed slightly corrupted corpus text{RESET}")
print()

# Take a corpus passage and corrupt some characters
passage = "To be, or not to be, that is the"
passage_tokens = encode(passage)

# Corrupt: swap two characters
corrupted_passages = [
    ("To be, or not to be, that is the", "Original (no corruption)"),
    ("To eb, or not to be, that is the", "Swap 'b' and 'e'"),
    ("To be, ro not to be, that is the", "Swap 'or' → 'ro'"),
    ("To be, or not ot be, that is the", "Swap 'to' → 'ot'"),
    ("Xo be, or not to be, that is the", "T → X (unknown pattern)"),
]

print(f"  {'Input':<40} {'Pred':>6} {'True':>6} {'P(pred)':>8} {'P(true)':>8} {'Correct?'}")
print(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*9}")

overconf_data = []

for text, desc in corrupted_passages:
    safe = ''.join(c for c in text if c in char_to_idx)
    tokens = encode(safe)
    if len(tokens) < 5:
        continue

    inp = torch.tensor([tokens[:-1]], dtype=torch.long)
    target = tokens[-1]

    with torch.no_grad():
        logits = model(inp)
    probs = F.softmax(logits[0, -1], dim=-1)
    pred_token = probs.argmax().item()
    p_pred = probs[pred_token].item()
    p_true = probs[target].item()

    correct = pred_token == target
    color = GREEN if correct else RED
    true_char = idx_to_char[target]
    pred_char = idx_to_char[pred_token]

    print(f"  \"{safe}\"")
    print(f"  {desc:<40} '{pred_char:>4}' '{true_char:>4}' "
          f"{p_pred:>7.1%}  {p_true:>7.1%}  {color}{'YES' if correct else 'NO':>7}{RESET}")

    overconf_data.append((desc, p_pred, p_true, correct))
    print()

print(f"  {BOLD}The calibration problem:{RESET}")
print()

# Show calibration: binned probability vs actual accuracy
print(f"  If a model says P=90%, is it right 90% of the time?")
print()

# Test on full corpus
all_probs = []
all_correct = []

with torch.no_grad():
    for start in range(0, len(data) - CONTEXT_LEN - 1, 4):
        inp = data[start:start + CONTEXT_LEN].unsqueeze(0)
        tgt = data[start + 1:start + CONTEXT_LEN + 1]
        logits = model(inp)
        probs = F.softmax(logits[0], dim=-1)

        for pos in range(CONTEXT_LEN):
            p_max = probs[pos].max().item()
            pred = probs[pos].argmax().item()
            correct = (pred == tgt[pos].item())
            all_probs.append(p_max)
            all_correct.append(correct)

all_probs = np.array(all_probs)
all_correct = np.array(all_correct)

# Bin by predicted probability
bins = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
print(f"  {'Predicted P':>15} {'Count':>8} {'Actual accuracy':>17} {'Calibration'}")
print(f"  {'-'*15} {'-'*8} {'-'*17} {'-'*15}")

cal_predicted = []
cal_actual = []

for i in range(len(bins) - 1):
    mask = (all_probs >= bins[i]) & (all_probs < bins[i + 1])
    if mask.sum() == 0:
        continue
    avg_pred = all_probs[mask].mean()
    avg_correct = all_correct[mask].mean()
    count = mask.sum()

    gap = abs(avg_pred - avg_correct)
    cal_str = f"{GREEN}good{RESET}" if gap < 0.05 else (f"{YELLOW}off by {gap:.0%}{RESET}" if gap < 0.15 else f"{RED}off by {gap:.0%}{RESET}")

    print(f"  {bins[i]:.0%}-{bins[i+1]:.0%}  {count:>10,} {avg_correct:>16.1%} {cal_str}")

    cal_predicted.append(avg_pred)
    cal_actual.append(avg_correct)

print()
print(f"  {BOLD}Mitigation — temperature scaling for calibration:{RESET}")
print()

# Find optimal temperature for calibration
best_temp = 1.0
best_ece = float('inf')

for temp in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
    # Rescale logits
    ece = 0
    for i in range(len(bins) - 1):
        mask = (all_probs >= bins[i]) & (all_probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        # We'd need to recompute with temp, but approximate:
        scaled_probs = all_probs[mask] ** (1 / temp)
        scaled_probs = scaled_probs / scaled_probs.mean() * all_probs[mask].mean()
        avg_pred = scaled_probs.mean()
        avg_correct = all_correct[mask].mean()
        ece += abs(avg_pred - avg_correct) * mask.sum()

    ece /= len(all_probs)
    if ece < best_ece:
        best_ece = ece
        best_temp = temp

print(f"  Expected Calibration Error (ECE) at T=1.0: {best_ece:.4f}")
print(f"  Best temperature for calibration: T={best_temp}")
print(f"  Temperature scaling adjusts confidence WITHOUT changing rankings.")
print(f"  T > 1 → less confident (often better calibrated)")
print(f"  T < 1 → more confident (often overconfident)")
print()


# ===================================================================
# Visualization
# ===================================================================
print(f"{BOLD}GENERATING VISUALIZATION...{RESET}")
print(f"{'='*65}")
print()

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("LLM Failure Modes: Trigger → Diagnose → Mitigate",
             fontsize=14, fontweight='bold')

# --- Panel 1: Exposure bias — accuracy over position ---
ax = axes[0, 0]
positions_plot = list(range(seed_len, seed_len + len(matches)))
# Smooth with sliding window
win = 3
smooth_tf = [np.mean([1 if preds_tf[max(0, i-win):i+win+1][j] == tgt_tf[max(0, i-win):i+win+1][j]
             else 0 for j in range(len(preds_tf[max(0, i-win):i+win+1]))]) for i in range(len(preds_tf))]
smooth_ar = [np.mean(matches[max(0, i-win):i+win+1]) for i in range(len(matches))]

ax.plot(range(len(smooth_tf)), smooth_tf, 'g-', linewidth=2, label='Teacher-forced', alpha=0.8)
ax.plot(range(len(smooth_ar)), smooth_ar, 'r-', linewidth=2, label='Autoregressive', alpha=0.8)
ax.set_xlabel("Position")
ax.set_ylabel("Accuracy (smoothed)")
ax.set_title("Exposure Bias:\nTeacher-Forced vs Autoregressive")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# --- Panel 2: Repetition — entropy per position for different temperatures ---
ax = axes[0, 1]
for temp, color, label in [(0.3, 'red', 'T=0.3 (loops)'),
                            (1.0, 'blue', 'T=1.0'),
                            (1.5, 'green', 'T=1.5 (diverse)')]:
    tokens = encode("To be")
    generated = list(tokens)
    entropies = []
    with torch.no_grad():
        for _ in range(50):
            inp = torch.tensor([generated[-CONTEXT_LEN:]], dtype=torch.long)
            logits = model(inp)[0, -1] / temp
            probs = F.softmax(logits, dim=-1)
            ent = -(probs * torch.log(probs + 1e-10)).sum().item()
            entropies.append(ent)
            next_token = logits.argmax().item()
            generated.append(next_token)

    ax.plot(entropies, color=color, linewidth=1.5, label=label, alpha=0.8)

ax.set_xlabel("Generation step")
ax.set_ylabel("Prediction entropy (nats)")
ax.set_title("Repetition Loops:\nEntropy Collapse at Low Temperature")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Panel 3: Prompt sensitivity — cosine similarity matrix ---
ax = axes[0, 2]
# Compute embeddings for all variations
embeddings = []
labels = []
for prompt, change in variations:
    safe = ''.join(c for c in prompt if c in char_to_idx)
    tokens = encode(safe)
    if len(tokens) == 0:
        continue
    inp = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        # Get last hidden state
        _, hidden_states, _ = model.forward_with_intermediates(inp)
        emb = hidden_states[-1][0, -1]  # last position, post-layernorm
    embeddings.append(emb)
    labels.append(change[:15])

if len(embeddings) > 1:
    emb_stack = torch.stack(embeddings)
    # Compute cosine similarity matrix
    cos_matrix = torch.zeros(len(embeddings), len(embeddings))
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            cos_matrix[i, j] = F.cosine_similarity(
                emb_stack[i].unsqueeze(0), emb_stack[j].unsqueeze(0)
            ).item()

    im = ax.imshow(cos_matrix.numpy(), cmap='RdYlGn', vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Prompt Sensitivity:\nHidden State Cosine Similarity")
    plt.colorbar(im, ax=ax, fraction=0.046)

# --- Panel 4: Calibration curve ---
ax = axes[1, 0]
if cal_predicted and cal_actual:
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax.scatter(cal_predicted, cal_actual, s=100, c='red', zorder=5, label='Model')
    ax.plot(cal_predicted, cal_actual, 'r-', alpha=0.5)
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.05, color='red')
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.05, color='green')
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Actual accuracy")
    ax.set_title("Overconfidence:\nCalibration Curve")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

# --- Panel 5: Confidence distribution ---
ax = axes[1, 1]
ax.hist(all_probs, bins=50, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
ax.axvline(x=np.mean(all_probs), color='red', linestyle='--', linewidth=2,
           label=f'Mean conf: {np.mean(all_probs):.1%}')
ax.set_xlabel("Predicted probability (max)")
ax.set_ylabel("Count")
ax.set_title(f"Confidence Distribution\n(Overall accuracy: {np.mean(all_correct):.1%})")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# --- Panel 6: Failure mode summary ---
ax = axes[1, 2]
ax.axis('off')
ax.set_title("Failure Mode Summary", fontsize=12)

summary_items = [
    ("Hallucination", "#e74c3c",
     "Trigger: out-of-distribution input\n"
     "Fix: entropy thresholds, RAG, RLHF"),
    ("Exposure Bias", "#e67e22",
     "Trigger: long autoregressive gen\n"
     "Fix: scheduled sampling, top-p"),
    ("Repetition", "#f1c40f",
     "Trigger: greedy + low temperature\n"
     "Fix: T↑, sampling, freq penalty"),
    ("Prompt Sensitivity", "#9b59b6",
     "Trigger: whitespace/casing changes\n"
     "Fix: normalization, instruction tuning"),
    ("Overconfidence", "#3498db",
     "Trigger: corrupted/OOD input\n"
     "Fix: temperature scaling, calibration"),
]

y = 0.95
for title, color, desc in summary_items:
    bbox = dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.15,
                edgecolor=color, linewidth=1.5)
    ax.text(0.02, y, title, fontsize=9, fontweight='bold',
            transform=ax.transAxes, va='top', bbox=bbox, color=color)
    ax.text(0.35, y, desc, fontsize=7.5, transform=ax.transAxes,
            va='top', fontfamily='monospace', color='#333333')
    y -= 0.19

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), 'failure_modes.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"  Saved visualization → {save_path}")
print()


# ===================================================================
# Summary
# ===================================================================
print(f"""
{'='*65}
{BOLD}DAY 6 SUMMARY: FAILURE MODES — THINK LIKE A MODEL ENGINEER{RESET}
{'='*65}

{BOLD}The five failure modes:{RESET}

  {RED}1. Hallucination{RESET}
     The model maximizes P(next_token), not P(truth).
     It has no mechanism for "I don't know."
     → Detect via entropy, mitigate with RAG / grounding.

  {RED}2. Exposure Bias{RESET}
     Training: sees ground truth. Generation: sees own mistakes.
     Errors compound — one wrong token derails everything.
     → Mitigate with scheduled sampling, nucleus sampling.

  {RED}3. Repetition Loops{RESET}
     Greedy decoding at low temperature → entropy collapses to 0.
     The model gets stuck in high-probability attractor states.
     → Fix with temperature ↑, sampling, frequency penalty.

  {RED}4. Prompt Sensitivity{RESET}
     "To be, or" vs "to be, or" → different hidden states →
     potentially different outputs. Small input changes propagate.
     → Normalize prompts, use instruction tuning, ensemble.

  {RED}5. Overconfidence{RESET}
     Cross-entropy training pushes toward 100% confidence.
     On training data: well-calibrated. On OOD: dangerously wrong.
     → Temperature scaling, conformal prediction, calibration.

{BOLD}The engineer's mindset:{RESET}

  When an LLM fails, don't just say "it's dumb."
  Ask: {CYAN}which failure mode is this?{RESET}

    Output is wrong but confident?      → Hallucination / Overconfidence
    Output starts good, then degrades?  → Exposure Bias
    Output repeats itself?              → Repetition (decoding issue)
    Tiny prompt change, big output change? → Prompt Sensitivity

  Each failure has a {GREEN}specific root cause{RESET} and {GREEN}specific mitigations{RESET}.
  Model engineering is about matching the right fix to the right failure.

  {BOLD}You now think like a model engineer.{RESET}
""")
