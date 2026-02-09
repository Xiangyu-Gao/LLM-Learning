"""
Step 3: Inspect early vs late tokens — exposure bias made visible.

Trains the model from step 2 to a partially-learned state, then generates
multiple sequences and color-codes correct vs wrong tokens to show where
generations diverge and collapse.
"""

import torch
import torch.nn.functional as F

from train_tiny_char_level_lm import (
    CausalMLP_LM, CONTEXT_LEN, EMBED_DIM, HIDDEN_DIM,
    vocab_size, idx_to_char, encode, decode, data, get_batch,
)

# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------
GREEN = "\033[92m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"
BG_RED = "\033[41m"

def color_by_correctness(gen_tokens, expected_tokens):
    """Green for correct, RED background for wrong, dim after end of corpus."""
    parts = []
    for i, t in enumerate(gen_tokens):
        ch = idx_to_char[t]
        if i < len(expected_tokens):
            if t == expected_tokens[i]:
                parts.append(f"{GREEN}{ch}{RESET}")
            else:
                parts.append(f"{BG_RED}{BOLD}{ch}{RESET}")
        else:
            # Past the end of corpus — no ground truth
            parts.append(f"{DIM}{ch}{RESET}")
    return "".join(parts)

# ---------------------------------------------------------------------------
# Autoregressive generation with per-token confidence tracking
# ---------------------------------------------------------------------------
def generate(model, seed_tokens, length=200):
    """Generate tokens and record the model's confidence at each step."""
    model.eval()
    generated = list(seed_tokens)
    confidences = []

    with torch.no_grad():
        for _ in range(length):
            inp = torch.tensor([generated[-CONTEXT_LEN:]], dtype=torch.long)
            logits = model(inp)
            probs = F.softmax(logits[0, -1], dim=0)
            next_char = torch.multinomial(probs, 1).item()
            confidences.append(probs[next_char].item())
            generated.append(next_char)

    model.train()
    gen_tokens = generated[len(seed_tokens):]
    return gen_tokens, confidences

# ---------------------------------------------------------------------------
# Train to a partially-learned state
# ---------------------------------------------------------------------------
def train_model(steps):
    model = CausalMLP_LM(vocab_size, CONTEXT_LEN, EMBED_DIM, HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    for step in range(1, steps + 1):
        x, y = get_batch(128)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 1:
            print(f"  Step {step:>4d} | Loss: {loss.item():.4f}")

    return model

# ---------------------------------------------------------------------------
# Run the experiment
# ---------------------------------------------------------------------------
GEN_LENGTH = 200
NUM_SAMPLES = 3

corpus_tokens = data.tolist()
corpus_str = decode(corpus_tokens)

for train_steps, label in [(30, "UNDERTRAINED (30 steps)"), (500, "WELL-TRAINED (500 steps)")]:
    print(f"\n{'='*70}")
    print(f" {label}")
    print(f"{'='*70}")
    model = train_model(train_steps)

    seeds = [
        ("FAMILIAR seed (from corpus)", data[:CONTEXT_LEN].tolist()),
        ("NOVEL seed (never seen in training)", encode("Now is the winter of our discon")),
    ]

    for seed_label, seed in seeds:
        print(f"\n--- {seed_label} ---")
        print(f"Seed: {DIM}{decode(seed)}{RESET}")

        # Find expected continuation (if seed is in corpus)
        seed_str = decode(seed)
        seed_pos = corpus_str.find(seed_str)
        if seed_pos >= 0:
            expected_start = seed_pos + len(seed_str)
            expected_tokens = corpus_tokens[expected_start:]
            print(f"Color key: {GREEN}correct{RESET} | {BG_RED}{BOLD}wrong{RESET} | {DIM}past corpus end{RESET}\n")
        else:
            expected_tokens = None
            print(f"(No ground truth — seed not in corpus)\n")

        for sample_i in range(NUM_SAMPLES):
            gen_tokens, confidences = generate(model, seed, length=GEN_LENGTH)

            # Color-coded output
            if expected_tokens is not None:
                display = color_by_correctness(gen_tokens, expected_tokens)
                # Count errors
                n_compare = min(len(gen_tokens), len(expected_tokens))
                errors = [(i, gen_tokens[i], expected_tokens[i])
                          for i in range(n_compare) if gen_tokens[i] != expected_tokens[i]]
                first_err = errors[0][0] if errors else None
                accuracy = (n_compare - len(errors)) / n_compare * 100
            else:
                display = decode(gen_tokens)
                first_err = None
                accuracy = None

            # Confidence stats
            early_conf = sum(confidences[:10]) / 10
            mid_idx = len(confidences) // 2
            mid_conf = sum(confidences[mid_idx-5:mid_idx+5]) / 10
            late_conf = sum(confidences[-10:]) / 10

            print(f"Sample {sample_i + 1}:")
            print(f"  {display}")
            print(f"  Confidence:  early={early_conf:.3f}  mid={mid_conf:.3f}  late={late_conf:.3f}")
            if accuracy is not None:
                print(f"  Accuracy: {accuracy:.1f}% ({len(errors)} errors in {n_compare} chars)")
                if first_err is not None:
                    print(f"  First error at token #{first_err}: "
                          f"expected '{idx_to_char[expected_tokens[first_err]]}' "
                          f"got '{idx_to_char[gen_tokens[first_err]]}'")
                    # Show the cascade: count errors in first half vs second half
                    half = n_compare // 2
                    errs_first_half = sum(1 for i, _, _ in errors if i < half)
                    errs_second_half = sum(1 for i, _, _ in errors if i >= half)
                    print(f"  Error cascade: {errs_first_half} errors in first half, "
                          f"{errs_second_half} errors in second half")
                else:
                    print(f"  {GREEN}Perfect match{RESET}")
            print()

# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------
print(f"""{'='*70}
 WHY THIS HAPPENS — EXPOSURE BIAS
{'='*70}

{BOLD}Why are early tokens more stable?{RESET}
  The first few generated tokens follow a seed of REAL corpus text.
  The model's context window is full of text it saw during training,
  so its predictions are confident and accurate.

{BOLD}Why do errors compound?{RESET}
  Once the model makes a mistake, that wrong character enters the
  context window. The model never saw this wrong context during
  training (it was always fed real text via teacher forcing).
  So it makes another mistake, which makes the next prediction
  worse, and so on — a snowball effect.

{BOLD}This is exposure bias:{RESET}
  Training: model always sees ground-truth context (teacher forcing)
  Generation: model sees its own (possibly wrong) outputs
  The model was never trained to recover from its own errors.

{BOLD}Where will generations collapse?{RESET}
  Look at the "Error cascade" line above:
  - More errors in the second half = errors are compounding
  - First error triggers a chain reaction
  - Novel seeds collapse immediately (no ground truth match)
""")
