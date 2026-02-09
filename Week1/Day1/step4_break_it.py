"""
Step 4: Deliberately break it — low loss ≠ good generation.

Three sabotage experiments:
  1. Shuffled sequences   — destroy sequential structure
  2. Truncated context    — starve the model of history
  3. Corrupted prefixes   — poison the input with noise

Each one: does loss still decrease?  How are generations?
"""

import torch
import torch.nn.functional as F

from train_tiny_char_level_lm import (
    CausalMLP_LM, CONTEXT_LEN, EMBED_DIM, HIDDEN_DIM,
    vocab_size, idx_to_char, encode, decode, data,
)

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

STEPS = 500
LOG_EVERY = 100

# ---------------------------------------------------------------------------
# Generation helper (reused across experiments)
# ---------------------------------------------------------------------------
def generate(model, seed_tokens, length=150):
    model.eval()
    generated = list(seed_tokens)
    with torch.no_grad():
        for _ in range(length):
            inp = torch.tensor([generated[-CONTEXT_LEN:]], dtype=torch.long)
            logits = model(inp)
            probs = F.softmax(logits[0, -1], dim=0)
            generated.append(torch.multinomial(probs, 1).item())
    model.train()
    return decode(generated[len(seed_tokens):])

# ---------------------------------------------------------------------------
# Baseline batch function (normal, for reference)
# ---------------------------------------------------------------------------
def get_batch_normal(batch_size):
    ix = torch.randint(0, len(data) - CONTEXT_LEN - 1, (batch_size,))
    x = torch.stack([data[i : i + CONTEXT_LEN] for i in ix])
    y = torch.stack([data[i + 1 : i + CONTEXT_LEN + 1] for i in ix])
    return x, y

# ---------------------------------------------------------------------------
# Experiment 1: Shuffled sequences
# The characters within each training window are randomly permuted.
# The model sees the same characters but in random order — no sequential
# pattern to learn. Loss can still decrease (it learns unigram/bigram
# frequencies) but generation will be incoherent.
# ---------------------------------------------------------------------------
def get_batch_shuffled(batch_size):
    x, y = get_batch_normal(batch_size)
    # Shuffle each sequence independently
    for i in range(batch_size):
        perm = torch.randperm(CONTEXT_LEN)
        x[i] = x[i][perm]
        y[i] = y[i][perm]
    return x, y

# ---------------------------------------------------------------------------
# Experiment 2: Truncated context
# Only show the model the last 4 characters instead of 32.
# The rest of the context is zeroed out. The model has almost no history
# to predict from — it can only learn very local patterns.
# ---------------------------------------------------------------------------
def get_batch_truncated(batch_size, keep=4):
    x, y = get_batch_normal(batch_size)
    # Zero out everything except the last `keep` positions
    x[:, :-keep] = 0
    return x, y

# ---------------------------------------------------------------------------
# Experiment 3: Corrupted prefixes
# Replace the first half of each context window with random characters.
# The model sees garbage followed by real text. During generation, the
# context is self-generated (no garbage), so the model encounters a
# distribution mismatch.
# ---------------------------------------------------------------------------
def get_batch_corrupted(batch_size):
    x, y = get_batch_normal(batch_size)
    corrupt_len = CONTEXT_LEN // 2
    x[:, :corrupt_len] = torch.randint(0, vocab_size, (batch_size, corrupt_len))
    return x, y

# ---------------------------------------------------------------------------
# Run all experiments
# ---------------------------------------------------------------------------
experiments = [
    ("BASELINE (normal training)", get_batch_normal),
    ("EXP 1: Shuffled sequences",  get_batch_shuffled),
    ("EXP 2: Truncated context (last 4 chars only)", get_batch_truncated),
    ("EXP 3: Corrupted prefixes (first half = noise)", get_batch_corrupted),
]

seed = data[:CONTEXT_LEN].tolist()
results = []

for name, batch_fn in experiments:
    print(f"\n{'='*70}")
    print(f" {BOLD}{name}{RESET}")
    print(f"{'='*70}")

    model = CausalMLP_LM(vocab_size, CONTEXT_LEN, EMBED_DIM, HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    losses = []
    for step in range(1, STEPS + 1):
        x, y = batch_fn(128)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % LOG_EVERY == 0 or step == 1:
            print(f"  Step {step:>4d} | Loss: {loss.item():.4f}")

    # Generate samples
    print(f"\n  {BOLD}Generations:{RESET}")
    for i in range(3):
        sample = generate(model, seed)
        # Show first 80 chars for readability
        preview = sample[:80].replace('\n', '\\n')
        print(f"    Sample {i+1}: {preview}")

    final_loss = sum(losses[-10:]) / 10
    results.append((name, losses[0], final_loss))

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print(f"\n\n{'='*70}")
print(f" {BOLD}SUMMARY{RESET}")
print(f"{'='*70}")
print(f"  {'Experiment':<45} {'Start':>8} {'Final':>8}  {'Loss ↓?':>7}")
print(f"  {'-'*45} {'-'*8} {'-'*8}  {'-'*7}")
for name, start, final in results:
    decreased = f"{GREEN}YES{RESET}" if final < start * 0.5 else f"{RED}barely{RESET}"
    print(f"  {name:<45} {start:>8.3f} {final:>8.3f}  {decreased}")

# ---------------------------------------------------------------------------
# Write-up
# ---------------------------------------------------------------------------
print(f"""
{'='*70}
 {BOLD}WHAT THE LOSS ACTUALLY MEASURES VS WHAT WE WANT{RESET}
{'='*70}

{BOLD}What cross-entropy loss measures:{RESET}
  "Given this exact input context, how well does the model predict the
  next token?" It's a per-step, teacher-forced metric. The model always
  sees the REAL previous tokens during training — never its own outputs.

{BOLD}What we actually want:{RESET}
  Coherent, structured text over many steps of autoregressive generation,
  where the model must consume its own (possibly wrong) predictions.

{BOLD}Why these break:{RESET}
  - {BOLD}Shuffled:{RESET} Loss decreases because the model learns character
    frequencies (e.g., 'e' is common after many contexts). But generation
    is nonsense because there's no sequential pattern to reproduce.

  - {BOLD}Truncated:{RESET} Loss decreases because even 4 characters give some
    local signal ("th" → "e"). But with no long-range context, the model
    can't maintain coherence beyond a few characters.

  - {BOLD}Corrupted:{RESET} Loss decreases because the model learns to ignore the
    noisy prefix and focus on the clean suffix. But during generation,
    it sees clean text everywhere — a distribution it wasn't trained on
    — causing unexpected behavior.

{BOLD}The lesson:{RESET}
  Low training loss is necessary but NOT sufficient for good generation.
  The loss measures one-step prediction under ideal conditions. Generation
  quality depends on the model's robustness to its own errors and the
  match between training and inference distributions.
""")
