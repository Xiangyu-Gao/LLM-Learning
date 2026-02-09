"""
Step 4: Causal masking intuition — training without mask leaks the future.

Builds a tiny attention-based LM and trains it twice:
  1. WITH causal mask   (can only see past + present)
  2. WITHOUT causal mask (can see entire sequence including future)

Then compares loss curves and generation quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Day1'))
sys.path.insert(0, os.path.dirname(__file__))

from train_tiny_char_level_lm import (
    vocab_size, encode, decode, data, CONTEXT_LEN,
)
from step1_scaled_dot_product_attention import (
    scaled_dot_product_attention, make_causal_mask,
)

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Tiny attention-based LM
# ---------------------------------------------------------------------------
class AttentionLM(nn.Module):
    """
    Single-layer, single-head attention LM.
    The `use_causal_mask` flag is the only difference between the two models.
    """
    def __init__(self, vocab_size, context_len, embed_dim, head_dim, use_causal_mask=True):
        super().__init__()
        self.use_causal_mask = use_causal_mask
        self.context_len = context_len
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(context_len, embed_dim)
        # Attention projections
        self.W_q = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, head_dim, bias=False)
        # Output: project attention output back to vocab
        self.out_proj = nn.Linear(head_dim, vocab_size)

    def forward(self, x):
        B, C = x.shape
        tok_emb = self.embed(x)
        pos_emb = self.pos_embed(torch.arange(C, device=x.device))
        emb = tok_emb + pos_emb              # (B, C, E)

        Q = self.W_q(emb)                    # (B, C, D)
        K = self.W_k(emb)
        V = self.W_v(emb)

        mask = make_causal_mask(C).to(x.device) if self.use_causal_mask else None
        attn_out, self._attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        logits = self.out_proj(attn_out)      # (B, C, vocab)
        return logits

# ---------------------------------------------------------------------------
# Training + generation helpers
# ---------------------------------------------------------------------------
EMBED_DIM = 64
HEAD_DIM = 32
STEPS = 500
LR = 3e-3

def get_batch(batch_size):
    ix = torch.randint(0, len(data) - CONTEXT_LEN - 1, (batch_size,))
    x = torch.stack([data[i : i + CONTEXT_LEN] for i in ix])
    y = torch.stack([data[i + 1 : i + CONTEXT_LEN + 1] for i in ix])
    return x, y

def train(model, steps):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    losses = []
    for step in range(1, steps + 1):
        x, y = get_batch(128)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 100 == 0 or step == 1:
            print(f"    Step {step:>4d} | Loss: {loss.item():.4f}")
    return losses

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
# Run both experiments
# ---------------------------------------------------------------------------
seed = data[:CONTEXT_LEN].tolist()
results = {}

for use_mask, label in [(True, "WITH causal mask"), (False, "WITHOUT causal mask (cheating)")]:
    print(f"\n{'='*60}")
    print(f"  {BOLD}{label}{RESET}")
    print(f"{'='*60}")

    torch.manual_seed(42)  # same init for fair comparison
    model = AttentionLM(vocab_size, CONTEXT_LEN, EMBED_DIM, HEAD_DIM, use_causal_mask=use_mask)
    losses = train(model, STEPS)

    print(f"\n  {BOLD}Generations:{RESET}")
    for i in range(3):
        sample = generate(model, seed)
        preview = sample[:80].replace('\n', '\\n')
        print(f"    Sample {i+1}: {preview}")

    # Also inspect attention pattern on a real input
    model.eval()
    with torch.no_grad():
        test_input = data[:CONTEXT_LEN].unsqueeze(0)
        _ = model(test_input)
        attn_w = model._attn_weights[0]  # (C, C)

    # How much future attention?
    future_attn = torch.triu(attn_w, diagonal=1).sum().item()
    total_attn = attn_w.sum().item()
    future_pct = future_attn / total_attn * 100

    results[label] = {
        'losses': losses,
        'final_loss': sum(losses[-10:]) / 10,
        'future_pct': future_pct,
        'attn_weights': attn_w,
    }

    print(f"\n  Final loss (avg last 10): {results[label]['final_loss']:.4f}")
    print(f"  Future attention: {future_pct:.1f}% of total attention goes to future tokens")

# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------
r_causal = results["WITH causal mask"]
r_no_mask = results["WITHOUT causal mask (cheating)"]

print(f"""
{'='*60}
  {BOLD}COMPARISON{RESET}
{'='*60}
                        With mask     Without mask
  Final loss:           {r_causal['final_loss']:>10.4f}     {r_no_mask['final_loss']:>10.4f}
  Future attention:     {r_causal['future_pct']:>9.1f}%     {r_no_mask['future_pct']:>9.1f}%""")

if r_no_mask['final_loss'] < r_causal['final_loss']:
    diff = r_causal['final_loss'] - r_no_mask['final_loss']
    print(f"""
  {RED}The unmasked model has LOWER loss!{RESET}
  It's {diff:.4f} lower — but this is cheating, not learning.
  The model peeks at future tokens to "predict" them.
  At generation time, there are no future tokens to peek at.""")

print(f"""
{'='*60}
  {BOLD}KEY REALIZATION: TRAINING WITHOUT MASK LEAKS THE FUTURE{RESET}
{'='*60}

{BOLD}What the unmasked model learns:{RESET}
  To predict token t, it looks at token t+1, t+2, ... in the input.
  This is trivial — the answer is right there in the attention window.
  Loss drops fast because the task is easy (copy, don't predict).

{BOLD}Why this breaks at generation time:{RESET}
  During generation, position t has no future tokens — they don't
  exist yet. The model has learned to rely on information that
  won't be available. Result: incoherent or repetitive output.

{BOLD}What the causal mask enforces:{RESET}
  Position t can ONLY attend to positions 0..t.
  The model must genuinely learn to predict the future from the past.
  Higher training loss, but the model actually works at generation.

{BOLD}The analogy:{RESET}
  Unmasked training = studying for an exam with the answer key open.
  You "score" well but learn nothing. The causal mask takes away
  the answer key and forces real understanding.
""")
