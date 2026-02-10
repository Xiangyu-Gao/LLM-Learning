"""
Step 5: Attention is routing, not memory.

Feed the same token in different contexts, show that attention patterns
change completely. Then ask: where is "memory" actually stored?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Day1'))
sys.path.insert(0, os.path.dirname(__file__))

from train_tiny_char_level_lm import vocab_size, encode, decode, data, CONTEXT_LEN
from step1_scaled_dot_product_attention import scaled_dot_product_attention, make_causal_mask

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Trained attention LM (same as step4, train it once)
# ---------------------------------------------------------------------------
class AttentionLM(nn.Module):
    def __init__(self, vocab_size, context_len, embed_dim, head_dim):
        super().__init__()
        self.context_len = context_len
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(context_len, embed_dim)
        self.W_q = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, head_dim, bias=False)
        self.out_proj = nn.Linear(head_dim, vocab_size)

    def forward(self, x):
        B, C = x.shape
        tok_emb = self.embed(x)
        pos_emb = self.pos_embed(torch.arange(C, device=x.device))
        emb = tok_emb + pos_emb
        Q = self.W_q(emb)
        K = self.W_k(emb)
        V = self.W_v(emb)
        mask = make_causal_mask(C).to(x.device)
        attn_out, self._attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        return self.out_proj(attn_out)

EMBED_DIM = 64
HEAD_DIM = 32

print(f"{BOLD}Training attention LM...{RESET}")
torch.manual_seed(42)
model = AttentionLM(vocab_size, CONTEXT_LEN, EMBED_DIM, HEAD_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

for step in range(1, 501):
    ix = torch.randint(0, len(data) - CONTEXT_LEN - 1, (128,))
    x = torch.stack([data[i : i + CONTEXT_LEN] for i in ix])
    y = torch.stack([data[i + 1 : i + CONTEXT_LEN + 1] for i in ix])
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f"  Done. Final loss: {loss.item():.4f}\n")

# ---------------------------------------------------------------------------
# Experiment: same token "the", three different contexts
# ---------------------------------------------------------------------------
# Find positions of "t","h","e" sequences in corpus to get contexts
# where "the" appears with different surrounding text
corpus_str = decode(data.tolist())

contexts = [
    "To be, or not to be, that is the",            # "e" → expect " question"
    "The slings and arrows of outrageous fortune",  # "e" → expect ","
    "Devoutly to be wished. To die, to sleep—\nTo slee",  # "e" → expect "p"
]
# All three end with 'e' but in completely different contexts

# Pad/trim to CONTEXT_LEN
def make_input(text):
    tokens = encode(text[-CONTEXT_LEN:])
    # Pad if shorter
    while len(tokens) < CONTEXT_LEN:
        tokens = [0] + tokens
    return torch.tensor([tokens], dtype=torch.long)

print(f"{'='*65}")
print(f"{BOLD}EXPERIMENT: Same token 'e' at the final position, 3 contexts{RESET}")
print(f"{'='*65}")
print(f"All three inputs end with the same character 'e'.")
print(f"If attention were memory, the pattern would be the same.")
print()

attn_patterns = []
predictions = []

model.eval()
with torch.no_grad():
    for i, ctx in enumerate(contexts):
        inp = make_input(ctx)
        logits = model(inp)
        attn_w = model._attn_weights[0, -1]  # last position's attention, (C,)
        probs = F.softmax(logits[0, -1], dim=0)
        top5 = torch.topk(probs, 5)

        attn_patterns.append(attn_w)
        predictions.append((probs, top5))

        print(f"{BOLD}Context {i+1}:{RESET} \"{ctx}\"")
        print(f"  Last token: '{ctx[-1]}'")
        print()

        # Show where the last position attends
        # Only show last 10 positions for readability
        show_len = min(15, CONTEXT_LEN)
        ctx_chars = list(ctx[-show_len:])
        weights = attn_w[-show_len:].tolist()

        print(f"  Attention from last position (last {show_len} chars):")
        # Visual bar chart
        max_w = max(weights)
        for j, (ch, w) in enumerate(zip(ctx_chars, weights)):
            bar_len = int(w / max_w * 30) if max_w > 0 else 0
            bar = "█" * bar_len
            color = GREEN if w > 0.1 else DIM
            print(f"    '{ch}' {color}{bar} {w:.3f}{RESET}")

        print(f"\n  Top 5 next-token predictions:")
        for j in range(5):
            idx = top5.indices[j].item()
            p = top5.values[j].item()
            ch = decode([idx])
            print(f"    '{ch}' ({p:.3f})")
        print()

# ---------------------------------------------------------------------------
# Compare attention patterns across contexts
# ---------------------------------------------------------------------------
print(f"{'='*65}")
print(f"{BOLD}ATTENTION PATTERN COMPARISON{RESET}")
print(f"{'='*65}")
print(f"Same final token 'e' — are the attention patterns the same?\n")

for i in range(3):
    for j in range(i+1, 3):
        cos_sim = F.cosine_similarity(attn_patterns[i].unsqueeze(0),
                                       attn_patterns[j].unsqueeze(0)).item()
        l1_dist = (attn_patterns[i] - attn_patterns[j]).abs().sum().item()
        print(f"  Context {i+1} vs {j+1}:  cosine_sim={cos_sim:.3f}  L1_dist={l1_dist:.3f}")

print(f"\n  {BOLD}Result:{RESET} The patterns are different despite the same token.")
print(f"  Attention routes information differently based on context.")

# ---------------------------------------------------------------------------
# Where the model stores its predictions
# ---------------------------------------------------------------------------
print(f"""
{'='*65}
{BOLD}EXPERIMENT 2: Where is "memory" stored?{RESET}
{'='*65}
""")

# Show what lives in each component
with torch.no_grad():
    inp = make_input(contexts[0])
    tok_emb = model.embed(inp)
    pos_emb = model.pos_embed(torch.arange(CONTEXT_LEN))
    emb = tok_emb + pos_emb

    Q = model.W_q(emb)
    K = model.W_k(emb)
    V = model.W_v(emb)

    # The V vectors are what attention selects from
    # Show that V for the same token at different positions differs
    # because of positional embeddings
    e_token = encode('e')[0]
    e_positions = [i for i in range(CONTEXT_LEN) if inp[0, i].item() == e_token]

    if len(e_positions) >= 2:
        p1, p2 = e_positions[0], e_positions[-1]
        v1 = V[0, p1]
        v2 = V[0, p2]
        cos = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
        print(f"  Token 'e' appears at positions {e_positions}")
        print(f"  V vector at pos {p1} vs pos {p2}: cosine_sim = {cos:.3f}")
        print(f"  Same token, different V vectors — because position embeddings differ.")
        print()

# Show that the weights encode all "knowledge"
param_count = sum(p.numel() for p in model.parameters())
embed_params = model.embed.weight.numel() + model.pos_embed.weight.numel()
attn_params = sum(p.numel() for p in [model.W_q.weight, model.W_k.weight, model.W_v.weight])
out_params = model.out_proj.weight.numel() + model.out_proj.bias.numel()

print(f"  {BOLD}Where knowledge lives in this model:{RESET}")
print(f"    Embeddings (token + position): {embed_params:>6,} params  ← what tokens 'mean'")
print(f"    W_q, W_k, W_v projections:     {attn_params:>6,} params  ← what to attend to")
print(f"    Output projection:              {out_params:>6,} params  ← map attention output → prediction")
print(f"    Total:                          {param_count:>6,} params")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"""
{'='*65}
{BOLD}STEP 5 SUMMARY: ATTENTION IS ROUTING, NOT MEMORY{RESET}
{'='*65}

{BOLD}What attention does:{RESET}
  It ROUTES — decides which past tokens to pull information from
  for the current prediction. The routing changes every time based
  on the current input. Nothing is "remembered" between calls.

{BOLD}What attention does NOT do:{RESET}
  It does NOT store facts. It has no persistent state.
  After processing "the cat sat", there's no variable holding
  "subject=cat" or "verb=sat". That information exists only
  as weighted combinations of value vectors, recomputed fresh
  each forward pass.

{BOLD}Where is "memory" actually stored?{RESET}
  1. {GREEN}Weights (W_q, W_k, W_v, embeddings){RESET} — learned during training.
     These encode WHAT patterns to look for and HOW to route.
     This is long-term memory (survives across all inputs).

  2. {GREEN}KV cache (at inference){RESET} — stores K and V vectors of past tokens
     so they don't need recomputation. This is short-term memory
     within a single sequence, but it's just a performance cache,
     not a learned memory.

  3. {RED}Attention weights themselves{RESET} — NOT memory. They're computed
     fresh every forward pass, discarded immediately. They're
     routing decisions, not stored information.

{BOLD}Why can attention "forget"?{RESET}
  - Fixed context window: tokens beyond the window are gone forever.
  - Soft attention: every token competes for attention weight.
    As the sequence grows, early tokens get diluted.
  - No recurrence: unlike RNNs, there's no hidden state carried
    forward. If a fact isn't in the current window, it's lost.

{BOLD}Why this matters for agents:{RESET}
  When you use an LLM as an agent with a long conversation:
  - It can't "remember" tool results from 10k tokens ago
    if they've left the context window.
  - It re-reads the ENTIRE context every single forward pass.
  - "Memory" must be managed externally (retrieval, summarization).
  - The model doesn't accumulate knowledge within a conversation —
    it just routes differently through whatever context it can see.
""")
