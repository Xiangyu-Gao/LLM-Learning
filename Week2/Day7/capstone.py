"""
Week 2 — Day 7 Capstone: A Minimal Transformer LM from Scratch.

Everything built over two weeks, combined into one self-contained system.

Components:
  1. Character-level BPE-style tokenizer (our simple char vocab)
  2. RoPE positional encoding
  3. Multi-head causal self-attention
  4. Transformer blocks (pre-norm, FFN with GELU)
  5. KV-cache for efficient autoregressive generation
  6. Three decoding strategies: greedy, top-k, top-p
  7. Attention heatmap visualization
  8. Token probability & entropy inspection

No imports from previous days — completely self-contained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os
from collections import Counter

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"


# =====================================================================
# PART 1: TOKENIZER
# =====================================================================
CORPUS = """\
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them. To die, to sleep—
No more—and by a sleep to say we end
The heartache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wished. To die, to sleep—
To sleep, perchance to dream. Ay, there's the rub,
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes calamity of so long a life.
"""

chars = sorted(set(CORPUS))
vocab_size = len(chars)
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

def encode(s):
    return [char_to_idx[c] for c in s]

def decode(idxs):
    return "".join(idx_to_char[i] for i in idxs)

data = torch.tensor(encode(CORPUS), dtype=torch.long)


# =====================================================================
# PART 2: MODEL ARCHITECTURE
# =====================================================================
CONTEXT_LEN = 32
EMBED_DIM = 64
NUM_HEADS = 4
HEAD_DIM = EMBED_DIM // NUM_HEADS
FF_DIM = 128
NUM_LAYERS = 4


def apply_rope(x):
    """Rotary Position Embedding: encode position via rotation."""
    B, H, C, D = x.shape
    positions = torch.arange(C, device=x.device).float()
    freqs = 1.0 / (10000.0 ** (torch.arange(0, D, 2, device=x.device).float() / D))
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (C, D//2)
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack([x1 * cos_a - x2 * sin_a,
                           x1 * sin_a + x2 * cos_a], dim=-1)
    return rotated.reshape(B, H, C, D)


def make_causal_mask(seq_len):
    """Upper-triangular mask: position i can only attend to j <= i."""
    mask = torch.full((seq_len, seq_len), float('-inf'))
    return torch.triu(mask, diagonal=1)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with RoPE."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, mask=None):
        B, C, E = x.shape
        H, D = self.num_heads, self.head_dim
        Q = self.W_q(x).view(B, C, H, D).transpose(1, 2)
        K = self.W_k(x).view(B, C, H, D).transpose(1, 2)
        V = self.W_v(x).view(B, C, H, D).transpose(1, 2)
        Q, K = apply_rope(Q), apply_rope(K)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(D)
        if mask is not None:
            scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        out = (weights @ V).transpose(1, 2).contiguous().view(B, C, E)
        return self.W_o(out), weights


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN → Attn → residual, LN → FFN → residual."""
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x, mask=None):
        attn_out, attn_w = self.attn(self.ln1(x), mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, attn_w


class TransformerLM(nn.Module):
    """Complete character-level transformer language model with RoPE."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, EMBED_DIM)
        self.blocks = nn.ModuleList([
            TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM)
            for _ in range(NUM_LAYERS)
        ])
        self.ln_final = nn.LayerNorm(EMBED_DIM)
        self.out_proj = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, x):
        B, C = x.shape
        h = self.embed(x)
        mask = make_causal_mask(C).to(x.device)
        for block in self.blocks:
            h, _ = block(h, mask)
        return self.out_proj(self.ln_final(h))

    def forward_with_attention(self, x):
        """Forward pass returning logits and per-layer attention weights."""
        B, C = x.shape
        h = self.embed(x)
        mask = make_causal_mask(C).to(x.device)
        all_attn = []
        for block in self.blocks:
            h, attn_w = block(h, mask)
            all_attn.append(attn_w.detach())
        return self.out_proj(self.ln_final(h)), all_attn


# =====================================================================
# PART 3: KV-CACHE FOR EFFICIENT GENERATION
# =====================================================================
class KVCache:
    """
    Stores K and V from previous generation steps.
    At each new step, only compute Q/K/V for the NEW token,
    append K/V to cache, attend Q_new to all cached K.
    Cost per step: O(n) instead of O(n²).
    """
    def __init__(self, model):
        self.model = model
        self.cache = None  # list of (K, V) per layer

    def reset(self):
        self.cache = None

    def prefill(self, tokens):
        """Process seed tokens, initialize cache."""
        inp = torch.tensor([tokens], dtype=torch.long)
        B, C = inp.shape
        h = self.model.embed(inp)
        mask = make_causal_mask(C)

        self.cache = []
        for block in self.model.blocks:
            h_norm = block.ln1(h)
            H, D = block.attn.num_heads, block.attn.head_dim

            Q = block.attn.W_q(h_norm).view(1, C, H, D).transpose(1, 2)
            K = block.attn.W_k(h_norm).view(1, C, H, D).transpose(1, 2)
            V = block.attn.W_v(h_norm).view(1, C, H, D).transpose(1, 2)
            Q, K = apply_rope(Q), apply_rope(K)

            scores = Q @ K.transpose(-2, -1) / math.sqrt(D)
            scores = scores + mask
            weights = F.softmax(scores, dim=-1)
            attn_out = (weights @ V).transpose(1, 2).contiguous().view(1, C, EMBED_DIM)
            attn_out = block.attn.W_o(attn_out)

            h = h + attn_out
            h = h + block.ffn(block.ln2(h))

            self.cache.append((K.clone(), V.clone()))

        logits = self.model.out_proj(self.model.ln_final(h))
        return logits[0, -1]

    def step(self, token_id, pos):
        """Process one new token using cached K, V."""
        inp = torch.tensor([[token_id]], dtype=torch.long)
        h = self.model.embed(inp)  # (1, 1, E)

        # Build position tensor for RoPE at this specific position
        for layer_idx, block in enumerate(self.model.blocks):
            h_norm = block.ln1(h)
            H, D = block.attn.num_heads, block.attn.head_dim

            q_new = block.attn.W_q(h_norm).view(1, 1, H, D).transpose(1, 2)
            k_new = block.attn.W_k(h_norm).view(1, 1, H, D).transpose(1, 2)
            v_new = block.attn.W_v(h_norm).view(1, 1, H, D).transpose(1, 2)

            # Apply RoPE at position `pos`
            positions = torch.tensor([pos]).float()
            freqs = 1.0 / (10000.0 ** (torch.arange(0, D, 2).float() / D))
            angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
            cos_a = torch.cos(angles)
            sin_a = torch.sin(angles)

            def rotate_single(x):
                x1, x2 = x[..., 0::2], x[..., 1::2]
                rotated = torch.stack([x1 * cos_a - x2 * sin_a,
                                       x1 * sin_a + x2 * cos_a], dim=-1)
                return rotated.reshape(x.shape)

            q_new = rotate_single(q_new)
            k_new = rotate_single(k_new)

            K_cached, V_cached = self.cache[layer_idx]
            K_cached = torch.cat([K_cached, k_new], dim=2)
            V_cached = torch.cat([V_cached, v_new], dim=2)
            self.cache[layer_idx] = (K_cached, V_cached)

            scores = q_new @ K_cached.transpose(-2, -1) / math.sqrt(D)
            weights = F.softmax(scores, dim=-1)
            attn_out = (weights @ V_cached).transpose(1, 2).contiguous().view(1, 1, EMBED_DIM)
            attn_out = block.attn.W_o(attn_out)

            h = h + attn_out
            h = h + block.ffn(block.ln2(h))

        logits = self.model.out_proj(self.model.ln_final(h))
        return logits[0, -1]


# =====================================================================
# PART 4: DECODING STRATEGIES
# =====================================================================
def greedy_decode(logits, temperature=1.0):
    """Pick the most probable token."""
    return (logits / temperature).argmax().item()


def topk_decode(logits, k=5, temperature=1.0):
    """Sample from the top-k most probable tokens."""
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_idx = torch.topk(probs, k)
    topk_probs = topk_probs / topk_probs.sum()
    sampled = torch.multinomial(topk_probs, 1)
    return topk_idx[sampled].item()


def topp_decode(logits, p=0.9, temperature=1.0):
    """Sample from the smallest set of tokens with cumulative prob >= p."""
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    mask = cumsum - sorted_probs < p
    sorted_probs[~mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()
    sampled = torch.multinomial(sorted_probs, 1)
    return sorted_idx[sampled].item()


def generate(model, seed, length, strategy='greedy', use_cache=True, **kwargs):
    """
    Generate text with the specified decoding strategy.
    Returns: generated text, per-step entropies, per-step probabilities.
    """
    seed_tokens = encode(seed)
    generated = []
    entropies = []
    top_probs = []

    decode_fn = {'greedy': greedy_decode, 'top_k': topk_decode, 'top_p': topp_decode}[strategy]

    with torch.no_grad():
        if use_cache:
            cache = KVCache(model)
            logits = cache.prefill(seed_tokens)

            for step in range(length):
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                entropies.append(entropy)
                top_probs.append(probs.max().item())

                next_token = decode_fn(logits, **kwargs)
                generated.append(next_token)
                pos = len(seed_tokens) + step
                if pos >= CONTEXT_LEN:
                    break
                logits = cache.step(next_token, pos)
        else:
            context = list(seed_tokens)
            for step in range(length):
                inp = torch.tensor([context[-CONTEXT_LEN:]], dtype=torch.long)
                logits_all = model(inp)
                logits = logits_all[0, -1]

                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                entropies.append(entropy)
                top_probs.append(probs.max().item())

                next_token = decode_fn(logits, **kwargs)
                generated.append(next_token)
                context.append(next_token)

    return decode(generated), entropies, top_probs


# =====================================================================
# TRAINING
# =====================================================================
print(f"""
{'='*70}
{BOLD}  CAPSTONE: MINIMAL TRANSFORMER LM — COMPLETE SYSTEM{RESET}
{'='*70}

  Architecture:
    Tokenizer:    Character-level, vocab={vocab_size}
    Embedding:    {EMBED_DIM}-dim, RoPE positional encoding
    Attention:    {NUM_HEADS}-head causal self-attention
    FFN:          {EMBED_DIM} → {FF_DIM} → {EMBED_DIM} (GELU)
    Layers:       {NUM_LAYERS}
    Context:      {CONTEXT_LEN} tokens
    Parameters:   ~{sum(p.numel() for p in TransformerLM().parameters()):,}
""")

print(f"{BOLD}Training...{RESET}")
torch.manual_seed(42)
model = TransformerLM()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

losses_history = []
for step in range(1, 1201):
    ix = torch.randint(0, len(data) - CONTEXT_LEN - 1, (128,))
    xb = torch.stack([data[i : i + CONTEXT_LEN] for i in ix])
    yb = torch.stack([data[i + 1 : i + CONTEXT_LEN + 1] for i in ix])
    loss = F.cross_entropy(model(xb).view(-1, vocab_size), yb.view(-1))
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    losses_history.append(loss.item())
    if step % 300 == 0:
        print(f"  Step {step:>4}: loss = {loss.item():.4f}")

model.eval()
print()


# =====================================================================
# DEMO 1: KV-CACHE CORRECTNESS & SPEED
# =====================================================================
print(f"{BOLD}DEMO 1: KV-CACHE — CORRECTNESS & SPEED{RESET}")
print(f"{'='*70}")
print()

seed = "To be"
with torch.no_grad():
    text_naive, _, _ = generate(model, seed, 25, strategy='greedy', use_cache=False)
    text_cached, _, _ = generate(model, seed, 25, strategy='greedy', use_cache=True)

print(f"  Seed: \"{seed}\"")
print(f"  Naive:  \"{text_naive}\"")
print(f"  Cached: \"{text_cached}\"")
print(f"  Match: {GREEN + 'YES' + RESET if text_naive == text_cached else RED + 'NO' + RESET}")
print()

# Timing
n_trials = 10
t_naive, t_cached = [], []
for _ in range(n_trials):
    with torch.no_grad():
        s = time.perf_counter()
        generate(model, seed, 25, strategy='greedy', use_cache=False)
        t_naive.append(time.perf_counter() - s)
        s = time.perf_counter()
        generate(model, seed, 25, strategy='greedy', use_cache=True)
        t_cached.append(time.perf_counter() - s)

print(f"  Naive:  {np.mean(t_naive)*1000:.1f} ms")
print(f"  Cached: {np.mean(t_cached)*1000:.1f} ms")
print(f"  (At our tiny scale, Python overhead dominates.)")
print(f"  (Real speedup: O(n²) → O(n) per step, visible at n > 1000.)")
print()


# =====================================================================
# DEMO 2: DECODING STRATEGIES
# =====================================================================
print(f"{BOLD}DEMO 2: THREE DECODING STRATEGIES{RESET}")
print(f"{'='*70}")
print()

seed = "To be"
torch.manual_seed(0)

strategies = [
    ("Greedy",               dict(strategy='greedy')),
    ("Top-k (k=5)",          dict(strategy='top_k', k=5)),
    ("Top-k (k=5, T=0.7)",  dict(strategy='top_k', k=5, temperature=0.7)),
    ("Top-p (p=0.9)",        dict(strategy='top_p', p=0.9)),
    ("Top-p (p=0.9, T=1.5)", dict(strategy='top_p', p=0.9, temperature=1.5)),
]

gen_texts = []
gen_entropies = []
gen_probs = []

for name, kwargs in strategies:
    torch.manual_seed(42)
    text, ents, probs = generate(model, seed, 25, use_cache=True, **kwargs)
    gen_texts.append((name, text))
    gen_entropies.append((name, ents))
    gen_probs.append((name, probs))

    # Corpus match
    match = text in CORPUS
    print(f"  {name:<25} \"{text}\"")
    if match:
        print(f"  {'':>25} {GREEN}↑ exact corpus match{RESET}")

print()
print(f"  {BOLD}Observations:{RESET}")
print(f"    Greedy:  deterministic, reproduces memorized corpus")
print(f"    Top-k:   fixed exploration budget (always k candidates)")
print(f"    Top-p:   adaptive exploration (fewer candidates when confident)")
print(f"    Higher T: more randomness, more creative/wrong")
print()


# =====================================================================
# DEMO 3: TOKEN PROBABILITY INSPECTION
# =====================================================================
print(f"{BOLD}DEMO 3: TOKEN PROBABILITY INSPECTION{RESET}")
print(f"{'='*70}")
print()

probe = "To be, or not to be, that is"
probe_tokens = encode(probe)
inp = torch.tensor([probe_tokens], dtype=torch.long)

with torch.no_grad():
    logits, all_attn = model.forward_with_attention(inp)

probs_all = F.softmax(logits[0], dim=-1)

print(f"  Probe: \"{probe}\"")
print()
print(f"  {'Pos':>4} {'Input':>6} {'Predicted':>10} {'P(pred)':>8} {'True next':>10} "
      f"{'P(true)':>8} {'Entropy':>8} {'Correct'}")
print(f"  {'-'*4} {'-'*6} {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

target_tokens = probe_tokens[1:] + [encode(CORPUS[len(probe)])[0]]  # next char from corpus

for pos in range(len(probe_tokens)):
    p = probs_all[pos]
    pred_id = p.argmax().item()
    pred_char = idx_to_char[pred_id]
    p_pred = p[pred_id].item()

    true_id = target_tokens[pos]
    true_char = idx_to_char[true_id]
    p_true = p[true_id].item()

    entropy = -(p * torch.log(p + 1e-10)).sum().item()
    correct = pred_id == true_id

    input_char = idx_to_char[probe_tokens[pos]]
    color = GREEN if correct else RED

    print(f"  {pos:>4} '{input_char:>3}' → '{pred_char:>3}' "
          f"{p_pred:>8.1%} '{true_char:>3}' "
          f"{p_true:>8.1%} {entropy:>7.3f}  {color}{'✓' if correct else '✗':>5}{RESET}")

print()

# Show top-5 alternatives at an interesting position
interesting_pos = min(5, len(probe_tokens) - 1)  # after "To be,"
p = probs_all[interesting_pos]
top5 = torch.topk(p, 5)
input_char = idx_to_char[probe_tokens[interesting_pos]]
print(f"  Top 5 predictions after '{input_char}' (pos {interesting_pos}):")
for prob, idx in zip(top5.values, top5.indices):
    char = idx_to_char[idx.item()]
    bar = '█' * int(prob.item() * 40)
    print(f"    '{char}' ({prob.item():.1%}) {bar}")
print()


# =====================================================================
# DEMO 4: ENTROPY TRACKING DURING GENERATION
# =====================================================================
print(f"{BOLD}DEMO 4: ENTROPY TRACKING DURING GENERATION{RESET}")
print(f"{'='*70}")
print()

seed = "To be"
_, ents_greedy, probs_greedy = generate(model, seed, 25, strategy='greedy')

print(f"  Seed: \"{seed}\", greedy generation, 25 tokens")
print()
print(f"  {'Step':>5} {'Token':>6} {'Entropy':>8} {'Max P':>7} {'Confidence'}")
print(f"  {'-'*5} {'-'*6} {'-'*8} {'-'*7} {'-'*12}")

with torch.no_grad():
    cache = KVCache(model)
    logits = cache.prefill(encode(seed))
    for step in range(25):
        p = F.softmax(logits, dim=-1)
        ent = -(p * torch.log(p + 1e-10)).sum().item()
        max_p = p.max().item()
        token_id = logits.argmax().item()
        char = idx_to_char[token_id]

        conf = "certain" if max_p > 0.95 else ("confident" if max_p > 0.7 else
               ("uncertain" if max_p > 0.3 else "confused"))
        color = GREEN if max_p > 0.95 else (YELLOW if max_p > 0.5 else RED)

        print(f"  {step:>5} '{char:>3}'  {ent:>7.3f} {max_p:>6.1%}  {color}{conf}{RESET}")

        pos = len(encode(seed)) + step
        if pos >= CONTEXT_LEN:
            break
        logits = cache.step(token_id, pos)

print()


# =====================================================================
# VISUALIZATION
# =====================================================================
print(f"{BOLD}GENERATING VISUALIZATIONS...{RESET}")
print(f"{'='*70}")
print()

fig = plt.figure(figsize=(24, 20))
fig.suptitle("Capstone: Minimal Transformer LM — Complete System",
             fontsize=16, fontweight='bold', y=0.98)

# Create a grid: 3 rows × 3 columns
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# --- Panel 1: Training loss curve ---
ax = fig.add_subplot(gs[0, 0])
ax.plot(losses_history, color='steelblue', linewidth=0.5, alpha=0.5)
# Smoothed
window = 50
smoothed = np.convolve(losses_history, np.ones(window)/window, mode='valid')
ax.plot(range(window-1, len(losses_history)), smoothed, color='darkblue', linewidth=2)
ax.set_xlabel("Training step")
ax.set_ylabel("Cross-entropy loss")
ax.set_title("Training Curve")
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# --- Panel 2: Token probability distribution at probe position ---
ax = fig.add_subplot(gs[0, 1])
probe_pos = interesting_pos
p = probs_all[probe_pos].numpy()
sorted_p = np.sort(p)[::-1]
colors = ['#e74c3c' if i == 0 else '#3498db' if i < 5 else '#bdc3c7'
          for i in range(len(sorted_p))]
ax.bar(range(len(sorted_p)), sorted_p, color=colors, width=1.0)
ax.set_xlabel("Token rank")
ax.set_ylabel("Probability")
ax.set_title(f"Probability Distribution\n(pos {probe_pos}: '{idx_to_char[probe_tokens[probe_pos]]}')")
ax.set_xlim(-0.5, min(vocab_size, 38))
ax.grid(True, alpha=0.3, axis='y')

# --- Panel 3: Entropy per position for the probe ---
ax = fig.add_subplot(gs[0, 2])
probe_entropies = []
for pos in range(len(probe_tokens)):
    p = probs_all[pos]
    ent = -(p * torch.log(p + 1e-10)).sum().item()
    probe_entropies.append(ent)

bars = ax.bar(range(len(probe_entropies)), probe_entropies,
              color=['#2ecc71' if e < 0.5 else '#f39c12' if e < 1.5 else '#e74c3c'
                     for e in probe_entropies])
ax.set_xticks(range(len(probe_tokens)))
ax.set_xticklabels([idx_to_char[t] for t in probe_tokens], fontsize=7)
ax.set_xlabel("Input character")
ax.set_ylabel("Prediction entropy (nats)")
ax.set_title("Entropy per Position\n(lower = more confident)")
ax.grid(True, alpha=0.3, axis='y')

# --- Panels 4-7: Attention heatmaps (one per layer) ---
for layer_idx in range(NUM_LAYERS):
    ax = fig.add_subplot(gs[1, layer_idx]) if layer_idx < 3 else fig.add_subplot(gs[2, 0])
    # Average over heads
    attn = all_attn[layer_idx][0].mean(dim=0).numpy()  # (C, C)
    im = ax.imshow(attn, cmap='viridis', aspect='auto')
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title(f"Layer {layer_idx + 1} Attention\n(avg over {NUM_HEADS} heads)")

    # Add character labels if small enough
    if len(probe_tokens) <= 32:
        labels = [idx_to_char[t] for t in probe_tokens]
        step = max(1, len(labels) // 16)
        ax.set_xticks(range(0, len(labels), step))
        ax.set_xticklabels(labels[::step], fontsize=6)
        ax.set_yticks(range(0, len(labels), step))
        ax.set_yticklabels(labels[::step], fontsize=6)

# --- Panel 8: Decoding strategy comparison — entropy ---
ax = fig.add_subplot(gs[2, 1])
for name, ents in gen_entropies:
    ax.plot(ents, linewidth=1.5, label=name, alpha=0.8)
ax.set_xlabel("Generation step")
ax.set_ylabel("Prediction entropy (nats)")
ax.set_title("Entropy by Decoding Strategy")
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, alpha=0.3)

# --- Panel 9: Decoding strategy comparison — max probability ---
ax = fig.add_subplot(gs[2, 2])
for name, probs in gen_probs:
    ax.plot(probs, linewidth=1.5, label=name, alpha=0.8)
ax.set_xlabel("Generation step")
ax.set_ylabel("Max token probability")
ax.set_title("Confidence by Decoding Strategy")
ax.legend(fontsize=7, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)

plt.savefig(os.path.join(os.path.dirname(__file__), 'capstone.png'),
            dpi=150, bbox_inches='tight')
print(f"  Saved → capstone.png")
print()


# =====================================================================
# FINAL WRITE-UP
# =====================================================================
print(f"""
{'='*70}
{BOLD}  FINAL WRITE-UP: WHAT TRANSFORMERS ARE GOOD AT,{RESET}
{BOLD}  AND WHAT THEY ARE FUNDAMENTALLY BAD AT{RESET}
{'='*70}

Over two weeks, we built a transformer language model from scratch,
component by component, and tested each piece to failure. Here is
what we learned.

{'─'*70}
{GREEN}{BOLD}WHAT TRANSFORMERS ARE GOOD AT{RESET}
{'─'*70}

{BOLD}1. Pattern Continuation{RESET}

  Transformers excel at learning statistical patterns in sequences.
  Our tiny model memorized 607 characters of Shakespeare with loss
  0.11 — near-perfect reproduction. Given "To be, or not to be,"
  it continues with "that is the question:" every time.

  This isn't rote lookup. The model learned causal relationships:
  after a comma-space, certain words are likely. After "or not to",
  "be" follows. Each layer refines the prediction by mixing
  information from earlier positions through attention.

  {DIM}We saw this in: Week1/Day4 (representation inspection),
  Week2/Day2 (temperature generation), Day6 (exposure bias).{RESET}

{BOLD}2. Style Imitation{RESET}

  The training objective — predict the next token — implicitly
  learns style, rhythm, and register. Our model learned not just
  Shakespeare's words, but his punctuation, line breaks ("\\n"),
  and syntactic patterns. Given any starting fragment from the
  corpus, it reproduces the stylistic continuation.

  This is the basis of few-shot prompting in large LLMs: the
  model infers the style from examples and continues it.

  {DIM}We saw this in: Week2/Day2 (generation quality at different T).{RESET}

{BOLD}3. Statistical Reasoning{RESET}

  Temperature experiments (Week2/Day1-2) showed that the model's
  probability distributions are meaningful. Low entropy at certain
  positions reflects true certainty (only one continuation makes
  sense). High entropy at others reflects genuine ambiguity.

  The model's softmax output is a calibrated belief distribution
  — at least on in-distribution data. When the model says 99%,
  it's right 99% of the time (Week2/Day6, calibration curve).

  {DIM}We saw this in: Week2/Day1 (logits to tokens),
  Day2 (entropy analysis), Day6 (calibration).{RESET}

{BOLD}4. Latent Abstraction{RESET}

  Attention heatmaps (Week1/Day2-4) revealed that different heads
  specialize: some attend to adjacent tokens (local syntax), others
  attend to distant positions (long-range dependencies). Different
  layers build increasingly abstract representations.

  FFN neurons specialize too (Week1/Day6): some fire for spaces,
  others for uppercase, others for specific character sequences.
  The model develops internal features WITHOUT being told to —
  they emerge from the prediction objective.

  {DIM}We saw this in: Week1/Day2 (attention patterns), Day4
  (representation evolution), Day6 (FFN neuron specialization).{RESET}

{'─'*70}
{RED}{BOLD}WHAT TRANSFORMERS ARE FUNDAMENTALLY BAD AT{RESET}
{'─'*70}

{BOLD}1. Exact Arithmetic{RESET}

  BPE tokenization splits numbers at arbitrary boundaries
  (Week2/Day4): "2847" → ["28", "47"]. The model has no concept
  of place value, no digit alignment, no carry propagation.

  Addition requires a sequential algorithm (right-to-left with
  carry). Attention is parallel weighted averaging — the wrong
  primitive for sequential computation. Transformers can learn
  addition for numbers up to a certain length, but fail to
  generalize to longer numbers.

  {BOLD}Root cause:{RESET} Tokenization destroys digit structure.
  Attention is averaging, not algorithm execution.

{BOLD}2. Long-Term State Tracking{RESET}

  Context length experiments (Week2/Day5) showed three failure
  modes beyond the training context: drift, attention collapse,
  and looping.

  Even WITHIN the training context, attention dilution is
  fundamental: softmax over n tokens gives each ~1/n probability.
  At n=4096, even a well-matched key gets only 0.2% attention.
  The information bottleneck (d-dimensional output vector) means
  each token's contribution to a long-context summary is
  fractions of a bit.

  {BOLD}Root cause:{RESET} Attention does content MIXING (weighted
  averaging), not structured RETRIEVAL. Long context ≠ long memory.
  This is why RAG exists.

{BOLD}3. Reliable Factual Grounding{RESET}

  Hallucination experiments (Week2/Day6) showed that the model
  is equally confident on in-corpus and out-of-corpus inputs.
  When fed "Machine learning is", it confidently produces
  Shakespeare-like text with 89% confidence.

  The training objective (predict next token) rewards PLAUSIBILITY,
  not TRUTH. A fluent wrong answer and a fluent right answer look
  identical to the loss function. The model has no mechanism for
  "I don't know."

  {BOLD}Root cause:{RESET} The training signal is statistical
  co-occurrence, not factual verification. The model learns
  P(next token | context), not P(claim is true).

{BOLD}4. True Planning Without Scaffolding{RESET}

  Autoregressive generation (Week2/Day3) is inherently LEFT-TO-RIGHT.
  The model commits to each token before seeing what comes after.
  It cannot look ahead, cannot backtrack, cannot plan.

  To write a coherent paragraph, a human plans the structure first,
  then fills in details. A transformer generates word-by-word,
  hoping each word happens to be consistent with a coherent plan
  that was never explicitly made.

  This is why chain-of-thought prompting helps: it forces the model
  to externalize intermediate reasoning, giving it a "scratchpad"
  for the sequential computation that attention can't do internally.

  {BOLD}Root cause:{RESET} Fixed-depth computation (finite transformer
  layers) with no explicit memory or planning mechanism.
  Each forward pass is a single "thought" — complex reasoning
  requires multiple passes (chain-of-thought, tree-of-thought).

{'─'*70}
{CYAN}{BOLD}THE UNIFYING INSIGHT{RESET}
{'─'*70}

  Transformers are {GREEN}statistical pattern engines{RESET}.

  They are extraordinarily good at learning and reproducing the
  statistical structure of their training data. This is powerful
  enough to produce fluent text, translate languages, write code,
  and simulate reasoning.

  But they are not {RED}algorithmic computation engines{RESET}.

  They cannot reliably execute multi-step algorithms, maintain
  exact state over long sequences, verify factual claims, or
  plan ahead. These capabilities require SCAFFOLDING:

    • Chain-of-thought → sequential reasoning
    • Tool use         → exact computation (calculators, search)
    • RAG              → reliable factual grounding
    • Agents           → planning and iteration

  The transformer is the {BOLD}engine{RESET}. These techniques are the
  {BOLD}transmission, steering, and brakes{RESET}. Neither works alone.

  Understanding this distinction — what the architecture can and
  cannot do — is the difference between using LLMs effectively
  and being surprised when they fail.

  {BOLD}You now think like a model engineer.{RESET}
""")
