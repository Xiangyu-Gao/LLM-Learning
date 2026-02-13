"""
Step 1: Inspect representations layer by layer.

Build a multi-layer transformer and watch how token representations
evolve from raw embeddings through each layer to final logits.

Architecture:
  Embedding + PosEmbed → [TransformerBlock × N_LAYERS] → LayerNorm → Linear → logits
  Each TransformerBlock = LayerNorm → MultiHeadAttention → residual
                        + LayerNorm → FFN → residual

Key measurements at each layer:
  1. Cosine similarity between tokens (do they converge or diverge?)
  2. Token entropy (how "sharp" are the representations?)
  3. Representation norms (do they grow or shrink?)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Day1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Day2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Day3'))

from train_tiny_char_level_lm import vocab_size, encode, decode, data, CONTEXT_LEN
from step1_scaled_dot_product_attention import make_causal_mask
from step1_multihead_attention import MultiHeadAttention

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Transformer block (Pre-Norm style: LN → Attn → residual, LN → FFN → residual)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x, mask=None):
        # Pre-norm attention + residual
        attn_out, attn_weights = self.attn(self.ln1(x), mask)
        x = x + attn_out
        # Pre-norm FFN + residual
        x = x + self.ffn(self.ln2(x))
        return x, attn_weights

# ---------------------------------------------------------------------------
# Multi-layer transformer LM
# ---------------------------------------------------------------------------
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_len, embed_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        self.context_len = context_len
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(context_len, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, C = x.shape
        emb = self.embed(x) + self.pos_embed(torch.arange(C, device=x.device))
        mask = make_causal_mask(C).to(x.device)

        h = emb
        for block in self.blocks:
            h, _ = block(h, mask)

        h = self.ln_final(h)
        return self.out_proj(h)

    def forward_with_intermediates(self, x):
        """Run forward pass but capture hidden states at every layer."""
        B, C = x.shape
        emb = self.embed(x) + self.pos_embed(torch.arange(C, device=x.device))
        mask = make_causal_mask(C).to(x.device)

        hidden_states = [emb.detach()]  # layer 0 = embedding output
        attn_weights_all = []

        h = emb
        for block in self.blocks:
            h, attn_w = block(h, mask)
            hidden_states.append(h.detach())
            attn_weights_all.append(attn_w.detach())

        h_final = self.ln_final(h)
        hidden_states.append(h_final.detach())  # post-LayerNorm

        return self.out_proj(h_final), hidden_states, attn_weights_all

# ---------------------------------------------------------------------------
# Config & training
# ---------------------------------------------------------------------------
EMBED_DIM = 64
NUM_HEADS = 4
FF_DIM = 128
NUM_LAYERS = 4
STEPS = 1200

if __name__ == "__main__":
    print(f"{BOLD}Training {NUM_LAYERS}-layer transformer LM...{RESET}")
    print(f"  embed_dim={EMBED_DIM}, heads={NUM_HEADS}, ff_dim={FF_DIM}, layers={NUM_LAYERS}")
    print()

    torch.manual_seed(42)
    model = TransformerLM(vocab_size, CONTEXT_LEN, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

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
        if step % 300 == 0:
            print(f"  Step {step:>4d} | Loss: {loss.item():.4f}")

    print()

    # -------------------------------------------------------------------
    # Extract hidden states for a probe sentence
    # -------------------------------------------------------------------
    probe_text = "To be, or not to be, that is the"
    tokens = encode(probe_text)
    char_labels = list(probe_text)

    x = torch.tensor([tokens], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        logits, hidden_states, attn_weights_all = model.forward_with_intermediates(x)

    # hidden_states: list of (1, C, E) tensors, len = num_layers + 2
    #   [0] = embedding, [1..num_layers] = after each block, [num_layers+1] = after final LN
    layer_names = ["Embed"] + [f"Layer {i+1}" for i in range(NUM_LAYERS)] + ["Final LN"]

    print(f"{'='*65}")
    print(f"{BOLD}PROBE SENTENCE: \"{probe_text}\"{RESET}")
    print(f"  Tokens: {len(tokens)} characters")
    print(f"  Hidden states captured at {len(hidden_states)} points")
    print()

    # -------------------------------------------------------------------
    # Measurement 1: Cosine similarity between all token pairs at each layer
    # -------------------------------------------------------------------
    print(f"{'='*65}")
    print(f"{BOLD}1. COSINE SIMILARITY BETWEEN TOKENS (per layer){RESET}")
    print(f"{'='*65}")
    print(f"  Do tokens become more similar (converge) or more distinct (diverge)?")
    print()

    cos_sim_matrices = []
    avg_cos_sims = []

    for layer_idx, hs in enumerate(hidden_states):
        h = hs[0]  # (C, E)
        h_norm = F.normalize(h, dim=-1)  # unit vectors
        sim = h_norm @ h_norm.T  # (C, C) cosine similarity
        cos_sim_matrices.append(sim.numpy())

        # Average off-diagonal similarity
        mask = ~torch.eye(h.size(0), dtype=torch.bool)
        avg_sim = sim[mask].mean().item()
        avg_cos_sims.append(avg_sim)

        print(f"  {layer_names[layer_idx]:>10}: avg pairwise cosine sim = {avg_sim:.4f}")

    print()
    trend = "CONVERGING" if avg_cos_sims[-1] > avg_cos_sims[0] else "DIVERGING"
    print(f"  Trend: representations are {BOLD}{trend}{RESET}")
    print(f"    Embed: {avg_cos_sims[0]:.4f} → Final: {avg_cos_sims[-1]:.4f}")
    print()

    # -------------------------------------------------------------------
    # Measurement 2: Representation norm per layer
    # -------------------------------------------------------------------
    print(f"{'='*65}")
    print(f"{BOLD}2. REPRESENTATION NORMS (per layer){RESET}")
    print(f"{'='*65}")
    print(f"  How large are the hidden state vectors?")
    print()

    avg_norms = []
    for layer_idx, hs in enumerate(hidden_states):
        h = hs[0]  # (C, E)
        norms = h.norm(dim=-1)  # (C,)
        avg_norm = norms.mean().item()
        std_norm = norms.std().item()
        avg_norms.append(avg_norm)
        print(f"  {layer_names[layer_idx]:>10}: avg norm = {avg_norm:.3f}  (std = {std_norm:.3f})")

    print()

    # -------------------------------------------------------------------
    # Measurement 3: Token entropy per layer
    # -------------------------------------------------------------------
    print(f"{'='*65}")
    print(f"{BOLD}3. TOKEN PREDICTION ENTROPY (per layer){RESET}")
    print(f"{'='*65}")
    print(f"  Project each layer's hidden state to logits and measure entropy.")
    print(f"  Low entropy = model is confident about next token.")
    print()

    layer_entropies = []
    for layer_idx, hs in enumerate(hidden_states):
        h = hs[0]  # (C, E)
        with torch.no_grad():
            h_normed = model.ln_final(h.unsqueeze(0))
            layer_logits = model.out_proj(h_normed)[0]  # (C, V)
            probs = F.softmax(layer_logits, dim=-1)
            entropy = -(probs * probs.log().clamp(min=-100)).sum(dim=-1)  # (C,)
            avg_entropy = entropy.mean().item()
            layer_entropies.append(avg_entropy)

        max_entropy = np.log(vocab_size)
        bar_len = int(30 * avg_entropy / max_entropy)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {layer_names[layer_idx]:>10}: {avg_entropy:.3f} / {max_entropy:.2f}  {bar}")

    print()
    print(f"  Entropy drops from {layer_entropies[0]:.3f} → {layer_entropies[-1]:.3f}")
    print(f"  The model gets progressively more confident about predictions.")
    print()

    # -------------------------------------------------------------------
    # Measurement 4: How individual token representations evolve
    # -------------------------------------------------------------------
    print(f"{'='*65}")
    print(f"{BOLD}4. REPRESENTATION EVOLUTION FOR SPECIFIC TOKENS{RESET}")
    print(f"{'='*65}")
    print(f"  Track cosine similarity between adjacent layers for each token.")
    print(f"  Low similarity = big change at that layer.")
    print()

    C = len(tokens)
    positions = [0, C // 4, C // 2, 3 * C // 4, C - 1]
    pos_labels = {p: f"pos {p} ('{char_labels[p]}')" for p in positions}

    print(f"  {'Layer transition':<20}", end="")
    for p in positions:
        print(f"  {pos_labels[p]:>16}", end="")
    print()
    print(f"  {'-'*20}", end="")
    for _ in positions:
        print(f"  {'-'*16}", end="")
    print()

    for layer_idx in range(len(hidden_states) - 1):
        h_curr = hidden_states[layer_idx][0]
        h_next = hidden_states[layer_idx + 1][0]
        trans = f"{layer_names[layer_idx]:>8} → {layer_names[layer_idx+1]:<8}"
        print(f"  {trans:<20}", end="")
        for p in positions:
            sim = F.cosine_similarity(
                h_curr[p].unsqueeze(0), h_next[p].unsqueeze(0)
            ).item()
            if sim > 0.95:
                color = GREEN
            elif sim > 0.8:
                color = YELLOW
            else:
                color = RED
            print(f"  {color}{sim:>16.4f}{RESET}", end="")
        print()

    print()
    print(f"  {DIM}Green = representation barely changed, Red = significant transformation{RESET}")
    print()

    # -------------------------------------------------------------------
    # Measurement 5: What does each layer "predict"?
    # -------------------------------------------------------------------
    print(f"{'='*65}")
    print(f"{BOLD}5. LAYER-BY-LAYER PREDICTIONS (last position){RESET}")
    print(f"{'='*65}")
    print(f"  If the model stopped at each layer, what would it predict next?")
    print(f"  Probe: \"{probe_text}\" → next char?")
    print()

    for layer_idx, hs in enumerate(hidden_states):
        h = hs[0]
        with torch.no_grad():
            h_normed = model.ln_final(h.unsqueeze(0))
            layer_logits = model.out_proj(h_normed)[0]
            last_logits = layer_logits[-1]
            probs = F.softmax(last_logits, dim=0)
            top5 = torch.topk(probs, 5)

        top5_str = ", ".join(
            f"'{decode([idx.item()])}' ({prob.item():.2f})"
            for idx, prob in zip(top5.indices, top5.values)
        )
        print(f"  {layer_names[layer_idx]:>10}: {top5_str}")

    with torch.no_grad():
        final_probs = F.softmax(logits[0, -1], dim=0)
        pred_char = decode([torch.argmax(final_probs).item()])
    print(f"\n  Final prediction: '{pred_char}'")
    print()

    # -------------------------------------------------------------------
    # Visualization: 4-panel figure
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Representation Evolution Across Transformer Layers", fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(range(len(avg_cos_sims)), avg_cos_sims, 'o-', color='steelblue', linewidth=2, markersize=6)
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Avg Pairwise Cosine Similarity")
    ax.set_title("Token Convergence/Divergence")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(range(len(layer_entropies)), layer_entropies, 's-', color='darkorange', linewidth=2, markersize=6)
    ax.axhline(y=np.log(vocab_size), color='gray', linestyle='--', alpha=0.5, label=f'Max entropy ({np.log(vocab_size):.1f})')
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Avg Prediction Entropy")
    ax.set_title("Prediction Confidence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    n_tokens = len(char_labels)
    im = ax.imshow(cos_sim_matrices[0], cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_title("Token Similarity: Embedding Layer")
    ax.set_xticks(range(n_tokens))
    ax.set_xticklabels(char_labels, fontsize=6)
    ax.set_yticks(range(n_tokens))
    ax.set_yticklabels(char_labels, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    im = ax.imshow(cos_sim_matrices[-1], cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_title("Token Similarity: Final Layer")
    ax.set_xticks(range(n_tokens))
    ax.set_xticklabels(char_labels, fontsize=6)
    ax.set_yticks(range(n_tokens))
    ax.set_yticklabels(char_labels, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'representation_evolution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization → {save_path}")
    print()

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print(f"""
{'='*65}
{BOLD}STEP 1 SUMMARY: INSPECTING REPRESENTATIONS LAYER BY LAYER{RESET}
{'='*65}

{BOLD}Architecture:{RESET}
  {NUM_LAYERS}-layer pre-norm transformer (Embed → [{NUM_LAYERS}× Block] → LN → Linear)
  Each block: LN → MultiHeadAttn → residual + LN → FFN → residual

{BOLD}What we measured:{RESET}

  1. {CYAN}Cosine similarity between tokens{RESET}
     Embed: {avg_cos_sims[0]:.4f} → Final: {avg_cos_sims[-1]:.4f}
     Tokens {'converge' if avg_cos_sims[-1] > avg_cos_sims[0] else 'diverge'} as depth increases.
     {'Early layers: tokens are relatively distinct (each char is its own embedding).' if avg_cos_sims[0] < avg_cos_sims[-1] else 'Later layers refine representations to be more context-specific.'}
     {'Later layers: tokens absorb context and become more similar.' if avg_cos_sims[-1] > avg_cos_sims[0] else ''}

  2. {CYAN}Prediction entropy{RESET}
     Embed: {layer_entropies[0]:.3f} → Final: {layer_entropies[-1]:.3f}  (max: {np.log(vocab_size):.1f})
     The model gets more confident with each layer.
     Early layers: near-uniform guessing (high entropy).
     Final layer: sharp predictions (low entropy).

  3. {CYAN}Layer-to-layer change{RESET}
     The biggest representational shifts happen in the early layers.
     Later layers make smaller, refinement-level adjustments.
     This is because residual connections preserve most information.

{BOLD}The big picture — what each depth level does:{RESET}

  {GREEN}Early layers (1-2):{RESET}  Local/syntactic patterns
    - Character identity and position encoding
    - Adjacent character dependencies (bigrams, trigrams)
    - The representation changes the most here

  {YELLOW}Middle layers (2-3):{RESET}  Relational structure
    - Longer-range dependencies (word-level patterns)
    - Context integration across the full sequence
    - Tokens that are contextually related become more similar

  {RED}Later layers (3-4):{RESET}  Task-specific abstraction
    - Representations optimized for next-character prediction
    - Entropy drops sharply — the model "decides" what comes next
    - Small refinements, not dramatic changes

{BOLD}Key insight:{RESET}
  A transformer doesn't just transform input once — it iteratively
  refines representations. Each layer reads the SAME sequence but
  with richer context. The residual stream acts as a "memory bus"
  that layers read from and write to, building up from raw tokens
  to task-ready representations.
""")
