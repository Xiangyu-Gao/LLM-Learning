"""
Step 2: Train a tiny character-level LM from scratch.
Architecture: Embeddings + 1-layer causal MLP (no RNN, no transformer).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Corpus – a Shakespeare snippet (small enough to memorize, big enough to learn)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Character-level tokenization
# ---------------------------------------------------------------------------
chars = sorted(set(CORPUS))
vocab_size = len(chars)
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

def encode(s):
    return [char_to_idx[c] for c in s]

def decode(idxs):
    return "".join(idx_to_char[i] for i in idxs)

data = torch.tensor(encode(CORPUS), dtype=torch.long)

if __name__ == "__main__":
    print(f"Corpus length: {len(CORPUS)} chars")
    print(f"Vocab size: {vocab_size}")
    print(f"Vocabulary: {''.join(chars)}")
    print()

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
CONTEXT_LEN = 32      # fixed context window
EMBED_DIM = 64        # per-character embedding dimension
HIDDEN_DIM = 256      # MLP hidden size
BATCH_SIZE = 128
LR = 3e-3
STEPS = 50
LOG_EVERY = 2

# ---------------------------------------------------------------------------
# Dataset: random context windows with teacher forcing
# ---------------------------------------------------------------------------
def get_batch(batch_size):
    """Sample random (context, target) pairs – teacher forcing."""
    ix = torch.randint(0, len(data) - CONTEXT_LEN - 1, (batch_size,))
    x = torch.stack([data[i : i + CONTEXT_LEN] for i in ix])        # (B, C)
    y = torch.stack([data[i + 1 : i + CONTEXT_LEN + 1] for i in ix])  # (B, C)
    return x, y

# ---------------------------------------------------------------------------
# Model: Embedding + 1-layer causal MLP
# ---------------------------------------------------------------------------
class CausalMLP_LM(nn.Module):
    """
    For each position t, we predict the next character using only
    characters at positions <= t (causal masking via input construction).

    At position t the model sees embeddings of chars [t-ctx+1 .. t],
    concatenated and fed through a single hidden-layer MLP.
    """

    def __init__(self, vocab_size, context_len, embed_dim, hidden_dim):
        super().__init__()
        self.context_len = context_len
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # Each position gets its own positional embedding
        self.pos_embed = nn.Embedding(context_len, embed_dim)
        # MLP: project concatenated embeddings -> hidden -> vocab logits
        self.mlp = nn.Sequential(
            nn.Linear(context_len * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, x):
        """
        x: (B, C) token indices
        returns: (B, C, vocab_size) logits for next-token prediction
        """
        B, C = x.shape
        tok_emb = self.embed(x)                              # (B, C, E)
        pos = torch.arange(C, device=x.device)
        pos_emb = self.pos_embed(pos)                        # (C, E)
        emb = tok_emb + pos_emb                              # (B, C, E)

        # For each position t, gather embeddings from a causal window
        # Pad the beginning so early positions see zeros for "before the start"
        pad = torch.zeros(B, self.context_len - 1, emb.size(2), device=x.device)
        emb_padded = torch.cat([pad, emb], dim=1)            # (B, C + ctx-1, E)

        # Unfold into windows: for each position t, grab [t .. t+ctx-1]
        # which corresponds to original positions [t - ctx + 1 .. t]
        windows = emb_padded.unfold(1, self.context_len, 1)  # (B, C, E, ctx)
        windows = windows.permute(0, 1, 3, 2)                # (B, C, ctx, E)
        windows = windows.reshape(B, C, -1)                   # (B, C, ctx*E)

        logits = self.mlp(windows)                            # (B, C, vocab)
        return logits

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = CausalMLP_LM(vocab_size, CONTEXT_LEN, EMBED_DIM, HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Architecture: Embedding({vocab_size}, {EMBED_DIM}) + PosEmbed({CONTEXT_LEN}, {EMBED_DIM})")
    print(f"             -> concat({CONTEXT_LEN}*{EMBED_DIM}) -> Linear({CONTEXT_LEN*EMBED_DIM}, {HIDDEN_DIM}) -> ReLU -> Linear({HIDDEN_DIM}, {vocab_size})")
    print()

    for step in range(1, STEPS + 1):
        x, y = get_batch(BATCH_SIZE)                     # (B, C), (B, C)
        logits = model(x)                                 # (B, C, vocab)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % LOG_EVERY == 0 or step == 1:
            print(f"Step {step:>5d} | Loss: {loss.item():.4f}")

            # ---------- autoregressive generation ----------
            model.eval()
            with torch.no_grad():
                # Seed with first few chars of corpus
                seed = data[:CONTEXT_LEN].unsqueeze(0)       # (1, ctx)
                generated = seed[0].tolist()

                for _ in range(150):
                    inp = torch.tensor([generated[-CONTEXT_LEN:]], dtype=torch.long)
                    logits = model(inp)                       # (1, ctx, vocab)
                    probs = F.softmax(logits[0, -1], dim=0)     # take the last position's logits -> (vocab,)
                    next_char = torch.multinomial(probs, 1).item()  # sample a char from the distribution
                    generated.append(next_char)

                sample = decode(generated[CONTEXT_LEN:])      # skip the seed
                print(f"  Sample: {sample!r}")
            model.train()
            print()
