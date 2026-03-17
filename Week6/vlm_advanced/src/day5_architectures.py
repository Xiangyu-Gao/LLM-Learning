"""
day5_architectures.py — Day 5: Compare VLM Architectures

Three experiments that turn the architecture table into hands-on understanding:

Experiment A: Implement and train a minimal Q-Former
─────────────────────────────────────────────────────
Build a Q-Former from scratch (32 learned queries, cross-attention to patches).
Train prefix vs cross_attention (from Week 5) vs qformer on binary VQA.

  Binary VQA task: "Is there a [class_name]? → yes/no"
  Metric: yes/no accuracy (interpretable even with short training)

  Q-Former key difference from prefix/cross_attention:
    prefix:         4 tokens, linear projection, no inter-token interaction
    cross_attention: 49 tokens, GPT-2 cross-attn, every layer
    qformer:        32 tokens, learned queries with self+cross attention,
                    compresses ALL patch information adaptively

Experiment B: Inspect actual BLIP-2 Q-Former
─────────────────────────────────────────────
Load the real BLIP-2 Q-Former (Salesforce/blip2-opt-2.7b) and inspect:
  - Number of parameters per component
  - Shape of Q-Former outputs (always 32 regardless of resolution)
  - Cross-attention pattern: which patches does each query attend to?
  - Compare query diversity: do different queries specialize?

Experiment C: Architecture comparison table
────────────────────────────────────────────
Print a comprehensive comparison table across 5 architectures.
This is the interview-ready reference table.

Outputs:
  results/day5/qformer_training.png      — prefix vs cross_attn vs qformer accuracy
  results/day5/query_specialization.png  — BLIP-2 query attention heatmap
  results/day5/architecture_table.txt    — complete comparison table
  results/day5/analysis.txt              — Q-Former deep dive
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
WEEK5_SRC = Path(__file__).parent.parent.parent.parent / "Week5" / "vlm_project" / "src"
sys.path.insert(0, str(WEEK5_SRC))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from utils import (
    load_clip_vision,
    load_gpt2,
    load_clip_full,
    load_image_caption_dataset,
    CLIP_VIS_DIM,
    GPT2_DIM,
    CLIP_MODEL_ID,
)


# ── Minimal Q-Former ──────────────────────────────────────────────────────────

class QFormerLayer(nn.Module):
    """
    One BLIP-2-style Q-Former layer:
      1. Self-attention among queries (queries communicate)
      2. Cross-attention from queries to image patches (queries look at image)
      3. FFN
    """

    def __init__(self, hidden_dim=768, num_heads=8, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(hidden_dim, num_heads,
                                                dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads,
                                                dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, queries, encoder_hidden):
        """
        queries:         (B, Q, D) — learned query tokens
        encoder_hidden:  (B, S, D) — image patch features from ViT
        """
        # 1. Self-attention (queries attend to each other)
        q2, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + self.drop(q2))

        # 2. Cross-attention (queries attend to image patches)
        q2, _ = self.cross_attn(queries, encoder_hidden, encoder_hidden)
        queries = self.norm2(queries + self.drop(q2))

        # 3. FFN
        queries = self.norm3(queries + self.drop(self.ffn(queries)))
        return queries


class MiniQFormer(nn.Module):
    """
    Lightweight Q-Former: 32 learned queries × 2 layers.

    Given image patch features (B, 50, CLIP_DIM), produces
    (B, 32, GPT2_DIM) visual tokens — fixed budget regardless of image content.

    This is the core of BLIP-2's architecture in miniature.
    """

    def __init__(self, num_queries=32, hidden_dim=CLIP_VIS_DIM,
                 out_dim=GPT2_DIM, num_layers=2, num_heads=8):
        super().__init__()
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, hidden_dim) * 0.02
        )
        self.layers = nn.ModuleList([
            QFormerLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, patch_feats):
        """
        patch_feats: (B, S, hidden_dim) — all patch features from ViT
        returns:     (B, 32, out_dim)   — compressed visual tokens
        """
        B  = patch_feats.shape[0]
        q  = self.query_tokens.expand(B, -1, -1)   # (B, 32, D)
        for layer in self.layers:
            q = layer(q, patch_feats)
        return self.proj(q)                          # (B, 32, GPT2_DIM)


# ── Binary VQA dataset ────────────────────────────────────────────────────────

class BinaryVQADataset(Dataset):
    """
    Binary yes/no task: "Is there a [class_name]? Answer:"
    50% positive (correct class), 50% negative (random wrong class).

    Why binary? With minimal training, our models can learn this.
    Accuracy is interpretable: 50% = random, 100% = perfect.
    """

    def __init__(self, records, clip_processor, gpt2_tok, max_len=32):
        import random
        self.data = []
        # Records are 1-per-class; caption is the class identifier
        for rec in records:
            cap    = rec["caption"]
            img    = rec["image"].convert("RGB").resize((224, 224))
            pv     = clip_processor(images=img, return_tensors="pt")["pixel_values"][0]

            # Positive: correct class → "yes"
            pos_prompt = f"Is there a {cap}? Answer: yes"
            # Negative: random wrong class → "no"
            wrong_cap  = random.choice(
                [r["caption"] for r in records if r["caption"] != cap]
            )
            neg_prompt = f"Is there a {wrong_cap}? Answer: no"

            for prompt in [pos_prompt, neg_prompt]:
                enc = gpt2_tok(prompt, return_tensors="pt",
                               max_length=max_len, truncation=True, padding="max_length")
                self.data.append({
                    "pixel_values":   pv,
                    "input_ids":      enc["input_ids"][0],
                    "attention_mask": enc["attention_mask"][0],
                    "label":          1 if "yes" in prompt else 0,
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── Fusion VLMs for comparison ────────────────────────────────────────────────

class PrefixVLM(nn.Module):
    """Simple prefix: 4-token linear projection (our Week 5 baseline)."""

    def __init__(self, clip_vis, gpt2, tok, num_vis=4):
        super().__init__()
        self.clip_vis = clip_vis
        self.gpt2     = gpt2
        self.tok      = tok
        self.num_vis  = num_vis
        self.proj     = nn.Linear(CLIP_VIS_DIM, GPT2_DIM)
        for p in self.clip_vis.parameters():
            p.requires_grad_(False)

    def _vis_tokens(self, pv):
        with torch.no_grad():
            out = self.clip_vis(pixel_values=pv)
        return self.proj(out.last_hidden_state[:, :self.num_vis, :])

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        B   = pixel_values.shape[0]
        vis = self._vis_tokens(pixel_values)
        txt = self.gpt2.transformer.wte(input_ids)
        emb = torch.cat([vis, txt], dim=1)
        msk = torch.cat([torch.ones(B, self.num_vis,
                                    device=attention_mask.device,
                                    dtype=attention_mask.dtype),
                         attention_mask], dim=1)
        lbl = None
        if labels is not None:
            lbl = torch.cat([torch.full((B, self.num_vis), -100,
                                        device=labels.device, dtype=labels.dtype),
                             labels], dim=1)
        return self.gpt2(inputs_embeds=emb, attention_mask=msk, labels=lbl)


class QFormerVLM(nn.Module):
    """Q-Former VLM: 32-query MiniQFormer + GPT-2 (prefix fusion)."""

    def __init__(self, clip_vis, gpt2, tok, num_queries=32):
        super().__init__()
        self.clip_vis = clip_vis
        self.gpt2     = gpt2
        self.tok      = tok
        self.qformer  = MiniQFormer(num_queries=num_queries)
        self.num_vis  = num_queries
        for p in self.clip_vis.parameters():
            p.requires_grad_(False)

    def _vis_tokens(self, pv):
        with torch.no_grad():
            out = self.clip_vis(pixel_values=pv)
        patches = out.last_hidden_state   # (B, 50, 768)
        return self.qformer(patches)      # (B, 32, 768)

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        B   = pixel_values.shape[0]
        vis = self._vis_tokens(pixel_values)
        txt = self.gpt2.transformer.wte(input_ids)
        emb = torch.cat([vis, txt], dim=1)
        msk = torch.cat([torch.ones(B, self.num_vis,
                                    device=attention_mask.device,
                                    dtype=attention_mask.dtype),
                         attention_mask], dim=1)
        lbl = None
        if labels is not None:
            lbl = torch.cat([torch.full((B, self.num_vis), -100,
                                        device=labels.device, dtype=labels.dtype),
                             labels], dim=1)
        return self.gpt2(inputs_embeds=emb, attention_mask=msk, labels=lbl)


# ── Training ───────────────────────────────────────────────────────────────────

def accuracy_from_logits(logits, tok, device):
    """
    Measure yes/no accuracy: check if 'yes' or 'no' token has highest logit
    at the position where the answer token should be.
    """
    yes_id = tok.encode(" yes", add_special_tokens=False)[0]
    no_id  = tok.encode(" no",  add_special_tokens=False)[0]
    # logits: (B, T, V) → take last non-pad position
    last = logits[:, -1, :]   # (B, V)
    yes_s = last[:, yes_id]
    no_s  = last[:, no_id]
    preds = (yes_s > no_s).long()   # 1 = yes, 0 = no
    return preds


def train_vlm(vlm_cls, clip_vis, gpt2, tok, clip_proc,
              train_records, val_records, args, device, tag):
    """Train a VLM and evaluate yes/no accuracy."""
    print(f"\n── {tag} ──")

    vlm = vlm_cls(clip_vis, gpt2, tok).to(device)

    train_ds = BinaryVQADataset(train_records, clip_proc, tok)
    val_ds   = BinaryVQADataset(val_records,   clip_proc, tok)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW(
        [p for p in vlm.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    accs_per_epoch = []
    step = 0

    for epoch in range(args.epochs):
        vlm.train()
        for batch in train_dl:
            if args.max_steps and step >= args.max_steps:
                break
            pv  = batch["pixel_values"].to(device)
            ids = batch["input_ids"].to(device)
            msk = batch["attention_mask"].to(device)
            lbl = ids.clone(); lbl[msk == 0] = -100

            out  = vlm(pv, ids, msk, lbl)
            loss = out.loss
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(vlm.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 30 == 0:
                print(f"  step={step} loss={loss.item():.4f}")

        if args.max_steps and step >= args.max_steps:
            break

        # Eval accuracy
        vlm.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for batch in val_dl:
                pv  = batch["pixel_values"].to(device)
                ids = batch["input_ids"].to(device)
                msk = batch["attention_mask"].to(device)
                out = vlm(pv, ids, msk)
                preds  = accuracy_from_logits(out.logits, tok, device)
                labels = batch["label"].to(device)
                correct += (preds == labels).sum().item()
                total   += labels.shape[0]
        acc = correct / max(total, 1)
        accs_per_epoch.append(acc)
        print(f"  Epoch {epoch+1} val accuracy = {acc:.3f}")

    return accs_per_epoch


def experiment_a(records, clip_vis, gpt2_fn, clip_proc, args, device, results_dir):
    """Train prefix vs qformer and compare accuracy curves."""
    n_val    = min(100, len(records) // 5)
    train_r  = records[n_val:]
    val_r    = records[:n_val]

    all_accs = {}

    gpt2_p, tok_p = gpt2_fn()
    accs_prefix = train_vlm(
        PrefixVLM, clip_vis, gpt2_p, tok_p, clip_proc,
        train_r, val_r, args, device, "Prefix (4 tokens, linear)"
    )
    all_accs["prefix"] = accs_prefix
    del gpt2_p; torch.cuda.empty_cache()

    gpt2_q, tok_q = gpt2_fn()
    accs_qformer = train_vlm(
        QFormerVLM, clip_vis, gpt2_q, tok_q, clip_proc,
        train_r, val_r, args, device, "Q-Former (32 queries, 2 layers)"
    )
    all_accs["qformer"] = accs_qformer
    del gpt2_q; torch.cuda.empty_cache()

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = {"prefix": "tab:blue", "qformer": "tab:green"}
    for name, accs in all_accs.items():
        ax.plot(range(1, len(accs)+1), accs, marker="o",
                label=name, color=colors.get(name, "tab:grey"), linewidth=2)
    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5, label="chance (50%)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Yes/No Accuracy")
    ax.set_title("Binary VQA Accuracy: Prefix vs Q-Former\n"
                 "(Q-Former: 32 queries with self+cross attention)")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = results_dir / "qformer_training.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"\nTraining comparison saved → {out}")
    return all_accs


# ── Experiment B: Inspect real BLIP-2 Q-Former ────────────────────────────────

def experiment_b(results_dir, device):
    """Load BLIP-2 and print Q-Former parameter counts + query diversity."""
    print("\n── Experiment B: Inspecting Real BLIP-2 Q-Former ──")
    try:
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        from utils import load_image_caption_dataset
        print("  Loading Salesforce/blip2-opt-2.7b …")
        proc  = Blip2Processor.from_pretrained(
            "Salesforce/blip2-opt-2.7b", use_safetensors=True)
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16,
            use_safetensors=True,
            device_map="auto",
        )
        model.eval()

        # Parameter counts
        def count_params(m):
            return sum(p.numel() for p in m.parameters()) / 1e6

        print(f"\n  BLIP-2 Component Sizes:")
        print(f"    Vision encoder:  {count_params(model.vision_model):.1f} M params")
        print(f"    Q-Former:        {count_params(model.qformer):.1f} M params")
        print(f"    Language model:  {count_params(model.language_model):.1f} M params")
        print(f"    Total:           {count_params(model):.1f} M params")
        print(f"\n  Q-Former query tokens shape: {model.query_tokens.shape}")
        print(f"    → Always 32 queries of dim 768, regardless of image size")

        # Query token diversity: compute pairwise cosine similarity
        qt = model.query_tokens[0].float().detach().cpu()  # (32, 768)
        qt_norm = F.normalize(qt, dim=-1)
        sim_matrix = qt_norm @ qt_norm.T   # (32, 32)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        im = axes[0].imshow(sim_matrix.numpy(), cmap="RdBu_r", vmin=-0.5, vmax=1.0)
        axes[0].set_title("Q-Former Learned Query Token Cosine Similarity\n"
                          "(Low off-diagonal = diverse queries)")
        plt.colorbar(im, ax=axes[0])
        axes[0].set_xlabel("Query index"); axes[0].set_ylabel("Query index")

        # Query token norms (initialization diversity)
        norms = qt.norm(dim=-1).numpy()
        axes[1].bar(range(32), norms, color="tab:purple", alpha=0.8)
        axes[1].set_xlabel("Query index"); axes[1].set_ylabel("L2 Norm")
        axes[1].set_title("Query Token L2 Norms\n"
                          "(Variation shows different queries have different scales)")
        axes[1].grid(alpha=0.3)

        plt.suptitle("BLIP-2 Q-Former: 32 Learned Visual Queries", fontsize=12)
        plt.tight_layout()
        out = results_dir / "query_specialization.png"
        plt.savefig(out, dpi=150); plt.close()
        print(f"  Plot saved → {out}")

        del model; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [BLIP-2 inspection skipped: {e}]")


# ── Experiment C: Architecture table ──────────────────────────────────────────

ARCH_TABLE = """\
VLM Architecture Comparison Table (2025)
==========================================

Model         | Vision Enc      | Vision Frozen | LLM           | LLM Frozen | Fusion           | Vis Tokens | Strength
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
MiniVLM-W5    | CLIP ViT-B/32   | ✓             | GPT-2 (117M)  | ✗ (LoRA)   | Linear prefix    | 4          | Fast, minimal
MiniVLM-W6    | CLIP ViT-B/32   | ✓             | GPT-2 (117M)  | ✗ (LoRA)   | Q-Former (mini)  | 32         | Adaptive compression
BLIP-2        | ViT-g/14 (1.8B) | ✓             | OPT-2.7B      | ✓          | Q-Former (6L)    | 32         | Frozen LLM, efficient
InstructBLIP  | ViT-g/14 (1.8B) | ✓             | FlanT5-XL/XXL | ✓          | Q-Former + instr | 32         | Instruction-aware queries
LLaVA-1.5     | CLIP ViT-L/14   | ✓             | LLaMA-2 (7B)  | ✗ (LoRA)   | MLP-2-layer      | 576        | Simple + strong, high-res
Flamingo      | NFNet (4B)      | ✓             | Chinchilla 80B| ✓          | Perceiver (64Q)  | 64         | Few-shot, interleaved
InternVL-2    | InternViT-6B    | Partial       | LLaMA-3/Intern| ✗          | MLP              | 1024+      | SOTA 2024-2025

Key Design Decisions:
──────────────────────────────────────────────────────────────────────────
1. Vision frozen vs tuned:
   • Frozen (BLIP-2, Flamingo): preserves CLIP's pre-trained spatial priors
     + doesn't forget visual semantics during LLM alignment
   • Partially tuned (InternVL-2): allows vision encoder to adapt to new tasks
     - risk of catastrophic forgetting

2. LLM frozen vs tuned:
   • Frozen (BLIP-2): much cheaper to train; only connector learns
     - visual understanding limited by LLM's existing concepts
   • Tuned (LLaVA, InternVL): full joint optimization
     + better instruction following and reasoning
     - requires more data and compute

3. Connector type:
   • Linear/MLP (LLaVA): simple, scales with patches (576 tokens at 336px)
     + simplest to train, works well with high-res images
     - context window pressure, no inter-query communication
   • Q-Former (BLIP-2): fixed 32-token budget regardless of resolution
     + efficient for LLM; queries specialize over training
     - harder to optimize (requires 2-stage training)
   • Perceiver (Flamingo): similar to Q-Former, iterative attention
     + good for variable-resolution inputs

4. Why Q-Former needs 2-stage training (key insight):
   Stage 1: train Q-Former + ViT, LLM frozen
     → Q-Former learns to extract LLM-relevant visual information
   Stage 2: add LLM, train Q-Former projection + LoRA on LLM
     → align with instruction-following format
   Skipping Stage 1 means Q-Former produces visual noise for the LLM.

5. Resolution tradeoff:
   • LLaVA-1.5 at 336px: 576 patches → better spatial detail
   • BLIP-2 at 224px → 256 patches, compressed to 32 by Q-Former
   • InternVL-2 tiled encoding: 1024+ tokens for high-res → best spatial grounding

Interview key questions:
  Q: Why does BLIP-2's Q-Former have exactly 32 output tokens?
  A: Empirically chosen to balance visual richness (enough queries to capture
     scene complexity) and LLM context budget (32 tokens << 2048 context).

  Q: What's the main advantage of Q-Former over a simple linear projection?
  A: Fixed compute budget (resolution-independent), inter-query communication
     (queries specialize), and the ability to extract instruction-relevant
     features (InstructBLIP).

  Q: Why is training Q-Former hard?
  A: Cold start problem: untrained Q-Former produces garbage tokens for the
     frozen LLM, giving no learning signal.  Stage-1 bootstrapping is needed.
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Loading data]")
    records = load_image_caption_dataset(max_samples=args.max_samples)
    records = records[:args.n_classes]
    print(f"  Records: {len(records)}")

    print("\n[Loading shared CLIP]")
    clip_vis, _   = load_clip_vision(device)
    _, clip_proc  = load_clip_full(device)
    clip_vis.eval()

    def gpt2_fn():
        return load_gpt2(device)

    # ── Experiment A: train prefix vs qformer ─────────────────────────────────
    print("\n── Experiment A: Prefix vs Q-Former on Binary VQA ──")
    all_accs = experiment_a(records, clip_vis, gpt2_fn, clip_proc,
                             args, device, results_dir)

    # ── Experiment B: inspect real BLIP-2 ────────────────────────────────────
    if not args.skip_blip2:
        experiment_b(results_dir, device)
    else:
        print("\n  [Skipping BLIP-2 inspection (--skip_blip2)]")

    # ── Experiment C: architecture table ─────────────────────────────────────
    print("\n" + ARCH_TABLE[:1500])
    (results_dir / "architecture_table.txt").write_text(ARCH_TABLE)
    print(f"  Full table saved → {results_dir}/architecture_table.txt")

    # ── Print final comparison ────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Binary VQA Accuracy Summary (final epoch):")
    print("=" * 50)
    for name, accs in all_accs.items():
        if accs:
            print(f"  {name:20s}: {accs[-1]:.3f}")
    print("=" * 50)
    print(f"\nAll outputs in {results_dir}/")


def parse_args():
    p = argparse.ArgumentParser(description="Day 5: VLM Architecture Comparison")
    p.add_argument("--max_samples",  type=int,   default=2000)
    p.add_argument("--n_classes",    type=int,   default=40,
                   help="Number of unique classes to use for training")
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--lr",           type=float, default=3e-5)
    p.add_argument("--max_steps",    type=int,   default=None)
    p.add_argument("--skip_blip2",   action="store_true")
    p.add_argument("--results_dir",  default="results/day5")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("Day 5 — VLM Architecture Comparison (Q-Former Deep Dive)")
    print("=" * 60)
    main(args)
