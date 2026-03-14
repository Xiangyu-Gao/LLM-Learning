"""
day1_clip.py — Day 1: Mini-CLIP Contrastive Learning

Goal: understand symmetric InfoNCE loss by training lightweight projection
heads on top of a frozen CLIP backbone.

Architecture:
  image → CLIP.get_image_features() → 512-dim → Linear(512, embed_dim) → L2-norm
  text  → CLIP.get_text_features()  → 512-dim → Linear(512, embed_dim) → L2-norm
  loss  = InfoNCE(image_embs, text_embs, learnable temperature)

Key learning:
  - Why dot-product similarity + temperature controls the sharpness of matching
  - Alignment: global semantic — NOT pixel/spatial correspondence
  - Retrieval@1 as an interpretable metric
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from utils import (
    load_clip_full,
    load_image_caption_dataset,
    compute_retrieval_at_k,
    plot_similarity_matrix,
    CLIP_PROJ_DIM,
)


# ── InfoNCE loss ──────────────────────────────────────────────────────────────

class InfoNCELoss(nn.Module):
    """
    Symmetric contrastive loss used in CLIP (NT-Xent / InfoNCE).

    For a batch of B image-text pairs, the loss is:
        L = 0.5 * (CE(sim / τ, labels) + CE(sim.T / τ, labels))
    where sim[i,j] = cosine_similarity(img_i, txt_j) and labels = [0,1,...,B-1].

    Temperature τ is a learnable scalar (log_temperature for numerical stability).
    """

    def __init__(self, init_temperature=0.07):
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.tensor(init_temperature).log()
        )

    def forward(self, image_embs, text_embs):
        """
        Args:
            image_embs: (B, D) L2-normalised image embeddings
            text_embs:  (B, D) L2-normalised text  embeddings
        Returns:
            loss: scalar
            logits: (B, B) cosine similarity matrix (for logging)
        """
        # Ensure unit vectors
        image_embs = F.normalize(image_embs, dim=-1)
        text_embs  = F.normalize(text_embs,  dim=-1)

        # Temperature: clamp to avoid explosion
        temp = self.log_temperature.exp().clamp(min=1e-4, max=100.0)

        # (B, B) similarity matrix
        logits = (image_embs @ text_embs.T) / temp

        B = logits.shape[0]
        labels = torch.arange(B, device=logits.device)

        loss_i2t = F.cross_entropy(logits,   labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2, logits.detach()


# ── Projection head ───────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """Single linear projection + L2 normalisation.

    When in_dim == out_dim the weight is initialised as an identity matrix so
    that CLIP's pre-aligned features are preserved at step 0.  Training then
    fine-tunes from a good starting point rather than recovering from random
    noise, which would require far more data and steps.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        if in_dim == out_dim:
            nn.init.eye_(self.fc.weight)

    def forward(self, x):
        return F.normalize(self.fc(x), dim=-1)


# ── Dataset wrapper ───────────────────────────────────────────────────────────

class CLIPDataset(Dataset):
    """
    Wraps image-caption records for CLIP training.
    Uses CLIPProcessor to produce pixel_values + tokenised text.
    """

    def __init__(self, records, clip_processor, max_text_length=77):
        self.records = records
        self.processor = clip_processor
        self.max_text_length = max_text_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = rec["image"].resize((224, 224)).convert("RGB")
        caption = rec["caption"]

        enc = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_text_length,
            truncation=True,
        )
        return {
            "pixel_values": enc["pixel_values"].squeeze(0),
            "input_ids":    enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "caption": caption,
        }


def _as_tensor(feats):
    """
    transformers 5.x sometimes returns a dataclass from get_image/text_features.
    This helper extracts the raw tensor regardless.
    """
    if isinstance(feats, torch.Tensor):
        return feats
    # BaseModelOutputWithPooling or similar dataclass
    if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
        return feats.pooler_output
    if hasattr(feats, "last_hidden_state"):
        return feats.last_hidden_state[:, 0]
    return feats[0]   # plain tuple fallback


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(clip_model, img_proj, txt_proj, loader, device):
    """Compute image→text Recall@1 over the entire loader."""
    all_img, all_txt = [], []
    clip_model.eval(); img_proj.eval(); txt_proj.eval()

    for batch in loader:
        pv  = batch["pixel_values"].to(device)
        ids = batch["input_ids"].to(device)
        msk = batch["attention_mask"].to(device)

        img_feats = _as_tensor(clip_model.get_image_features(pixel_values=pv))
        txt_feats = _as_tensor(clip_model.get_text_features(input_ids=ids, attention_mask=msk))

        all_img.append(img_proj(img_feats).cpu())
        all_txt.append(txt_proj(txt_feats).cpu())

    img_embs = torch.cat(all_img)
    txt_embs = torch.cat(all_txt)
    sim = img_embs @ txt_embs.T
    return compute_retrieval_at_k(sim, k=1)


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n[Data]")
    records = load_image_caption_dataset(
        max_samples=args.max_samples, split="train"
    )
    import random as _random
    _random.shuffle(records)   # mix classes so train/val share the same distribution
    n_val   = min(256, len(records) // 5)
    n_train = len(records) - n_val
    train_records = records[:n_train]
    val_records   = records[n_train:]
    print(f"  Train: {len(train_records)}  Val: {len(val_records)}")

    _, processor = load_clip_full(device)   # processor only (model loaded below)

    train_ds = CLIPDataset(train_records, processor)
    val_ds   = CLIPDataset(val_records,   processor)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[Model]")
    clip_model, _ = load_clip_full(device)

    # Freeze the backbone: only the projection heads are trainable
    for p in clip_model.parameters():
        p.requires_grad_(False)

    img_proj = ProjectionHead(CLIP_PROJ_DIM, args.embed_dim).to(device)
    txt_proj = ProjectionHead(CLIP_PROJ_DIM, args.embed_dim).to(device)
    loss_fn  = InfoNCELoss(init_temperature=args.temperature).to(device)

    trainable = (
        list(img_proj.parameters())
        + list(txt_proj.parameters())
        + list(loss_fn.parameters())
    )
    n_trainable = sum(p.numel() for p in trainable)
    n_frozen    = sum(p.numel() for p in clip_model.parameters())
    print(f"  Frozen backbone: {n_frozen:,} params")
    print(f"  Trainable projection + temperature: {n_trainable:,} params")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_dl)
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\n[Training]")
    baseline = evaluate(clip_model, img_proj, txt_proj, val_dl, device)
    print(f"  Zero-shot baseline (identity proj) val_recall@1={baseline:.3f}")
    train_losses  = []
    val_recalls   = []
    step = 0

    for epoch in range(args.epochs):
        clip_model.eval(); img_proj.train(); txt_proj.train(); loss_fn.train()
        epoch_loss = 0.0

        for batch in train_dl:
            pv  = batch["pixel_values"].to(device)
            ids = batch["input_ids"].to(device)
            msk = batch["attention_mask"].to(device)

            with torch.no_grad():
                img_feats = _as_tensor(clip_model.get_image_features(pixel_values=pv))
                txt_feats = _as_tensor(clip_model.get_text_features(input_ids=ids, attention_mask=msk))

            img_embs = img_proj(img_feats)
            txt_embs = txt_proj(txt_feats)
            loss, logits = loss_fn(img_embs, txt_embs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            step += 1

            if step % max(1, len(train_dl) // 4) == 0:
                τ = loss_fn.log_temperature.exp().item()
                print(f"  Epoch {epoch+1}/{args.epochs} "
                      f"step {step} loss={loss.item():.4f} τ={τ:.4f}")

        avg_loss = epoch_loss / len(train_dl)
        recall   = evaluate(clip_model, img_proj, txt_proj, val_dl, device)
        train_losses.append(avg_loss)
        val_recalls.append(recall)
        print(f"  ── Epoch {epoch+1} avg_loss={avg_loss:.4f} val_recall@1={recall:.3f}")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    ckpt = {
        "img_proj": img_proj.state_dict(),
        "txt_proj": txt_proj.state_dict(),
        "log_temperature": loss_fn.log_temperature.item(),
        "embed_dim": args.embed_dim,
    }
    torch.save(ckpt, results_dir / "mini_clip.pt")
    print(f"\nCheckpoint saved → {results_dir}/mini_clip.pt")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_training(train_losses, val_recalls, results_dir)
    _plot_similarity(clip_model, img_proj, txt_proj, val_dl, device, results_dir)

    print(f"\nAll results saved to {results_dir}/")


def _plot_training(losses, recalls, results_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(range(1, len(losses)+1), losses, marker="o")
    ax1.set_title("InfoNCE Training Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, len(recalls)+1), recalls, marker="s", color="tab:orange")
    ax2.set_title("Validation Recall@1 (image→text)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Recall@1")
    ax2.set_ylim(0, 1); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=150)
    plt.close()
    print(f"  Plot saved → {results_dir}/training_curves.png")


@torch.no_grad()
def _plot_similarity(clip_model, img_proj, txt_proj, val_dl, device, results_dir):
    """Plot 16×16 similarity matrix from first val batch."""
    clip_model.eval(); img_proj.eval(); txt_proj.eval()
    batch = next(iter(val_dl))
    N = min(16, len(batch["pixel_values"]))

    pv  = batch["pixel_values"][:N].to(device)
    ids = batch["input_ids"][:N].to(device)
    msk = batch["attention_mask"][:N].to(device)
    captions = batch["caption"][:N]

    img_embs = img_proj(_as_tensor(clip_model.get_image_features(pixel_values=pv)))
    txt_embs = txt_proj(_as_tensor(clip_model.get_text_features(input_ids=ids, attention_mask=msk)))
    sim = img_embs @ txt_embs.T   # (N, N)

    labels = [c[:25] for c in captions]
    plot_similarity_matrix(
        sim, labels_row=[f"img {i}" for i in range(N)],
        labels_col=labels,
        title="Image–Text Cosine Similarity\n(diagonal = correct pairs)",
        path=results_dir / "similarity_matrix.png",
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Day 1: Mini-CLIP contrastive training"
    )
    p.add_argument("--config",       default=None)
    p.add_argument("--max_samples",  type=int, default=200)
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--epochs",       type=int, default=10)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--embed_dim",    type=int, default=512)
    p.add_argument("--temperature",  type=float, default=0.07)
    p.add_argument("--results_dir",  default="results/day1")
    return p.parse_args()


def apply_config(args):
    """Override argparse defaults with YAML config if --config is given."""
    if args.config is None:
        return args
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)
    return args


if __name__ == "__main__":
    args = apply_config(parse_args())
    print("=" * 60)
    print("Day 1 — Mini-CLIP Contrastive Learning")
    print("=" * 60)
    print(f"  max_samples={args.max_samples}  batch_size={args.batch_size}")
    print(f"  epochs={args.epochs}  embed_dim={args.embed_dim}")
    train(args)
