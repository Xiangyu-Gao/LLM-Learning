"""
day2_vit.py — Day 2: Patch Tokens & ViT Attention

Experiments:
  1. Attention rollout across all 12 transformer layers → CLS saliency map.
     Visualise early vs mid vs late layer attention and the final rollout.
  2. Patch shuffle experiment: progressively shuffle 32×32 patch blocks in
     the input image; measure how CLIP similarity to the original caption drops.

Key insight to internalise:
  - Image patches are NOT equivalent to word tokens:
      • Word tokens carry discrete symbolic meaning independently of position.
      • Patch tokens carry local texture/colour, and their spatial arrangement
        is critical for scene understanding.
  - ViT has weaker spatial inductive bias than CNNs: no convolution, no
    translation equivariance — only learned positional embeddings.
  - CLIP still degrades under patch shuffling because global layout cues
    (sky above, ground below) are destroyed.
"""

import argparse
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from utils import (
    load_clip_full,
    load_clip_vision,
    load_image_caption_dataset,
    attention_rollout,
    overlay_attention_on_image,
)


# ── Patch shuffle ─────────────────────────────────────────────────────────────

def shuffle_patches(image, patch_size=32, fraction=1.0, seed=42):
    """
    Randomly shuffle a fraction of 32×32 patch blocks in a PIL image.

    Args:
        image:      PIL.Image, already resized to 224×224.
        patch_size: pixels per patch side (32 for CLIP ViT-B/32).
        fraction:   proportion of patches to shuffle (0.0 = no shuffle).
        seed:       RNG seed for reproducibility.

    Returns:
        PIL.Image with patches shuffled in-place.
    """
    rng = random.Random(seed)
    W, H = image.size
    nx = W // patch_size
    ny = H // patch_size

    # Extract all patches
    patches = []
    coords  = []
    for py in range(ny):
        for px in range(nx):
            box = (px * patch_size, py * patch_size,
                   (px + 1) * patch_size, (py + 1) * patch_size)
            patches.append(image.crop(box))
            coords.append(box)

    # Choose which patches to shuffle
    n_total   = len(patches)
    n_shuffle = int(n_total * fraction)
    idx_to_shuffle = rng.sample(range(n_total), n_shuffle)
    vals = [patches[i] for i in idx_to_shuffle]
    rng.shuffle(vals)
    for i, idx in enumerate(idx_to_shuffle):
        patches[idx] = vals[i]

    # Reconstruct image
    out = image.copy()
    for patch, box in zip(patches, coords):
        out.paste(patch, box)
    return out


# ── CLIP similarity helper ────────────────────────────────────────────────────

@torch.no_grad()
def clip_image_text_similarity(clip_model, processor, image, caption, device):
    """Return scalar cosine similarity between image and caption using CLIP."""
    inputs = processor(
        images=image, text=caption,
        return_tensors="pt", padding=True, truncation=True,
    ).to(device)
    def _to_tensor(feats):
        if isinstance(feats, torch.Tensor):
            return feats
        if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            return feats.pooler_output
        return feats[0]

    img_feat = F.normalize(_to_tensor(clip_model.get_image_features(**{
        k: v for k, v in inputs.items() if k == "pixel_values"
    })), dim=-1)
    txt_feat = F.normalize(_to_tensor(clip_model.get_text_features(**{
        k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")
    })), dim=-1)
    return (img_feat * txt_feat).sum().item()


# ── Attention extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def get_all_layer_attentions(vit_model, processor, image, device):
    """
    Run a single image through the ViT and return:
      - attentions: list of 12 (1, heads, 50, 50) tensors (one per layer)
    REQUIRES vit_model loaded with for_attention=True (attn_implementation='eager').
    """
    pv = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    outputs = vit_model(pixel_values=pv, output_attentions=True)
    # Sanity check — will be None if sdpa backend was used
    if outputs.attentions is None or outputs.attentions[0] is None:
        raise RuntimeError(
            "Attention tensors are None! "
            "Reload the model with load_clip_vision(for_attention=True)."
        )
    return outputs.attentions   # tuple of 12 tensors


# ── Visualisations ────────────────────────────────────────────────────────────

def visualise_attention_layers(image, attentions, results_dir):
    """
    4-panel figure: original image + early/mid/late layer + rollout.
    """
    # Pick representative layers (0-indexed)
    n_layers = len(attentions)
    layer_ids = [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]
    layer_labels = ["Layer 1 (early)", f"Layer {n_layers//3+1} (mid-early)",
                    f"Layer {2*n_layers//3+1} (mid-late)", f"Layer {n_layers} (late)"]

    rollout = attention_rollout(list(attentions))  # (7, 7)

    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    img224 = image.resize((224, 224))
    axes[0].imshow(img224)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    for col, (lid, label) in enumerate(zip(layer_ids, layer_labels)):
        attn = attentions[lid]               # (1, heads, 50, 50)
        attn_avg = attn.float().squeeze(0).mean(0)  # (50, 50)
        cls_attn = attn_avg[0, 1:].cpu().numpy().reshape(7, 7)
        overlaid = overlay_attention_on_image(img224, cls_attn)
        axes[col + 1].imshow(overlaid)
        axes[col + 1].set_title(label, fontsize=8)
        axes[col + 1].axis("off")

    # Rollout panel
    rollout_img = overlay_attention_on_image(img224, rollout)
    axes[5].imshow(rollout_img)
    axes[5].set_title("Attention Rollout", fontsize=10)
    axes[5].axis("off")

    plt.suptitle("CLIP ViT-B/32 — CLS→Patch Attention per Layer", fontsize=11)
    plt.tight_layout()
    out = results_dir / "attention_layers.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


def patch_shuffle_experiment(clip_model, vit_model, processor, image, caption,
                             device, results_dir):
    """
    Measure CLIP similarity vs patch shuffle fraction.
    Also visualises attention rollout before and after 100% shuffle.
    """
    fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    similarities = []
    print("\n[Patch Shuffle Experiment]")

    fig_attn, axes = plt.subplots(1, len(fractions), figsize=(3 * len(fractions), 3.5))

    for col, frac in enumerate(fractions):
        shuffled = shuffle_patches(image.resize((224, 224)), fraction=frac)
        sim = clip_image_text_similarity(clip_model, processor, shuffled, caption, device)
        similarities.append(sim)
        print(f"  shuffle={frac:.0%}  sim={sim:.4f}")

        # Attention rollout on shuffled image
        try:
            atts = get_all_layer_attentions(vit_model, processor, shuffled, device)
            rollout = attention_rollout(list(atts))
            overlaid = overlay_attention_on_image(shuffled, rollout)
        except Exception:
            overlaid = shuffled

        axes[col].imshow(overlaid)
        axes[col].set_title(f"{int(frac*100)}%\n{sim:.3f}", fontsize=8)
        axes[col].axis("off")

    plt.suptitle("Attention Rollout vs Patch Shuffle Fraction", fontsize=11)
    plt.tight_layout()
    fig_attn.savefig(results_dir / "patch_shuffle_attention.png", dpi=130)
    plt.close()

    # Similarity curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([f * 100 for f in fractions], similarities, marker="o", color="tab:blue")
    ax.axhline(similarities[0], color="grey", linestyle="--", label="baseline (no shuffle)")
    ax.set_xlabel("Patches shuffled (%)")
    ax.set_ylabel("CLIP image–text cosine similarity")
    ax.set_title("CLIP Similarity vs Patch Shuffle\n"
                 "(shows how much spatial layout matters to ViT)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = results_dir / "patch_shuffle_similarity.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")

    return fractions, similarities


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n[Loading models]")
    clip_model, processor = load_clip_full(device)
    vit_model, _          = load_clip_vision(device, for_attention=True)
    clip_model.eval(); vit_model.eval()

    # ── Load a few sample images ──────────────────────────────────────────────
    print("\n[Loading sample images]")
    records = load_image_caption_dataset(max_samples=8, split="train")
    if not records:
        raise RuntimeError("No data loaded — check dataset availability.")

    # ── Attention layer visualisation (first 3 images) ───────────────────────
    print("\n[Attention Layer Visualisation]")
    for i, rec in enumerate(records[:3]):
        image   = rec["image"].resize((224, 224)).convert("RGB")
        caption = rec["caption"]
        print(f"  Image {i+1}: '{caption}'")
        with torch.no_grad():
            attentions = get_all_layer_attentions(vit_model, processor, image, device)
        img_dir = results_dir / f"image_{i+1}"
        img_dir.mkdir(exist_ok=True)
        image.save(img_dir / "original.png")
        visualise_attention_layers(image, attentions, img_dir)

    # ── Patch shuffle experiment ──────────────────────────────────────────────
    print("\n[Patch Shuffle Experiment]")
    rec     = records[0]
    image   = rec["image"].resize((224, 224)).convert("RGB")
    caption = rec["caption"]
    print(f"  Caption: '{caption}'")
    patch_shuffle_experiment(
        clip_model, vit_model, processor, image, caption, device, results_dir
    )

    print(f"\nAll results saved to {results_dir}/")


def parse_args():
    p = argparse.ArgumentParser(description="Day 2: ViT attention visualisation")
    p.add_argument("--max_samples", type=int, default=8)
    p.add_argument("--results_dir", default="results/day2")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("Day 2 — ViT Attention & Patch Tokens")
    print("=" * 60)
    main(args)
