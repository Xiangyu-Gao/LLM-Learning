"""
day5_compare.py — Day 5: Frozen vs Fine-tuned Vision Encoder

Trains two variants of MiniVLM starting from the Day 3 checkpoint:
  A. freeze_vision=True   — only vision_proj + GPT-2 are updated
  B. freeze_vision=False  — CLIP ViT backbone is also updated

Metrics compared:
  1. Training loss curves (same data, same steps)
  2. CLIP feature drift: cosine similarity between original and post-training
     image features (measures catastrophic forgetting)
  3. Object-hallucination failure rate on 10 adversarial prompts

Insight to articulate:
  Frozen encoder:
    + Stable — CLIP's general visual features are preserved
    + Less compute, less risk of overfitting
    - Cannot adapt to domain shifts (e.g., synthetic images)
    - Alignment bottleneck: limited by CLIP's representation quality

  Fine-tuned encoder:
    + Can learn domain-specific features
    + Potentially better grounding for training-distribution tasks
    - Risk of catastrophic forgetting (destroys zero-shot generalization)
    - More parameters, slower, needs more data to avoid overfitting
    - Hallucination can INCREASE as the encoder overfits captions
"""

import argparse
import sys
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils import (
    load_clip_vision,
    load_clip_full,
    load_gpt2,
    load_image_caption_dataset,
    ImageCaptionDataset,
    save_vlm_checkpoint,
    COLOR_MAP,
    make_spatial_image,
    make_counting_image,
)
from day3_fusion import MiniVLM, make_dataloader
from day4_failures import (
    HALLUCINATION_QUESTIONS,
    run_hallucination_eval,
    is_failure,
)


# ── Feature drift probe ───────────────────────────────────────────────────────

@torch.no_grad()
def measure_feature_drift(original_clip_vis, trained_clip_vis,
                           clip_processor, records, device, n=50):
    """
    Compute mean cosine similarity between image features from the original
    CLIP ViT and the (potentially) fine-tuned version.

    similarity = 1.0 → no drift (frozen case)
    similarity < 1.0 → encoder has changed (fine-tuned case)
    """
    original_clip_vis.eval()
    trained_clip_vis.eval()
    sims = []
    for rec in records[:n]:
        image = rec["image"].resize((224, 224)).convert("RGB")
        pv = clip_processor(images=image, return_tensors="pt")["pixel_values"].to(device)

        feat_orig    = original_clip_vis(pixel_values=pv).pooler_output  # (1, 768)
        feat_trained = trained_clip_vis(pixel_values=pv).pooler_output   # (1, 768)

        sim = F.cosine_similarity(feat_orig, feat_trained, dim=-1).item()
        sims.append(sim)

    return float(np.mean(sims))


# ── Training one variant ──────────────────────────────────────────────────────

def train_variant(label, freeze_vision, records, clip_processor, args, device):
    """
    Train a VLM variant from scratch (same architecture as Day 3 prefix mode).
    Returns:
        vlm            — trained model
        train_losses   — list of per-step losses
    """
    print(f"\n── Variant: {label}  (freeze_vision={freeze_vision}) ────────────")

    clip_vis, _ = load_clip_vision(device)
    gpt2, tok   = load_gpt2(device, add_cross_attention=False)

    vlm = MiniVLM(
        clip_vis=clip_vis,
        gpt2=gpt2,
        tok=tok,
        fusion_mode="prefix",
        num_vis_tokens=args.num_vis_tokens,
        freeze_vision=freeze_vision,
    ).to(device)

    # Keep a snapshot of original CLIP features for drift measurement
    original_clip_vis = copy.deepcopy(clip_vis).to(device)
    for p in original_clip_vis.parameters():
        p.requires_grad_(False)

    n_val   = min(200, len(records) // 5)
    n_train = len(records) - n_val
    train_dl = make_dataloader(records[:n_train], clip_processor, tok,
                               args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        [p for p in vlm.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    train_losses = []
    step = 0
    vlm.train()

    for batch in train_dl:
        if step >= args.max_steps:
            break

        pv  = batch["pixel_values"].to(device)
        ids = batch["input_ids"].to(device)
        msk = batch["attention_mask"].to(device)
        lbl = ids.clone(); lbl[msk == 0] = -100

        out  = vlm(pixel_values=pv, input_ids=ids, attention_mask=msk, labels=lbl)
        loss = out.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vlm.parameters(), 1.0)
        optimizer.step()

        train_losses.append(loss.item())
        step += 1

        if step % 50 == 0:
            print(f"  step={step}/{args.max_steps} loss={loss.item():.4f}")

    # Feature drift
    drift_records = records[n_train:]
    drift = measure_feature_drift(
        original_clip_vis, vlm.clip_vis, clip_processor, drift_records, device
    )
    print(f"  Feature drift (cosine sim, original vs trained): {drift:.4f}")

    # Save checkpoint
    ckpt_path = Path(args.results_dir) / f"vlm-{label.replace(' ', '_')}" / "checkpoint.pt"
    save_vlm_checkpoint(
        ckpt_path, vlm,
        config_dict={
            "fusion_mode":    "prefix",
            "num_vis_tokens": args.num_vis_tokens,
            "freeze_vision":  freeze_vision,
            "label":          label,
            "feature_drift":  drift,
        },
    )

    return vlm, tok, train_losses, drift


# ── Hallucination rate helper ─────────────────────────────────────────────────

def measure_hallucination_rate(vlm, tok, clip_processor, records, device):
    results = []
    run_hallucination_eval(vlm, tok, clip_processor, records, device, results)
    rate = sum(r["failed"] for r in results) / len(results)
    return rate


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Data]")
    records = load_image_caption_dataset(max_samples=args.max_samples)
    _, clip_processor = load_clip_full(device)
    print(f"  Total records: {len(records)}")

    # ── Train both variants ───────────────────────────────────────────────────
    vlm_frozen, tok_f, losses_frozen, drift_frozen = train_variant(
        "frozen",        freeze_vision=True,
        records=records, clip_processor=clip_processor,
        args=args, device=device,
    )
    vlm_finetuned, tok_ft, losses_finetuned, drift_finetuned = train_variant(
        "finetuned",     freeze_vision=False,
        records=records, clip_processor=clip_processor,
        args=args, device=device,
    )

    # ── Hallucination rates ───────────────────────────────────────────────────
    print("\n[Hallucination Rate Evaluation]")
    sample_records = records[:20]
    hall_frozen    = measure_hallucination_rate(
        vlm_frozen,    tok_f,  clip_processor, sample_records, device
    )
    hall_finetuned = measure_hallucination_rate(
        vlm_finetuned, tok_ft, clip_processor, sample_records, device
    )
    print(f"\n  Hallucination rate — frozen:    {hall_frozen:.0%}")
    print(f"  Hallucination rate — finetuned: {hall_finetuned:.0%}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_comparison(
        losses_frozen, losses_finetuned,
        drift_frozen,  drift_finetuned,
        hall_frozen,   hall_finetuned,
        results_dir,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("FROZEN vs FINE-TUNED VISION ENCODER SUMMARY")
    print("=" * 55)
    print(f"{'Metric':<35} {'Frozen':>10} {'Fine-tuned':>12}")
    print("-" * 55)
    print(f"{'Final train loss':<35} {losses_frozen[-1]:>10.4f} {losses_finetuned[-1]:>12.4f}")
    print(f"{'Feature drift (1=no drift)':<35} {drift_frozen:>10.4f} {drift_finetuned:>12.4f}")
    print(f"{'Hallucination failure rate':<35} {hall_frozen:>10.0%} {hall_finetuned:>12.0%}")
    print("=" * 55)
    print(f"\nAll results saved to {results_dir}/")


def _plot_comparison(losses_f, losses_ft, drift_f, drift_ft, hall_f, hall_ft,
                     results_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curves
    axes[0].plot(losses_f,  label="Frozen",    color="tab:blue", alpha=0.8)
    axes[0].plot(losses_ft, label="Fine-tuned", color="tab:orange", alpha=0.8)
    axes[0].set_title("Training Loss per Step")
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Feature drift bar
    axes[1].bar(["Frozen", "Fine-tuned"], [drift_f, drift_ft],
                color=["tab:blue", "tab:orange"], edgecolor="black")
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Feature Drift\n(cosine sim to original CLIP features)")
    axes[1].set_ylabel("Cosine Similarity (higher = less drift)")
    axes[1].axhline(1.0, color="grey", linestyle="--", label="no drift")
    axes[1].legend()
    for i, v in enumerate([drift_f, drift_ft]):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)

    # Hallucination bar
    axes[2].bar(["Frozen", "Fine-tuned"], [hall_f, hall_ft],
                color=["tab:blue", "tab:orange"], edgecolor="black")
    axes[2].set_ylim(0, 1.15)
    axes[2].set_title("Object Hallucination Rate\n(lower is better)")
    axes[2].set_ylabel("Failure Rate")
    for i, v in enumerate([hall_f, hall_ft]):
        axes[2].text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=12)

    plt.suptitle("Frozen vs Fine-tuned Vision Encoder", fontsize=13)
    plt.tight_layout()
    out = results_dir / "comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nComparison plot saved → {out}")


def parse_args():
    p = argparse.ArgumentParser(description="Day 5: Frozen vs fine-tuned vision encoder")
    p.add_argument("--checkpoint",      default="results/vlm-prefix/checkpoint.pt",
                   help="Day 3 checkpoint (currently unused — trains from scratch)")
    p.add_argument("--max_samples",     type=int,   default=1000)
    p.add_argument("--max_steps",       type=int,   default=300)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=2e-5)
    p.add_argument("--num_vis_tokens",  type=int,   default=4)
    p.add_argument("--results_dir",     default="results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("Day 5 — Frozen vs Fine-tuned Vision Encoder")
    print("=" * 60)
    print(f"  max_samples={args.max_samples}  max_steps={args.max_steps}")
    main(args)
