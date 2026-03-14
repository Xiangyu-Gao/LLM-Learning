"""
day3_fusion.py — Day 3: Vision Encoder + LLM Fusion

Implements MiniVLM with two fusion strategies:

  1. prefix-concat  (default)
     CLIP-ViT → last_hidden_state[:, :V, :] → Linear(768,768) → V prefix tokens
     Prepend prefix to GPT-2 inputs_embeds → autoregressive captioning.

  2. cross_attention
     GPT-2 is re-initialised with add_cross_attention=True.
     CLIP patch tokens are passed as encoder_hidden_states → GPT-2 attends to them
     at every decoder layer (Flamingo-style, lightweight).

Both are trained on image captioning (tiny-imagenet / cifar10 captions).
At inference, the prompt is:
    "Question: What is in this image? Answer:"
and the model generates the class description.

After training, both checkpoints are compared on held-out val loss.

Key insight: Most VLMs don't truly "see" — they translate vision into
text-like embeddings that the LLM pattern-matches against training priors.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils import (
    load_clip_vision,
    load_gpt2,
    load_image_caption_dataset,
    ImageCaptionDataset,
    save_vlm_checkpoint,
    CLIP_VIS_DIM,
    GPT2_DIM,
)


# ── MiniVLM ───────────────────────────────────────────────────────────────────

class MiniVLM(nn.Module):
    """
    Mini Vision-Language Model.

    fusion_mode='prefix':
        CLIP patch tokens → linear projection → prefix token sequence
        [vis_prefix | text_embeds] → GPT-2 (inputs_embeds)

    fusion_mode='cross_attention':
        CLIP patch tokens → linear projection → encoder_hidden_states
        GPT-2 (add_cross_attention=True) attends to them at every layer.
        (Requires GPT-2 loaded with add_cross_attention=True.)
    """

    def __init__(self, clip_vis, gpt2, tok,
                 fusion_mode="prefix", num_vis_tokens=4, freeze_vision=True):
        super().__init__()
        assert fusion_mode in ("prefix", "cross_attention"), \
            f"Unknown fusion_mode: {fusion_mode}"

        self.clip_vis      = clip_vis
        self.gpt2          = gpt2
        self.tok           = tok
        self.fusion_mode   = fusion_mode
        self.num_vis_tokens = num_vis_tokens

        # Linear projection: CLIP hidden dim → GPT-2 hidden dim
        # Both are 768 for ViT-B/32 + GPT-2 small, but an explicit
        # projection is still important: it learns a re-parameterisation
        # from CLIP's representation space to GPT-2's.
        self.vision_proj = nn.Linear(CLIP_VIS_DIM, GPT2_DIM)

        if freeze_vision:
            for p in self.clip_vis.parameters():
                p.requires_grad_(False)

    def encode_image(self, pixel_values):
        """
        Returns visual tokens projected into GPT-2 embedding space.
        Shape: (B, num_vis_tokens, GPT2_DIM)

        We use last_hidden_state[:, :num_vis_tokens] so the model can
        choose how many patch tokens to expose to the LLM.
        Index 0 is the CLS token; indices 1..49 are spatial patches.
        """
        out = self.clip_vis(pixel_values=pixel_values)
        # (B, 50, 768): CLS at 0, patches at 1..49
        vis_tokens = out.last_hidden_state[:, :self.num_vis_tokens, :]
        return self.vision_proj(vis_tokens)   # (B, V, 768)

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        B = pixel_values.shape[0]
        vis_prefix = self.encode_image(pixel_values)   # (B, V, 768)

        if self.fusion_mode == "prefix":
            text_embs = self.gpt2.transformer.wte(input_ids)  # (B, T, 768)
            combined  = torch.cat([vis_prefix, text_embs], dim=1)  # (B, V+T, 768)

            V = self.num_vis_tokens
            vis_mask = torch.ones(B, V, dtype=attention_mask.dtype,
                                  device=attention_mask.device)
            full_mask = torch.cat([vis_mask, attention_mask], dim=1)

            if labels is not None:
                vis_lbl   = torch.full((B, V), -100, dtype=labels.dtype,
                                       device=labels.device)
                full_lbl  = torch.cat([vis_lbl, labels], dim=1)
            else:
                full_lbl = None

            return self.gpt2(
                inputs_embeds=combined,
                attention_mask=full_mask,
                labels=full_lbl,
            )

        else:  # cross_attention
            # encoder_attention_mask: all visual tokens visible
            enc_mask = torch.ones(B, self.num_vis_tokens,
                                  dtype=attention_mask.dtype,
                                  device=attention_mask.device)
            return self.gpt2(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=vis_prefix,
                encoder_attention_mask=enc_mask,
                labels=labels,
            )

    @torch.no_grad()
    def generate(self, pixel_values, prompt_ids=None, prompt_mask=None,
                 max_new_tokens=30, do_sample=False, temperature=1.0):
        """
        Generate text conditioned on an image.

        prefix mode:
            inputs_embeds = [vis_prefix | prompt_embeds (optional)]
            Returns new token IDs only (not the prefix).

        cross_attention mode:
            Passes vis_prefix as encoder_hidden_states.
            Generates from prompt_ids autoregressively.
        """
        # Suppress max_length conflict
        self.gpt2.generation_config.max_length = None

        B = pixel_values.shape[0]
        vis_prefix = self.encode_image(pixel_values)   # (B, V, 768)
        enc_mask   = torch.ones(B, self.num_vis_tokens,
                                device=pixel_values.device, dtype=torch.long)

        if self.fusion_mode == "prefix":
            if prompt_ids is not None:
                text_embs = self.gpt2.transformer.wte(prompt_ids)
                inputs_embeds = torch.cat([vis_prefix, text_embs], dim=1)
                if prompt_mask is not None:
                    full_mask = torch.cat([enc_mask, prompt_mask], dim=1)
                else:
                    full_mask = torch.cat([
                        enc_mask,
                        torch.ones(B, prompt_ids.shape[1],
                                   device=pixel_values.device, dtype=torch.long)
                    ], dim=1)
            else:
                inputs_embeds = vis_prefix
                full_mask = enc_mask

            return self.gpt2.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self.tok.eos_token_id,
            )

        else:  # cross_attention
            if prompt_ids is None:
                prompt_ids = torch.tensor(
                    [[self.tok.bos_token_id]], device=pixel_values.device
                ).expand(B, -1)
                prompt_mask = torch.ones_like(prompt_ids)
            return self.gpt2.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                encoder_hidden_states=vis_prefix,
                encoder_attention_mask=enc_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self.tok.eos_token_id,
            )


# ── Training utilities ────────────────────────────────────────────────────────

def make_dataloader(records, clip_processor, gpt2_tok, batch_size,
                    shuffle=True, max_text_length=64):
    ds = ImageCaptionDataset(records, clip_processor, gpt2_tok,
                             max_text_length=max_text_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, pin_memory=True)


def train_one_mode(fusion_mode, records, clip_processor, args, device,
                   results_dir):
    """Train a MiniVLM with the given fusion_mode. Returns loss history."""
    print(f"\n── Fusion mode: {fusion_mode} ─────────────────────────────────")

    is_xattn = (fusion_mode == "cross_attention")
    clip_vis, _ = load_clip_vision(device)
    gpt2, tok   = load_gpt2(device, add_cross_attention=is_xattn)

    vlm = MiniVLM(
        clip_vis=clip_vis,
        gpt2=gpt2,
        tok=tok,
        fusion_mode=fusion_mode,
        num_vis_tokens=args.num_vis_tokens,
        freeze_vision=args.freeze_vision,
    ).to(device)

    n_val   = min(200, len(records) // 5)
    n_train = len(records) - n_val
    train_dl = make_dataloader(records[:n_train],   clip_processor, tok,
                               args.batch_size, shuffle=True)
    val_dl   = make_dataloader(records[n_train:],   clip_processor, tok,
                               args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        [p for p in vlm.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    losses     = []
    val_losses = []
    step = 0

    for epoch in range(args.epochs):
        vlm.train()
        epoch_loss = 0.0

        for batch in train_dl:
            if args.max_steps and step >= args.max_steps:
                break

            pv   = batch["pixel_values"].to(device)
            ids  = batch["input_ids"].to(device)
            msk  = batch["attention_mask"].to(device)
            # Use input_ids as labels; GPT-2 handles the shift internally
            lbl  = ids.clone()
            # Mask padding tokens from loss
            lbl[msk == 0] = -100

            out  = vlm(pixel_values=pv, input_ids=ids,
                       attention_mask=msk, labels=lbl)
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vlm.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())
            epoch_loss += loss.item()
            step += 1

            if step % 50 == 0:
                print(f"  step={step} loss={loss.item():.4f}")

        if args.max_steps and step >= args.max_steps:
            break

        # Validation loss
        vlm.eval()
        with torch.no_grad():
            vl = 0.0
            vn = 0
            for batch in val_dl:
                pv  = batch["pixel_values"].to(device)
                ids = batch["input_ids"].to(device)
                msk = batch["attention_mask"].to(device)
                lbl = ids.clone(); lbl[msk == 0] = -100
                out = vlm(pixel_values=pv, input_ids=ids,
                          attention_mask=msk, labels=lbl)
                vl += out.loss.item(); vn += 1
        val_l = vl / max(vn, 1)
        val_losses.append(val_l)
        print(f"  Epoch {epoch+1} val_loss={val_l:.4f}")

    # Save checkpoint
    ckpt_path = results_dir / f"vlm-{fusion_mode}" / "checkpoint.pt"
    save_vlm_checkpoint(
        ckpt_path, vlm,
        config_dict={
            "fusion_mode":      fusion_mode,
            "num_vis_tokens":   args.num_vis_tokens,
            "freeze_vision":    args.freeze_vision,
            "clip_model_id":    "openai/clip-vit-base-patch32",
            "gpt2_model_id":    "gpt2",
            "final_train_loss": losses[-1] if losses else None,
            "final_val_loss":   val_losses[-1] if val_losses else None,
        },
    )

    # Quick generation sample
    print("\n  [Sample generations]")
    _sample_generation(vlm, tok, val_dl, device, n=3)

    return losses, val_losses


@torch.no_grad()
def _sample_generation(vlm, tok, loader, device, n=3):
    vlm.eval()
    vlm.gpt2.generation_config.max_length = None
    batch = next(iter(loader))
    pv = batch["pixel_values"][:n].to(device)
    captions = batch["caption"][:n]

    # Prompt: "Question: What is in this image? Answer:"
    prompt_text = "Question: What is in this image? Answer:"
    prompt_enc  = tok(prompt_text, return_tensors="pt", padding=True).to(device)
    prompt_ids  = prompt_enc["input_ids"].expand(n, -1)
    prompt_mask = prompt_enc["attention_mask"].expand(n, -1)

    gen_ids = vlm.generate(
        pv, prompt_ids=prompt_ids, prompt_mask=prompt_mask,
        max_new_tokens=20, do_sample=False,
    )
    for i in range(n):
        gen_text = tok.decode(gen_ids[i], skip_special_tokens=True)
        print(f"  [{i+1}] gt='{captions[i]}'  gen='{gen_text}'")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n[Data]")
    records = load_image_caption_dataset(max_samples=args.max_samples)
    print(f"  Total records: {len(records)}")

    from utils import load_clip_full
    _, clip_processor = load_clip_full(device)

    # ── Train both fusion modes ───────────────────────────────────────────────
    all_losses = {}

    for mode in args.fusion_modes:
        losses, val_losses = train_one_mode(
            mode, records, clip_processor, args, device, results_dir
        )
        all_losses[mode] = {"train": losses, "val": val_losses}

    # ── Compare loss curves ───────────────────────────────────────────────────
    _plot_comparison(all_losses, results_dir)
    print(f"\nAll results saved to {results_dir}/")


def _plot_comparison(all_losses, results_dir):
    """Plot training loss curves for all fusion modes on one figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    colors = {"prefix": "tab:blue", "cross_attention": "tab:orange"}

    for mode, data in all_losses.items():
        col = colors.get(mode, "tab:grey")
        ax1.plot(data["train"], label=mode, color=col, alpha=0.8)
        if data["val"]:
            ax2.plot(data["val"], label=mode, marker="o", color=col)

    ax1.set_title("Training Loss per Step"); ax1.set_xlabel("Step")
    ax1.set_ylabel("Cross-Entropy Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.set_title("Validation Loss per Epoch"); ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross-Entropy Loss"); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle("MiniVLM: Prefix-Concat vs Cross-Attention Fusion", fontsize=12)
    plt.tight_layout()
    out = results_dir / "fusion_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nComparison plot saved → {out}")


def parse_args():
    p = argparse.ArgumentParser(description="Day 3: VLM fusion strategies")
    p.add_argument("--config",          default=None)
    p.add_argument("--max_samples",     type=int,   default=2000)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--epochs",          type=int,   default=2)
    p.add_argument("--lr",              type=float, default=2e-5)
    p.add_argument("--max_steps",       type=int,   default=None,
                   help="Limit steps per fusion mode (for quick testing)")
    p.add_argument("--num_vis_tokens",  type=int,   default=4)
    p.add_argument("--freeze_vision",   action="store_true", default=True)
    p.add_argument("--no_freeze_vision", dest="freeze_vision", action="store_false")
    p.add_argument("--fusion_modes",    nargs="+",
                   default=["prefix", "cross_attention"])
    p.add_argument("--results_dir",     default="results/day3")
    p.add_argument("--checkpoint",      default=None,
                   help="Save primary checkpoint path (overrides results_dir)")
    return p.parse_args()


def apply_config(args):
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
    print("Day 3 — Vision Encoder + LLM Fusion")
    print("=" * 60)
    print(f"  fusion_modes={args.fusion_modes}")
    print(f"  max_samples={args.max_samples}  batch_size={args.batch_size}")
    print(f"  epochs={args.epochs}  max_steps={args.max_steps}")
    print(f"  num_vis_tokens={args.num_vis_tokens}  freeze_vision={args.freeze_vision}")
    main(args)
