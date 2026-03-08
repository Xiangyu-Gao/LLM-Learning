"""Full fine-tuning trainer: all parameters updated, identical setup to SFT.

Key insight
-----------
Full fine-tuning (full FT) updates *every* weight in the model.  For GPT-2
(124 M parameters) this is tractable on a single GPU.  For 7B+ models it
requires sharding across many GPUs or using LoRA/QLoRA instead.

Why run full FT alongside LoRA/QLoRA?
  Fair comparison.  All three variants use:
    - The same datasets (ifeval-like + do-not-answer + trivia_qa)
    - The same chat template and assistant-token masking
    - The same tokenizer and GPT-2 base model
    - The same eval scripts

  The only difference is *which parameters* receive gradients.

  Expected outcomes:
    Full FT   → highest potential quality, highest VRAM, largest checkpoint
    LoRA      → close quality (on small models), ~10× smaller adapters
    QLoRA     → similar to LoRA but 4× lower VRAM (at cost of slight noise)

Short runs vs. convergence
  --max_steps lets you run a fixed number of optimiser steps regardless of
  dataset size.  This is useful for:
    - Apples-to-apples wall-clock comparisons with LoRA
    - Quickly checking that training loss is decreasing before a full run
  Omit --max_steps (or set to -1) to train for --epochs full epochs.

Usage
-----
  # Quick comparison run (same steps as LoRA smoke-test):
  python src/train_fullft.py \\
      --data_dir data \\
      --output_dir results/fullft-gpt2 \\
      --max_per_source 50 \\
      --max_samples 100 \\
      --epochs 1 \\
      --batch_size 4

  # Fixed-step run for fair comparison:
  python src/train_fullft.py \\
      --data_dir data \\
      --output_dir results/fullft-gpt2 \\
      --max_steps 200 \\
      --batch_size 4
"""

import argparse
import json
import time
from pathlib import Path

import torch
from swanlab.integration.transformers import SwanLabCallback
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# Import dataset loaders and chat template from train_sft.py.
# Full FT uses the exact same data pipeline, tokenizer, and masking as SFT
# so the only variable between experiments is the training method.
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_sft import (
    CHAT_TEMPLATE,
    load_do_not_answer,
    load_ifeval,
    load_trivia_qa,
)
from datasets import concatenate_datasets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full fine-tuning trainer (identical setup to SFT, all params updated)"
    )
    parser.add_argument("--model_name", default="gpt2",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--output_dir", default="../results/fullft-gpt2")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max_steps", type=int, default=-1,
        help="If > 0, stop after this many optimizer steps (overrides --epochs). "
             "Use this for fixed-step comparisons against LoRA runs.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps. "
                             "Full FT may need larger grad_accum than LoRA to fit in VRAM.")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Full FT uses lower LR than LoRA (5e-5 vs 5e-4). "
                             "Updating all params is more destructive so smaller steps are safer.")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Fraction of steps used for LR warm-up.  Prevents early instability.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="L2 regularisation.  Helps prevent catastrophic forgetting "
                             "when fine-tuning on a small dataset.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_per_source", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    # SwanLab tracking
    parser.add_argument("--swanlab_project", type=str, default="fine-tune",
                        help="SwanLab project name")
    parser.add_argument("--swanlab_mode", type=str, default="local",
                        choices=["local", "cloud", "disabled"],
                        help="SwanLab logging mode: local (no login), cloud, or disabled")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    # ── load & merge datasets ────────────────────────────────────────────────
    print("Loading datasets...")
    lim = args.max_per_source
    ds_ifeval = load_ifeval(data_dir, limit=lim)
    ds_dna    = load_do_not_answer(data_dir, limit=lim)
    ds_tqa    = load_trivia_qa(data_dir, limit=lim)

    dataset = concatenate_datasets([ds_ifeval, ds_dna, ds_tqa]).shuffle(seed=args.seed)

    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Total training examples: {len(dataset):,}")

    # ── tokenizer ────────────────────────────────────────────────────────────
    # Identical setup to train_sft.py and train_lora_sft.py.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.chat_template = CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── model ────────────────────────────────────────────────────────────────
    # No quantisation, no PEFT.  All 124 M parameters are loaded in fp32/fp16
    # and will receive gradient updates.
    print(f"\nLoading full model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Count parameters for the comparison summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable_params:,}  (100% — full fine-tuning)")

    # ── SFT configuration ────────────────────────────────────────────────────
    # completion_only_loss=True: same assistant-token masking as all other trainers.
    # warmup_ratio and weight_decay are more important here than in LoRA runs
    # because we're updating every weight — overfitting and instability are
    # more likely on small datasets.
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,        # -1 means epoch-driven (HF convention)
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        completion_only_loss=True,       # ← assistant-token masking
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        seed=args.seed,
    )

    # ── SwanLab experiment tracking ──────────────────────────────────────────
    swanlab_callback = SwanLabCallback(
        project=args.swanlab_project,
        experiment_name=f"fullft-{args.model_name.replace('/', '-')}",
        mode=args.swanlab_mode,
        config={
            "model": args.model_name,
            "method": "full_ft",
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_pct": 100.0,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "learning_rate": args.lr,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
            "examples": len(dataset),
        },
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[swanlab_callback],
    )

    # ── train ────────────────────────────────────────────────────────────────
    print("\nStarting full fine-tuning (all parameters updated)...")
    print(f"  Model        : {args.model_name}")
    print(f"  Examples     : {len(dataset):,}")
    if args.max_steps > 0:
        print(f"  Max steps    : {args.max_steps}")
    else:
        print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}  (×{args.grad_accum} grad accum)")
    print(f"  LR           : {args.lr}  (lower than LoRA — updating all weights)")
    print(f"  Weight decay : {args.weight_decay}")
    print(f"  Warmup ratio : {args.warmup_ratio}")
    print()

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    # ── VRAM usage ───────────────────────────────────────────────────────────
    vram_stats: dict = {}
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.max_memory_allocated() / 1024**3
        vram_reserved  = torch.cuda.max_memory_reserved() / 1024**3
        vram_stats = {"vram_allocated_gb": vram_allocated, "vram_reserved_gb": vram_reserved}
        print(f"\n  VRAM allocated : {vram_allocated:.2f} GB")
        print(f"  VRAM reserved  : {vram_reserved:.2f} GB")

    # ── save ─────────────────────────────────────────────────────────────────
    # Full FT saves the entire model (~500 MB for GPT-2, much larger for 7B+).
    # LoRA saves only adapters (~few MB).  This is the storage trade-off.
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved full fine-tuned model → {args.output_dir}")
    print(f"  Training time  : {elapsed:.1f} s")

    # ── save summary ─────────────────────────────────────────────────────────
    total_tokens_approx = len(dataset) * args.epochs * (args.max_length // 2)
    tokens_per_sec = total_tokens_approx / elapsed if elapsed > 0 else 0

    summary = {
        "method": "full_ft",
        "model": args.model_name,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": 100.0,
        "examples": len(dataset),
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "elapsed_sec": elapsed,
        "tokens_per_sec": tokens_per_sec,
        **vram_stats,
    }

    summary_path = Path(args.output_dir) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Training summary → {summary_path}")


if __name__ == "__main__":
    main()
