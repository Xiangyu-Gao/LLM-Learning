"""LoRA SFT trainer: parameter-efficient fine-tuning with low-rank adapters.

Key insight
-----------
LoRA (Low-Rank Adaptation) freezes the pretrained weights and injects trainable
low-rank decomposition matrices (A, B) into each attention layer:

    W_new = W_frozen + (B @ A) * (alpha / r)

where:
  - r is the rank (4, 8, 16, ...) — controls adapter capacity
  - alpha is a scaling factor — typically set to 2×r or equal to r
  - Only A and B are trained (< 1% of total parameters)

This gives:
  ✓ Much faster training (fewer gradients, less VRAM)
  ✓ Smaller checkpoints (only adapters are saved, not full model)
  ✓ Same masking/dataset/template as full SFT
  ✗ Slightly lower final quality vs. full fine-tuning (depends on task)

We perform a hyperparameter sweep: rank r ∈ {4, 8, 16}, alpha ∈ {8, 16, 32}.

Datasets
--------
Identical to train_sft.py:
  - ifeval-like-data-subset
  - do-not-answer
  - trivia_qa-subset

Chat template and assistant-token masking are also identical.

Usage
-----
  # From YAML config (recommended):
  python src/train_lora_sft.py --config configs/lora_r8.yaml

  # Command-line override:
  python src/train_lora_sft.py \\
      --data_dir data \\
      --output_dir results/lora-r8 \\
      --lora_r 8 \\
      --lora_alpha 16 \\
      --max_samples 200 \\
      --epochs 1 \\
      --batch_size 4
"""

import argparse
import time
import yaml
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, TaskType
from swanlab.integration.transformers import SwanLabCallback
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# Import dataset loaders from train_sft.py (DRY principle)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_sft import (
    CHAT_TEMPLATE,
    load_ifeval,
    load_do_not_answer,
    load_trivia_qa,
)
from datasets import concatenate_datasets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT trainer with rank/alpha sweep")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (overrides defaults)")
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--output_dir", default="../results/lora-r8")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="LoRA typically uses 10× higher LR than full FT")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_per_source", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # LoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank (adapter dimension)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha (scaling factor, typically 2×r)")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="c_attn,c_proj",
                        help="Comma-separated module names to apply LoRA (GPT-2 uses c_attn/c_proj)")

    # SwanLab tracking
    parser.add_argument("--swanlab_project", type=str, default="fine-tune",
                        help="SwanLab project name")
    parser.add_argument("--swanlab_mode", type=str, default="local",
                        choices=["local", "cloud", "disabled"],
                        help="SwanLab logging mode: local (no login), cloud, or disabled")

    args = parser.parse_args()

    # ── load config from YAML if provided ───────────────────────────────────
    if args.config:
        print(f"Loading config from {args.config}")
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        # Override args with config values (command-line takes precedence)
        for section in ["model", "training", "lora", "data", "swanlab"]:
            if section in cfg:
                for key, val in cfg[section].items():
                    attr_name = key
                    if section == "lora":
                        attr_name = f"lora_{key}"
                    elif section == "model":
                        if key == "name":
                            attr_name = "model_name"
                    elif section == "data":
                        if key == "dir":
                            attr_name = "data_dir"
                        elif key == "max_samples":
                            attr_name = "max_samples"
                        elif key == "max_per_source":
                            attr_name = "max_per_source"
                    elif section == "training":
                        if key == "learning_rate":
                            attr_name = "lr"
                            val = float(val)  # PyYAML may return "5e-4" as str
                        elif key == "output_dir":
                            attr_name = "output_dir"
                    elif section == "swanlab":
                        attr_name = f"swanlab_{key}"  # project → swanlab_project, mode → swanlab_mode

                    # Only override if the arg wasn't explicitly set via CLI
                    if hasattr(args, attr_name):
                        default_val = parser.get_default(attr_name)
                        current_val = getattr(args, attr_name)
                        if current_val == default_val:
                            setattr(args, attr_name, val)

    data_dir = Path(args.data_dir)

    # ── load & merge datasets ────────────────────────────────────────────────
    print("Loading datasets...")
    lim = args.max_per_source
    ds_ifeval = load_ifeval(data_dir, limit=lim)
    ds_dna = load_do_not_answer(data_dir, limit=lim)
    ds_tqa = load_trivia_qa(data_dir, limit=lim)

    dataset = concatenate_datasets([ds_ifeval, ds_dna, ds_tqa]).shuffle(seed=args.seed)

    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Total training examples: {len(dataset):,}")

    # ── tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.chat_template = CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── base model ───────────────────────────────────────────────────────────
    print(f"\nLoading base model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # ── LoRA configuration ───────────────────────────────────────────────────
    # Parse target modules (comma-separated string → list)
    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]

    lora_config = LoraConfig(
        r=args.lora_r,                    # rank
        lora_alpha=args.lora_alpha,       # scaling factor
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",                      # don't train bias terms
    )

    print("\nApplying LoRA adapters...")
    print(f"  Rank (r)         : {args.lora_r}")
    print(f"  Alpha            : {args.lora_alpha}")
    print(f"  Target modules   : {target_modules}")
    print(f"  Dropout          : {args.lora_dropout}")

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable params : {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # ── SFT configuration ────────────────────────────────────────────────────
    # Identical masking strategy to train_sft.py (completion_only_loss=True)
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        completion_only_loss=True,   # ← assistant-token masking
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",            # SwanLab uses a callback, not report_to
        seed=args.seed,
    )

    # ── SwanLab experiment tracking ──────────────────────────────────────────
    swanlab_callback = SwanLabCallback(
        project=args.swanlab_project,
        experiment_name=f"lora-r{args.lora_r}-alpha{args.lora_alpha}",
        mode=args.swanlab_mode,
        config={
            "model": args.model_name,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": target_modules,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": round(trainable / total * 100, 4),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "learning_rate": args.lr,
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
    print("\nStarting LoRA training (loss on assistant tokens only)...")
    print(f"  Examples     : {len(dataset):,}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}  (×{args.grad_accum} grad accum)")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max length   : {args.max_length}")
    print()

    # Track throughput
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    # Estimate tokens/sec (rough calculation)
    # Assumes avg sequence length ≈ max_length / 2
    total_tokens_approx = len(dataset) * args.epochs * (args.max_length // 2)
    tokens_per_sec = total_tokens_approx / elapsed if elapsed > 0 else 0

    # ── save ─────────────────────────────────────────────────────────────────
    # Save only the LoRA adapters (not the base model)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nSaved LoRA adapters → {args.output_dir}")
    print(f"  Training time    : {elapsed:.1f} s")
    print(f"  Throughput (est) : {tokens_per_sec:.0f} tokens/s")

    # ── VRAM usage (if CUDA available) ───────────────────────────────────────
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        vram_reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(f"  VRAM allocated   : {vram_allocated:.2f} GB")
        print(f"  VRAM reserved    : {vram_reserved:.2f} GB")

    # ── save training summary ────────────────────────────────────────────────
    summary = {
        "model": args.model_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": trainable / total * 100,
        "examples": len(dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.lr,
        "elapsed_sec": elapsed,
        "tokens_per_sec": tokens_per_sec,
    }

    if torch.cuda.is_available():
        summary["vram_allocated_gb"] = vram_allocated
        summary["vram_reserved_gb"] = vram_reserved

    import json
    summary_path = Path(args.output_dir) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTraining summary saved → {summary_path}")


if __name__ == "__main__":
    main()
