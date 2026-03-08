"""QLoRA SFT trainer: 4-bit quantized base model + 16-bit LoRA adapters.

Key insight
-----------
QLoRA = Quantized LoRA.  The base model weights are loaded in 4-bit NF4
(NormalFloat4) format, cutting GPU memory by ~4×.  LoRA adapters are still
trained and stored in 16-bit, so the *adapter gradients* are precise while
*inference activations* dequantize on-the-fly to the compute dtype (bf16/fp16).

Memory comparison (GPT-2 ~124 M params):
  Full FT    : ~500 MB   (fp32 weights + optimizer states)
  LoRA       : ~250 MB   (fp16 weights, only adapter gradients)
  QLoRA      : ~80  MB   (4-bit weights, only adapter gradients + paged opt)

When does QLoRA matter?
  It's designed for large models (7B+) where 4-bit lets you fit the base
  model on a consumer GPU.  For GPT-2 the memory savings are academic, but
  the code is identical for 7B/13B/70B models.

Known failure modes (see docs/qlora_notes.md for details):
  1. Loss divergence    — high LR + fp16 compute dtype → use bf16 or lower LR
  2. NaN losses         — fp16 underflow in MLP → switch to bf16 compute dtype
  3. Long-context loss  — NF4 quantisation error accumulates → double_quant helps
  4. Gradient instability — gradient checkpointing needs prepare_model_for_kbit_training
  5. Slow wall-clock    — paged optimiser pages to CPU; may hurt small batches

Toggles exposed via argparse:
  --gradient_checkpointing     on/off  (saves VRAM, slows iteration)
  --compute_dtype bf16|fp16            (bf16 recommended on Ampere+)
  --use_paged_optimizer                (paged_adamw_8bit for large models)

Datasets
--------
Identical to train_sft.py and train_lora_sft.py (imported directly).

Usage
-----
  python src/train_qlora_sft.py \\
      --data_dir data \\
      --output_dir results/qlora-r8 \\
      --lora_r 8 \\
      --lora_alpha 16 \\
      --compute_dtype bf16 \\
      --gradient_checkpointing \\
      --max_samples 200 \\
      --epochs 1

  # Quick smoke-test (CPU-safe, skips 4-bit):
  python src/train_qlora_sft.py \\
      --data_dir data \\
      --max_per_source 50 --max_samples 100 --epochs 1 --batch_size 4
"""

import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from swanlab.integration.transformers import SwanLabCallback
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# Import dataset loaders and chat template from train_sft.py (DRY principle).
# QLoRA uses the same conversation format and assistant-token masking.
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
# Helpers
# ---------------------------------------------------------------------------

def _resolve_compute_dtype(dtype_str: str) -> torch.dtype:
    """Map a CLI string to the matching torch dtype.

    bf16 is preferred on Ampere+ GPUs (A100, RTX 30xx/40xx, TITAN RTX).
    fp16 works on older Turing GPUs but is more prone to NaN loss spikes.
    fp32 is the CPU fallback.
    """
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unknown dtype: {dtype_str!r}.  Choose bf16, fp16, or fp32.")
    return mapping[dtype_str]


def _build_bnb_config(compute_dtype: torch.dtype) -> BitsAndBytesConfig:
    """Build the 4-bit BitsAndBytes quantisation config.

    NF4 (NormalFloat4) is the quantisation type introduced by QLoRA.
    It assumes weights follow a zero-mean normal distribution and allocates
    quantisation bins accordingly — outperforming uniform 4-bit INT4.

    double_quant=True applies an additional 8-bit quantisation to the
    quantisation constants themselves, saving ~0.5 bits/parameter extra.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,   # quantise the quant constants too
        bnb_4bit_quant_type="nf4",        # NormalFloat4 — best for LLM weights
        bnb_4bit_compute_dtype=compute_dtype,  # dtype for dequant + forward pass
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QLoRA SFT trainer (4-bit base + 16-bit LoRA adapters)"
    )
    parser.add_argument("--model_name", default="gpt2",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--output_dir", default="../results/qlora-r8")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="QLoRA typically uses slightly lower LR than full LoRA "
                             "because 4-bit forward passes are noisier")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_per_source", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # LoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="c_attn,c_proj",
                        help="Comma-separated GPT-2 attention module names")

    # QLoRA-specific toggles
    parser.add_argument(
        "--compute_dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
        help="Floating-point dtype for 4-bit dequant + LoRA computations. "
             "bf16 is most stable on Ampere+ GPUs.  fp16 on older GPUs.  "
             "fp32 on CPU (disables 4-bit).",
    )
    parser.add_argument(
        "--gradient_checkpointing", action="store_true",
        help="Enable gradient checkpointing: recompute activations in the backward "
             "pass instead of storing them.  Cuts VRAM ~30–40%% at ~20%% slower "
             "training.  Requires prepare_model_for_kbit_training.",
    )
    parser.add_argument(
        "--use_paged_optimizer", action="store_true",
        help="Use paged_adamw_8bit: optimizer states live in paged CPU memory, "
             "released to GPU only when needed.  Essential for 70B+ models; "
             "modest benefit for small models.",
    )
    # SwanLab tracking
    parser.add_argument("--swanlab_project", type=str, default="fine-tune",
                        help="SwanLab project name")
    parser.add_argument("--swanlab_mode", type=str, default="local",
                        choices=["local", "cloud", "disabled"],
                        help="SwanLab logging mode: local (no login), cloud, or disabled")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    compute_dtype = _resolve_compute_dtype(args.compute_dtype)

    # ── 4-bit quantisation check ─────────────────────────────────────────────
    # bitsandbytes 4-bit requires CUDA.  On CPU we fall back to fp32 and skip
    # quantisation — this keeps the script runnable in CI / on CPU machines.
    use_4bit = torch.cuda.is_available()
    if not use_4bit:
        print("WARNING: CUDA not available.  Running in fp32 (no 4-bit quantisation).")
        print("         QLoRA memory savings will not apply.\n")

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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.chat_template = CHAT_TEMPLATE   # inject GPT-2 chat template
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── base model (4-bit or fp32 fallback) ─────────────────────────────────
    print(f"\nLoading base model: {args.model_name}")
    print(f"  Compute dtype         : {args.compute_dtype}")
    print(f"  4-bit quantisation    : {use_4bit}")
    print(f"  Gradient checkpointing: {args.gradient_checkpointing}")
    print(f"  Paged optimizer       : {args.use_paged_optimizer}")

    model_kwargs: dict = {}
    if use_4bit:
        model_kwargs["quantization_config"] = _build_bnb_config(compute_dtype)
        model_kwargs["device_map"] = "auto"  # let BnB place layers across GPUs

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    # ── prepare for k-bit training ───────────────────────────────────────────
    # CRITICAL: prepare_model_for_kbit_training() does three things:
    #   1. Casts layer norms to fp32 (quantised models need stable norms)
    #   2. Calls model.enable_input_require_grads() so grad flows through
    #      frozen quantised weights into the adapter parameters
    #   3. Enables gradient checkpointing if requested
    # Skipping this causes silent gradient bugs with 4-bit + PEFT.
    if use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    elif args.gradient_checkpointing:
        # Non-quantised path: enable manually
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # gradient checkpointing conflicts with the KV-cache — disable it
    if args.gradient_checkpointing:
        model.config.use_cache = False

    # ── LoRA configuration ───────────────────────────────────────────────────
    # The adapter matrices are stored in the compute dtype (bf16/fp16), NOT
    # in 4-bit.  Only the frozen base model weights are quantised.
    # This is the core QLoRA insight: high-precision adapters on a low-precision base.
    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"\nLoRA adapter stats:")
    print(f"  Rank (r)         : {args.lora_r}")
    print(f"  Alpha            : {args.lora_alpha}")
    print(f"  Target modules   : {target_modules}")
    print(f"  Trainable params : {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # ── SFT configuration ────────────────────────────────────────────────────
    # Identical masking strategy to train_sft.py and train_lora_sft.py
    optimizer_str = "paged_adamw_8bit" if (args.use_paged_optimizer and use_4bit) else "adamw_torch"

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        completion_only_loss=True,      # ← assistant-token masking
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        seed=args.seed,
        optim=optimizer_str,
        # fp16 / bf16 flags: these control the SFTTrainer's AMP scaler,
        # independently of the bnb compute dtype.
        fp16=(args.compute_dtype == "fp16" and use_4bit),
        bf16=(args.compute_dtype == "bf16" and use_4bit),
    )

    # ── SwanLab experiment tracking ──────────────────────────────────────────
    swanlab_callback = SwanLabCallback(
        project=args.swanlab_project,
        experiment_name=f"qlora-r{args.lora_r}-alpha{args.lora_alpha}",
        mode=args.swanlab_mode,
        config={
            "model": args.model_name,
            "method": "qlora",
            "compute_dtype": args.compute_dtype,
            "use_4bit": use_4bit,
            "gradient_checkpointing": args.gradient_checkpointing,
            "use_paged_optimizer": args.use_paged_optimizer,
            "optimizer": optimizer_str,
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
    print("\nStarting QLoRA training...")
    print(f"  Examples     : {len(dataset):,}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}  (×{args.grad_accum} grad accum)")
    print(f"  Optimizer    : {optimizer_str}")
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
    # Only the LoRA adapters are saved.  The 4-bit base weights stay in the
    # original pretrained repo and are loaded fresh at inference time.
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved QLoRA adapters → {args.output_dir}")
    print(f"  Training time  : {elapsed:.1f} s")

    # ── save summary ─────────────────────────────────────────────────────────
    total_tokens_approx = len(dataset) * args.epochs * (args.max_length // 2)
    tokens_per_sec = total_tokens_approx / elapsed if elapsed > 0 else 0

    summary = {
        "method": "qlora",
        "model": args.model_name,
        "compute_dtype": args.compute_dtype,
        "use_4bit": use_4bit,
        "gradient_checkpointing": args.gradient_checkpointing,
        "use_paged_optimizer": args.use_paged_optimizer,
        "optimizer": optimizer_str,
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
        **vram_stats,
    }

    summary_path = Path(args.output_dir) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Training summary → {summary_path}")


if __name__ == "__main__":
    main()
