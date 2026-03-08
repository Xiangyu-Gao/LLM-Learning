"""Minimal SFT trainer: cross-entropy on ASSISTANT tokens only (behavior cloning).

Key insight
-----------
SFT is behavior cloning.  By masking the loss on user/system tokens we teach the
model to predict only what the assistant wrote, given the conversation prefix.
This makes the model excellent at copying *style* and *format* from demonstrations
but fundamentally unable to resolve preference trade-offs — it has no signal about
which of many plausible continuations is *preferred*.

Datasets
--------
Three datasets are merged and shuffled before training:

  ifeval-like-data-subset  – format-constrained instructions with gold responses
  do-not-answer            – harmful questions paired with Claude refusal responses
  trivia_qa-subset         – factual QA pairs

Usage
-----
  python src/train_sft.py \\
      --data_dir data \\
      --output_dir results/sft-gpt2 \\
      --max_samples 200 \\   # omit to train on full dataset
      --epochs 1 \\
      --batch_size 4
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import Dataset, concatenate_datasets
from swanlab.integration.transformers import SwanLabCallback
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------
# GPT-2 ships with no chat template.  We inject a minimal Jinja2 template that
# uses <|user|> / <|assistant|> sentinels.  The template is stored on the
# tokenizer so TRL's SFTTrainer can apply it automatically.
#
# Why does the template matter for masking?
#   TRL's `completion_only_loss=True` scans the tokenised sequence for the
#   assistant header token(s) and sets labels=-100 for every position *before*
#   the first assistant header.  Picking a clean, unambiguous sentinel avoids
#   false-positive matches inside the actual content.

CHAT_TEMPLATE = (
    "{% for msg in messages %}"
    "{% if msg['role'] == 'user' %}"
    "<|user|>\n{{ msg['content'] }}\n"
    "{% endif %}"
    "{% if msg['role'] == 'assistant' %}"
    "<|assistant|>\n{{ msg['content'] }}{{ eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _read_parquet_capped(folder: Path, limit: int | None) -> pd.DataFrame:
    """Read parquet files from *folder*, stopping as soon as *limit* rows are collected.

    When *limit* is None all files are read.  When *limit* is set only the
    minimum number of files needed are opened, dramatically reducing I/O for
    quick smoke-tests.
    """
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {folder}")

    if limit is None:
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    frames: list[pd.DataFrame] = []
    collected = 0
    for f in files:
        chunk = pd.read_parquet(f)
        remaining = limit - collected
        frames.append(chunk.head(remaining))
        collected += len(frames[-1])
        if collected >= limit:
            break
    return pd.concat(frames, ignore_index=True)


def load_ifeval(data_dir: Path, limit: int | None = None) -> Dataset:
    """Load ifeval-like format-constrained instruction data.

    Each row becomes: user=instruction, assistant=format-constrained response.
    """
    df = _read_parquet_capped(data_dir / "ifeval-like-data-subset", limit)

    records = [
        {
            "messages": [
                {"role": "user", "content": row["instruction"]},
                {"role": "assistant", "content": row["response"]},
            ]
        }
        for _, row in df.iterrows()
    ]
    print(f"  ifeval-like : {len(records):>7,} examples")
    return Dataset.from_list(records)


def load_do_not_answer(data_dir: Path, limit: int | None = None) -> Dataset:
    """Load do-not-answer safety dataset.

    Uses Claude's refusal response as the gold assistant turn so the model
    learns to decline harmful queries in a polite, non-preachy way.
    """
    df = _read_parquet_capped(data_dir / "do-not-answer", limit)

    records = [
        {
            "messages": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["Claude_response"]},
            ]
        }
        for _, row in df.iterrows()
    ]
    print(f"  do-not-answer: {len(records):>6,} examples")
    return Dataset.from_list(records)


_CONTEXT_CHARS = 600  # max chars of grounding context kept per example


def _extract_context(row: pd.Series) -> str:
    """Return the best available grounding context for a TriviaQA row.

    Priority:
      1. Wikipedia full text (entity_pages.wiki_context) — highest quality
      2. Web search snippet (search_results.search_context) — always present
    Truncated to _CONTEXT_CHARS to keep token sequences manageable.
    """
    wiki_pages = list(row["entity_pages"]["wiki_context"])
    wiki_text = " ".join(p for p in wiki_pages if p).strip()
    if wiki_text:
        return wiki_text[:_CONTEXT_CHARS]

    search_ctxs = list(row["search_results"]["search_context"])
    search_text = " ".join(c for c in search_ctxs[:2] if c).strip()
    return search_text[:_CONTEXT_CHARS]


def load_trivia_qa(data_dir: Path, limit: int | None = None) -> Dataset:
    """Load TriviaQA as grounded QA pairs.

    The user turn includes a context snippet (Wikipedia or web search) so the
    model learns to answer *from evidence* rather than from parametric memory.
    This is the standard grounded-QA / RAG fine-tuning pattern.
    """
    df = _read_parquet_capped(data_dir / "trivia_qa-subset", limit)

    records = []
    for _, row in df.iterrows():
        aliases = row["answer"]["aliases"]
        answer = str(aliases[0]) if len(aliases) > 0 else "Unknown"

        context = _extract_context(row)
        if context:
            user_content = (
                f"Use the following context to answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {row['question']}"
            )
        else:
            user_content = row["question"]

        records.append(
            {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": answer},
                ]
            }
        )
    print(f"  trivia_qa   : {len(records):>7,} examples")
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal SFT trainer (assistant-token masking)")
    parser.add_argument("--model_name", default="gpt2",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--data_dir", default="../data",
                        help="Root directory containing the three dataset folders")
    parser.add_argument("--output_dir", default="../results/sft-gpt2",
                        help="Directory to save the fine-tuned model and tokenizer")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum token length per example (longer examples are truncated)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap total dataset size (useful for quick smoke-tests)")
    parser.add_argument("--max_per_source", type=int, default=None,
                        help="Cap rows read per dataset source (limits I/O; applied before merge)")
    parser.add_argument("--seed", type=int, default=42)
    # SwanLab tracking
    parser.add_argument("--swanlab_project", type=str, default="fine-tune",
                        help="SwanLab project name")
    parser.add_argument("--swanlab_mode", type=str, default="local",
                        choices=["local", "cloud", "disabled"],
                        help="SwanLab logging mode: local (no login), cloud, or disabled")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # ── load & merge ────────────────────────────────────────────────────────
    print("Loading datasets...")
    lim = args.max_per_source
    ds_ifeval = load_ifeval(data_dir, limit=lim)
    ds_dna = load_do_not_answer(data_dir, limit=lim)
    ds_tqa = load_trivia_qa(data_dir, limit=lim)

    dataset = concatenate_datasets([ds_ifeval, ds_dna, ds_tqa]).shuffle(seed=args.seed)

    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Total training examples: {len(dataset):,}")

    # ── tokenizer ───────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.chat_template = CHAT_TEMPLATE          # inject template for GPT-2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token    # GPT-2 has no pad token

    # ── model ────────────────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # ── SFT configuration ───────────────────────────────────────────────────
    # completion_only_loss=True  ←  THE MASKING FLAG
    #
    # What it does internally:
    #   After tokenising each chat-formatted sequence, TRL locates the token
    #   positions that belong to the assistant turn by finding the assistant
    #   header sentinel ("<|assistant|>\n").  All positions *before* the first
    #   assistant token get labels=-100, so PyTorch's CrossEntropyLoss ignores
    #   them.  Only assistant tokens contribute to the gradient.
    #
    # Why it matters:
    #   Without masking the model wastes capacity on predicting *deterministic*
    #   prompt text (the user's question is given — there's nothing to learn
    #   there).  With masking, every gradient step is informative about what a
    #   good assistant response looks like.

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        completion_only_loss=True,   # ← assistant-token masking (core SFT trick)
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",            # SwanLab uses a callback, not report_to
        seed=args.seed,
        # GPU/CPU: SFTConfig auto-detects CUDA; no use_cpu flag needed.
    )

    # ── SwanLab experiment tracking ──────────────────────────────────────────
    swanlab_callback = SwanLabCallback(
        project=args.swanlab_project,
        experiment_name=f"sft-{args.model_name.replace('/', '-')}",
        mode=args.swanlab_mode,
        config={
            "model": args.model_name,
            "method": "full_sft",
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
        processing_class=tokenizer,  # TRL ≥0.26 uses processing_class, not tokenizer
        callbacks=[swanlab_callback],
    )

    # ── train ────────────────────────────────────────────────────────────────
    print("\nStarting training  (loss on assistant tokens only)...")
    print(f"  Model        : {args.model_name}")
    print(f"  Examples     : {len(dataset):,}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}  (×{args.grad_accum} grad accum)")
    print(f"  Max length   : {args.max_length}")
    print()

    trainer.train()

    # ── save ─────────────────────────────────────────────────────────────────
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved model and tokenizer → {args.output_dir}")


if __name__ == "__main__":
    main()
