"""Grounded QA evaluation on TriviaQA data.

Key insight
-----------
"Grounding" means the model should answer questions using the provided
context rather than relying purely on parametric memory (weights).  A
fine-tuned model that learned from grounded demonstrations should:
  1. Cite / reproduce text from the provided context (citation presence)
  2. Produce answers that exactly or partially match gold aliases (exact/substring match)

Three complementary metrics
---------------------------
  exact_match      : normalised response exactly equals a gold alias
                     (case-fold, strip punctuation)
  substring_match  : any gold alias appears as a substring in the response
                     (less strict — good for longer generative responses)
  citation_present : a verbatim phrase (≥10 chars) from the context appears
                     in the response — proxy for context-grounded reasoning

Why compare base vs. fine-tuned?
  A base GPT-2 mostly ignores the context and produces fluent but irrelevant
  text.  A fine-tuned model that saw context→answer demonstrations should
  extract answers from the context.  The delta between base and FT is the
  measurable effect of grounding training.

Usage
-----
  # Evaluate a fine-tuned model:
  python src/eval_grounding.py \\
      --ft_model_dir results/sft-gpt2 \\
      --data_dir data \\
      --max_samples 50

  # Evaluate multiple models:
  python src/eval_grounding.py \\
      --ft_model_dir results/lora-r8-alpha16 \\
      --base_model gpt2 \\
      --data_dir data \\
      --max_samples 100 \\
      --output_json results/grounding_eval.json

  # Skip base model comparison (FT only):
  python src/eval_grounding.py \\
      --ft_model_dir results/sft-gpt2 \\
      --skip_base \\
      --data_dir data

  # Sanity-check with gold answers (EM/Sub should be ~100%):
  python src/eval_grounding.py \\
      --use_gold_responses \\
      --data_dir data \\
      --max_samples 100

  # Evaluate a base model (should score much lower than the fine-tuned model):
    python src/eval_grounding.py \\
        --base_model gpt2 \\
        --data_dir data \\
        --max_samples 100
"""

import argparse
import json
import re
import string
from pathlib import Path
from typing import Optional

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# Import dataset loader and chat template from train_sft.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_sft import CHAT_TEMPLATE, _read_parquet_capped


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace.

    Standard normalisation used in SQuAD / TriviaQA official eval scripts.
    Keeps comparison fair across models with different capitalisation habits.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def exact_match(prediction: str, aliases: list[str]) -> bool:
    """True if the normalised prediction equals any normalised alias."""
    pred_norm = _normalise(prediction)
    return any(_normalise(a) == pred_norm for a in aliases)


def substring_match(prediction: str, aliases: list[str]) -> bool:
    """True if any normalised alias appears as a substring in the prediction.

    This is more lenient than exact match and works better with generative
    models that tend to answer in full sentences.
    """
    pred_norm = _normalise(prediction)
    return any(_normalise(a) in pred_norm for a in aliases if a.strip())


def citation_present(prediction: str, context: str, min_phrase_len: int = 10) -> bool:
    """True if at least one verbatim phrase from the context appears in the prediction.

    The context is split into overlapping 10-char windows and we check whether
    any appear verbatim (case-insensitive) in the prediction.  This is a rough
    proxy for "the model is reading and copying from the context" rather than
    generating from parametric memory.

    min_phrase_len: minimum characters for a context snippet to count as a
    citation.  Short phrases (e.g. "the", "in") appear everywhere by chance.
    """
    if not context or not prediction:
        return False

    pred_lower = prediction.lower()
    ctx_lower  = context.lower()

    # Slide a window across the context; check each chunk against prediction.
    for start in range(0, len(ctx_lower) - min_phrase_len + 1, min_phrase_len // 2):
        chunk = ctx_lower[start : start + min_phrase_len].strip()
        if len(chunk) >= min_phrase_len and chunk in pred_lower:
            return True
    return False


# ---------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = "<|user|>\n{user_content}\n<|assistant|>\n"


def _load_model_pipeline(model_dir_or_name: str, chat_template: Optional[str] = None):
    """Load a model + tokenizer and return a text-generation pipeline."""
    print(f"  Loading model: {model_dir_or_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir_or_name)
    if chat_template:
        tokenizer.chat_template = chat_template   # needed for bare GPT-2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_cfg_path = Path(model_dir_or_name) / "adapter_config.json"
    if adapter_cfg_path.exists():
        import json
        with open(adapter_cfg_path) as f:
            base_model_name = json.load(f)["base_model_name_or_path"]
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, model_dir_or_name)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir_or_name)
    model.generation_config.max_length = None   # avoid conflict with max_new_tokens

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return gen


def generate_answers(gen_pipeline, user_contents: list[str], max_new_tokens: int) -> list[str]:
    """Generate one answer per user_content string using the given pipeline."""
    responses = []
    for i, content in enumerate(user_contents, 1):
        prompt = PROMPT_TEMPLATE.format(user_content=content)
        raw = gen_pipeline(prompt, max_new_tokens=max_new_tokens, max_length=None)[0]["generated_text"]
        # Strip prompt prefix; truncate at next user turn
        answer = raw[len(prompt):].split("<|user|>")[0].strip()
        responses.append(answer)
        if i % 10 == 0:
            print(f"    Generated {i}/{len(user_contents)}...")
    return responses


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _score_predictions(
    predictions: list[str],
    contexts: list[str],
    all_aliases: list[list[str]],
    model_label: str,
) -> dict:
    """Compute EM / Sub / Citation metrics for a list of pre-generated predictions."""
    em_scores, sub_scores, cit_scores = [], [], []
    details = []
    for pred, ctx, aliases in zip(predictions, contexts, all_aliases):
        em  = exact_match(pred, aliases)
        sub = substring_match(pred, aliases)
        cit = citation_present(pred, ctx)
        em_scores.append(em)
        sub_scores.append(sub)
        cit_scores.append(cit)
        details.append({
            "prediction": pred[:200],
            "exact_match": em,
            "substring_match": sub,
            "citation_present": cit,
        })
    n = len(predictions)
    return {
        "model": model_label,
        "n": n,
        "exact_match_pct":      sum(em_scores)  / n * 100,
        "substring_match_pct":  sum(sub_scores)  / n * 100,
        "citation_present_pct": sum(cit_scores) / n * 100,
        "details": details,
    }


def evaluate_model(
    gen_pipeline,
    user_contents: list[str],
    contexts: list[str],
    all_aliases: list[list[str]],
    max_new_tokens: int,
    model_label: str,
) -> dict:
    """Run generation + compute all three metrics for one model.

    Returns a dict with aggregate statistics and per-example details.
    """
    print(f"\nEvaluating: {model_label}")
    predictions = generate_answers(gen_pipeline, user_contents, max_new_tokens)
    return _score_predictions(predictions, contexts, all_aliases, model_label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict:
    parser = argparse.ArgumentParser(description="Grounded QA evaluation on TriviaQA")
    parser.add_argument(
        "--ft_model_dir", default="results/sft-gpt2",
        help="Fine-tuned model directory to evaluate",
    )
    parser.add_argument(
        "--base_model", default="gpt2",
        help="Base (untuned) model to compare against",
    )
    parser.add_argument("--data_dir", default="data")
    parser.add_argument(
        "--max_samples", type=int, default=50,
        help="Number of TriviaQA examples to evaluate",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--skip_base", action="store_true",
        help="Skip base model evaluation (saves time if you only need FT results)",
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Write JSON results to this path (for use by eval_summary.py)",
    )
    parser.add_argument(
        "--use_gold_responses", action="store_true",
        help="Use the first gold alias as the prediction (sanity check — EM/Sub should be ~100%)",
    )
    args = parser.parse_args()

    # ── load TriviaQA data ───────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    print(f"Loading TriviaQA data (max {args.max_samples} examples)...")
    df = _read_parquet_capped(data_dir / "trivia_qa-subset", limit=args.max_samples)
    df = df.head(args.max_samples)
    print(f"Loaded {len(df)} examples.")

    # Build user content strings (same format as training)
    from train_sft import _extract_context  # reuse the same context extractor
    user_contents, contexts, all_aliases = [], [], []

    for _, row in df.iterrows():
        ctx = _extract_context(row)
        aliases = list(row["answer"]["aliases"])
        if ctx:
            user_content = (
                f"Use the following context to answer the question.\n\n"
                f"Context:\n{ctx}\n\n"
                f"Question: {row['question']}"
            )
        else:
            user_content = row["question"]
        user_contents.append(user_content)
        contexts.append(ctx)
        all_aliases.append(aliases)

    # ── evaluate models ──────────────────────────────────────────────────────
    results_all = []

    if args.use_gold_responses:
        print("Using GOLD responses (sanity check — EM/Sub should be ~100%).")
        gold_preds = [aliases[0] for aliases in all_aliases]
        results_all.append(_score_predictions(gold_preds, contexts, all_aliases, "GOLD"))
    else:
        # Fine-tuned model
        ft_pipeline = _load_model_pipeline(args.ft_model_dir)
        ft_results = evaluate_model(
            ft_pipeline, user_contents, contexts, all_aliases,
            args.max_new_tokens, model_label=args.ft_model_dir,
        )
        results_all.append(ft_results)
        del ft_pipeline

        # Base model comparison
        if not args.skip_base:
            base_pipeline = _load_model_pipeline(args.base_model, chat_template=CHAT_TEMPLATE)
            base_results = evaluate_model(
                base_pipeline, user_contents, contexts, all_aliases,
                args.max_new_tokens, model_label=f"{args.base_model} (base)",
            )
            results_all.append(base_results)
            del base_pipeline

    # ── print report ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"Grounding Evaluation  |  TriviaQA  |  {len(df)} examples")
    print(f"{'='*72}")
    header = f"  {'Model':<40} {'EM':>6} {'Sub':>6} {'Cit':>6}"
    print(header)
    print(f"  {'-'*40} {'------':>6} {'------':>6} {'------':>6}")
    for r in results_all:
        label = r["model"][-40:]  # truncate long paths
        print(f"  {label:<40} "
              f"{r['exact_match_pct']:>5.1f}% "
              f"{r['substring_match_pct']:>5.1f}% "
              f"{r['citation_present_pct']:>5.1f}%")
    print(f"{'='*72}")
    print()
    print("Metric definitions:")
    print("  EM  = Exact match  (normalised prediction == normalised gold alias)")
    print("  Sub = Substring    (gold alias appears anywhere in prediction)")
    print("  Cit = Citation     (≥10-char verbatim phrase from context in prediction)")
    print()
    print("Interpretation:")
    print("  A base model ignores the context and generates fluent but irrelevant text.")
    print("  A grounding-trained model should score higher on Cit, indicating it")
    print("  reads and quotes from the supplied context passage.")

    # ── save JSON ────────────────────────────────────────────────────────────
    output = {
        "eval_type": "grounding",
        "dataset": "trivia_qa-subset",
        "n": len(df),
        "results": [{k: v for k, v in r.items() if k != "details"} for r in results_all],
        "details": {r["model"]: r.get("details", []) for r in results_all},
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written → {out_path}")

    return output


if __name__ == "__main__":
    main()
