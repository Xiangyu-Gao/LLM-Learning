"""Format-loss probe for ifeval-like data.

Measures what percentage of model outputs satisfy each format constraint
specified in the ifeval-like dataset's `instruction_id_list` / `kwargs` fields.

Key insight
-----------
SFT (behavior cloning) learns to *mimic* the format of training demonstrations
very well.  This probe makes that concrete: a freshly fine-tuned model should
score much higher than a base model on format constraints, while still failing
at *preference* questions (where format compliance is neither necessary nor
sufficient).

Constraint types checked
------------------------
  language:response_language              ASCII-ratio heuristic for English
  length_constraints:number_sentences     sentence count vs. relation/num_sentences
  detectable_content:number_placeholders  [PLACEHOLDER] spans via regex
  detectable_format:number_bullet_lists   lines starting with *, -, or •
  detectable_format:number_sections       "SECTION N" headers
  detectable_format:number_highlighted_sections   *highlighted* spans
  detectable_format:title                 Markdown # heading
  startend:nth_paragraph_first_word       first word of N-th paragraph == starter

Usage
-----
  # Evaluate a fine-tuned model:
  python src/eval_format.py \\
      --model_dir results/sft-gpt2 \\
      --data_dir data \\
      --max_samples 100

  # Sanity-check with gold responses (should score ~100%):
  python src/eval_format.py \\
      --use_gold_responses \\
      --data_dir data \\
      --max_samples 100

  # Evaluate a base model (should score much lower than the fine-tuned model):
  python src/eval_format.py \\
      --model_dir gpt2 \\
      --data_dir data \\
      --max_samples 100
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# ---------------------------------------------------------------------------
# Per-constraint checkers
# ---------------------------------------------------------------------------
# Each checker receives:
#   text  – the assistant response to evaluate
#   kw    – parsed kwargs dict from the dataset row


def check_response_language(text: str, kw: dict) -> bool:
    """Heuristic: English output should be >90% ASCII."""
    language = kw.get("language", "en")
    if language != "en":
        return True  # skip non-English checks (no lightweight detector)
    if not text:
        return False
    ascii_ratio = sum(c.isascii() for c in text) / len(text)
    return ascii_ratio > 0.90


def check_number_sentences(text: str, kw: dict) -> bool:
    """Count sentences and compare against the required count + relation."""
    num = kw.get("num_sentences")
    if num is None:
        return True  # constraint not active for this example

    # Strip markdown emphasis markers (*) so periods inside *sentence.* spans
    # are exposed as proper sentence boundaries (e.g. "*text.* Next" → "text.  Next").
    clean = re.sub(r"\*+", " ", text).strip()

    # Split on:
    #   1. ASCII terminal punctuation (.!?) followed by whitespace
    #   2. CJK full-width sentence-ending punctuation (。！？) — no whitespace required
    #   3. Blank lines (paragraph breaks) — each paragraph counts as ≥1 sentence,
    #      which handles responses that use paragraphs without terminal punctuation.
    sentences = re.split(r"(?<=[.!?])\s+|(?<=[。！？])|\n{2,}", clean)
    sentences = [s for s in sentences if s.strip()]
    count = len(sentences)

    relation = kw.get("relation", "at least")

    if relation == "at least":
        return count >= num
    elif relation == "at most":
        return count <= num
    elif relation == "exactly":
        return count == num
    return True  # unknown relation — don't penalise


def check_number_placeholders(text: str, kw: dict) -> bool:
    """Count [PLACEHOLDER] spans (square brackets with content inside)."""
    num = kw.get("num_placeholders")
    if num is None:
        return True
    placeholders = re.findall(r"\[[^\[\]\n]+\]", text)
    return len(placeholders) >= num


def check_number_bullet_lists(text: str, kw: dict) -> bool:
    """Count bullet-point items (*, -, or • prefix), including inline bullets."""
    num = kw.get("num_bullets")
    if num is None:
        return True
    # Line-start bullets: standard Markdown list items at the start of a line.
    line_bullets = [ln for ln in text.splitlines() if re.match(r"^\s*[\*\-\•]\s+\S", ln)]
    # Inline bullets: "sentence. * item" — a bullet marker preceded by sentence-ending
    # punctuation + plain spaces (not newline) on the same line.  Using " +" instead of
    # "\s+" prevents double-counting items that also appear at a line start.
    inline_bullets = re.findall(r"(?<=[.!?]) +\* +\S", text)
    return len(line_bullets) + len(inline_bullets) >= num


def check_number_sections(text: str, kw: dict) -> bool:
    """Count 'Section N' style section headers (case-insensitive)."""
    num = kw.get("num_sections")
    if num is None:
        return True
    splitter = kw.get("section_spliter", "Section")  # note: dataset typo "spliter"
    pattern = rf"{re.escape(splitter)}\s*\d+"
    sections = re.findall(pattern, text, re.IGNORECASE)
    return len(sections) >= num


def check_highlighted_sections(text: str, kw: dict) -> bool:
    """Count *highlighted* inline spans (single-star markdown emphasis)."""
    num = kw.get("num_highlights")
    if num is None:
        return True
    # Match *content* spans that don't span newlines.
    highlights = re.findall(r"\*[^*\n]+\*", text)
    return len(highlights) >= num


def check_title(text: str, kw: dict) -> bool:
    """Check for a title.

    The ifeval-like dataset uses the <<Title>> double-angle-bracket convention,
    not Markdown headings.  Both formats are accepted.
    """
    # <<Title>> format (used by the dataset)
    if re.search(r"<<[^<>\n]+>>", text):
        return True
    # Markdown heading fallback
    return bool(re.search(r"^#{1,6}\s+\S", text, re.MULTILINE))


def check_nth_paragraph_first_word(text: str, kw: dict) -> bool:
    """The N-th paragraph's first word must equal the required starter."""
    starter: Optional[str] = kw.get("starter")
    if not starter:
        return True  # constraint not applicable

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    # kwarg uses 1-based paragraph index; default to first paragraph
    nth = kw.get("nth_paragraph", 1)
    if nth > len(paragraphs):
        return False

    first_word = paragraphs[nth - 1].split()[0] if paragraphs[nth - 1].split() else ""
    # Strip trailing punctuation before comparing
    first_word = first_word.rstrip(".,!?:;")
    starter_clean = starter.rstrip(".,!?:;")
    return first_word.lower() == starter_clean.lower()


# Map dataset constraint IDs → checker functions
CONSTRAINT_CHECKERS = {
    "language:response_language": check_response_language,
    "length_constraints:number_sentences": check_number_sentences,
    "detectable_content:number_placeholders": check_number_placeholders,
    "detectable_format:number_bullet_lists": check_number_bullet_lists,
    "detectable_format:number_sections": check_number_sections,
    "detectable_format:number_highlighted_sections": check_highlighted_sections,
    "detectable_format:title": check_title,
    "startend:nth_paragraph_first_word": check_nth_paragraph_first_word,
}


def evaluate_response(text: str, instruction_ids: list, kw: dict) -> dict[str, Optional[bool]]:
    """Return a {constraint_id: pass/fail/None} dict for one response.

    None means the constraint type is not implemented — it is excluded from
    aggregate statistics rather than counted as a failure.
    """
    results: dict[str, Optional[bool]] = {}
    for cid in instruction_ids:
        checker = CONSTRAINT_CHECKERS.get(cid)
        results[cid] = checker(text, kw) if checker is not None else None
    return results


# ---------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = "<|user|>\n{instruction}\n<|assistant|>\n"


def generate_responses(
    model_dir: str,
    instructions: list[str],
    max_new_tokens: int,
) -> list[str]:
    """Load the model and generate one response per instruction."""
    print(f"Loading model from {model_dir}...")
    adapter_cfg_path = Path(model_dir) / "adapter_config.json"
    if adapter_cfg_path.exists():
        with open(adapter_cfg_path) as f:
            base_model_name = json.load(f)["base_model_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, model_dir)
        model = model.merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Reset max_length in generation_config so it doesn't conflict with
    # max_new_tokens (the saved checkpoint may have a small max_length set).
    model.generation_config.max_length = None

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    responses = []
    for i, instruction in enumerate(instructions, 1):
        prompt = PROMPT_TEMPLATE.format(instruction=instruction)
        raw = gen(prompt, max_new_tokens=max_new_tokens, max_length=None)[0]["generated_text"]

        # Strip the prompt prefix to isolate the assistant's reply.
        assistant_text = raw[len(prompt):]
        # Truncate at the next user turn (in case generation runs long).
        assistant_text = assistant_text.split("<|user|>")[0].strip()
        responses.append(assistant_text)

        if i % 10 == 0:
            print(f"  Generated {i}/{len(instructions)} responses...")

    return responses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict:
    parser = argparse.ArgumentParser(description="Format probe on ifeval-like subset")
    parser.add_argument(
        "--model_dir", default="results/sft-gpt2",
        help="Fine-tuned (or base) model directory",
    )
    parser.add_argument("--data_dir", default="data")
    parser.add_argument(
        "--max_samples", type=int, default=100,
        help="Number of ifeval-like examples to evaluate",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="Max tokens to generate per response",
    )
    parser.add_argument(
        "--use_gold_responses", action="store_true",
        help="Evaluate gold responses instead of generating (sanity check — should score ~100%)",
    )
    args = parser.parse_args()

    # ── load data ────────────────────────────────────────────────────────────
    # Read only as many rows as needed — stop opening parquet files once we
    # have enough, so --max_samples 50 doesn't force-load all 165 k rows.
    data_dir = Path(args.data_dir)
    files = sorted((data_dir / "ifeval-like-data-subset").glob("*.parquet"))
    frames: list[pd.DataFrame] = []
    collected = 0
    for f in files:
        chunk = pd.read_parquet(f)
        need = args.max_samples - collected
        frames.append(chunk.head(need))
        collected += len(frames[-1])
        if collected >= args.max_samples:
            break
    df = pd.concat(frames, ignore_index=True).head(args.max_samples)
    print(f"Loaded {len(df)} ifeval-like examples.")

    # ── get responses ────────────────────────────────────────────────────────
    if args.use_gold_responses:
        print("Using GOLD responses (sanity check — checkers should score ~100%).")
        responses = df["response"].tolist()
    else:
        responses = generate_responses(
            args.model_dir,
            df["instruction"].tolist(),
            args.max_new_tokens,
        )

    # ── evaluate ─────────────────────────────────────────────────────────────
    per_constraint: dict[str, list[bool]] = {}
    all_pass_count = 0

    for response, row in zip(responses, df.itertuples(index=False)):
        try:
            ids = list(row.instruction_id_list)
        except TypeError:
            ids = []
        try:
            kw = json.loads(row.kwargs) if isinstance(row.kwargs, str) else dict(row.kwargs)
        except Exception:
            kw = {}

        results = evaluate_response(response, ids, kw)

        # A row "passes" only if every implemented constraint passes.
        known_results = [v for v in results.values() if v is not None]
        if known_results and all(known_results):
            all_pass_count += 1

        for cid, passed in results.items():
            if passed is not None:
                per_constraint.setdefault(cid, []).append(passed)

    # ── report ───────────────────────────────────────────────────────────────
    n = len(df)
    print(f"\n{'='*60}")
    label = "GOLD" if args.use_gold_responses else args.model_dir
    print(f"Format probe  |  {label}")
    print(f"{'='*60}")
    print(f"  Examples evaluated          : {n}")
    print(f"  ALL constraints pass        : {all_pass_count/n*100:5.1f}%")
    print()
    print("  Per-constraint pass rate:")
    for cid, results in sorted(per_constraint.items()):
        pct = sum(results) / len(results) * 100
        bar = "#" * int(pct / 5)
        print(f"    {cid:<52} {pct:5.1f}%  {bar}")
    print(f"{'='*60}")

    # ── intuition note ───────────────────────────────────────────────────────
    print()
    print("Intuition: SFT = behavior cloning")
    print("  ✓ Great at format/style  — the model memorises surface patterns")
    print("    from demonstrations (bullet lists, section headers, placeholders).")
    print("  ✗ Weak at preference trade-offs — there is no signal telling the")
    print("    model *which* of many valid continuations is *better*; for that")
    print("    you need RLHF / DPO / reward-model feedback.")

    return {
        "n": n,
        "all_pass_pct": all_pass_count / n * 100,
        "per_constraint_pct": {
            k: sum(v) / len(v) * 100 for k, v in per_constraint.items()
        },
    }


if __name__ == "__main__":
    main()
