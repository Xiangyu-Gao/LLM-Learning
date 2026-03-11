"""Build (chosen, rejected) preference pairs for alignment training.

Key insight
-----------
Reward models need *comparative* signal: instead of "is this good?", they learn
"is this better than that?".  This script manufactures such pairs from the
Week 3 SFT datasets by applying deterministic quality heuristics:

  ifeval-like  →  pairs on FORMAT CORRECTNESS
      chosen  : the gold response that follows the stated format constraint
      rejected: a degraded version (truncated / uppercased / space-stripped)
                that violates the constraint

  trivia_qa    →  pairs on GROUNDEDNESS
      chosen  : answer that appears verbatim in the provided context snippet
      rejected: answer with the answer word replaced by a random distractor,
                making it clearly un-grounded

Why not human labels?
    Human labels are the gold standard, but they're expensive and slow.
    Heuristic pairs are far noisier but sufficient to bootstrap a reward
    model for research / coursework purposes.  Production systems (InstructGPT,
    Claude) use crowd-sourced or model-assisted comparisons.

Output
------
data/preferences/
  train.jsonl   – {"prompt": ..., "chosen": ..., "rejected": ...}
  eval.jsonl

Usage
-----
  python data/make_preferences.py \\
      --w3_data_dir ../Week3/fine_tuning_project/data \\
      --output_dir data/preferences \\
      --max_per_source 200

Interview Q&A
-------------
Q: Why not just SFT more?
A: SFT minimises cross-entropy over demonstrations; every demonstrated response
   is treated equally regardless of quality.  You can only guide the model
   *towards* the demonstrations — there is no mechanism to push it *away* from
   undesired completions.  Preference training introduces a contrastive signal:
   "this direction is better, that direction is worse."  That richer gradient
   allows the model to navigate trade-offs (conciseness vs. accuracy, refusal
   vs. helpfulness) that SFT cannot express.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Allow importing Week 3 dataset loaders directly
# ---------------------------------------------------------------------------
_W3_SRC = Path(__file__).resolve().parents[1].parent / "Week3" / "fine_tuning_project" / "src"
if _W3_SRC.exists():
    sys.path.insert(0, str(_W3_SRC))

# ---------------------------------------------------------------------------
# Format-degradation helpers
# ---------------------------------------------------------------------------

_DISTRACTORS = [
    "Napoleon Bonaparte", "Isaac Newton", "Marie Curie",
    "William Shakespeare", "Albert Einstein", "Leonardo da Vinci",
    "Christopher Columbus", "Julius Caesar",
]


def _degrade_response(response: str, seed: int = 0) -> str:
    """Return a clearly-inferior version of *response*.

    Strategies (applied in random order so the dataset is varied):
      - ALLCAPS  : hard to read, often fails format constraints
      - truncate : answer cut to half length  (loses information)
      - strip    : remove all whitespace padding / newlines
    """
    rng = random.Random(seed)
    choice = rng.choice(["caps", "truncate", "strip"])
    if choice == "caps":
        return response.upper()
    elif choice == "truncate":
        mid = max(1, len(response) // 2)
        return response[:mid] + "..."
    else:
        return " ".join(response.split())[:40]  # truncate to 40 chars of stripped text


def _distract_answer(answer: str, seed: int = 0) -> str:
    """Replace the answer token with an obviously wrong distractor."""
    rng = random.Random(seed)
    distractor = rng.choice(_DISTRACTORS)
    # If the answer already matches a distractor, pick another
    while distractor.lower() == answer.lower() and len(_DISTRACTORS) > 1:
        distractor = rng.choice(_DISTRACTORS)
    return distractor


# ---------------------------------------------------------------------------
# Pair builders
# ---------------------------------------------------------------------------

def _read_parquet_capped(folder: Path, limit: int | None) -> pd.DataFrame:
    """Read parquet files from *folder*, stopping as soon as *limit* rows are collected."""
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


def build_ifeval_pairs(data_dir: Path, limit: int | None = None) -> list[dict]:
    """Create format-correctness pairs from ifeval-like data.

    chosen  = gold response (follows the format instruction)
    rejected = deterministically degraded version (violates format)
    """
    df = _read_parquet_capped(data_dir / "ifeval-like-data-subset", limit)
    pairs = []
    for i, (_, row) in enumerate(df.iterrows()):
        prompt = row["instruction"]
        chosen = row["response"]
        rejected = _degrade_response(chosen, seed=i)
        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "source": "ifeval",
        })
    print(f"  ifeval pairs  : {len(pairs):>6,}")
    return pairs


_CONTEXT_CHARS = 600


def _extract_context(row: pd.Series) -> str:
    """Return the best available grounding context for a TriviaQA row."""
    wiki_pages = list(row["entity_pages"]["wiki_context"])
    wiki_text = " ".join(p for p in wiki_pages if p).strip()
    if wiki_text:
        return wiki_text[:_CONTEXT_CHARS]
    search_ctxs = list(row["search_results"]["search_context"])
    search_text = " ".join(c for c in search_ctxs[:2] if c).strip()
    return search_text[:_CONTEXT_CHARS]


def build_trivia_pairs(data_dir: Path, limit: int | None = None) -> list[dict]:
    """Create groundedness pairs from TriviaQA data.

    chosen  = correct answer grounded in the provided context
    rejected = clearly wrong distractor (ungrounded)
    """
    df = _read_parquet_capped(data_dir / "trivia_qa-subset", limit)
    pairs = []
    for i, (_, row) in enumerate(df.iterrows()):
        aliases = row["answer"]["aliases"]
        answer = str(aliases[0]) if len(aliases) > 0 else "Unknown"
        context = _extract_context(row)

        if context:
            prompt = (
                f"Use the following context to answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {row['question']}"
            )
        else:
            prompt = row["question"]

        chosen = answer
        rejected = _distract_answer(answer, seed=i)

        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "source": "trivia_qa",
        })
    print(f"  trivia_qa pairs: {len(pairs):>5,}")
    return pairs


# ---------------------------------------------------------------------------
# Split & save
# ---------------------------------------------------------------------------

def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  Saved {len(records):>6,} records → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build preference pairs from Week 3 SFT datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--w3_data_dir",
        default="/mnt/sdb/PROJECTS/LLM-Learning/Week3/fine_tuning_project/data",
        help="Path to Week 3 data/ directory (contains ifeval-like-data-subset, trivia_qa-subset)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/preferences",
        help="Directory to write train.jsonl and eval.jsonl",
    )
    parser.add_argument(
        "--max_per_source",
        type=int,
        default=None,
        help="Cap rows read per source dataset (useful for quick tests)",
    )
    parser.add_argument(
        "--eval_fraction",
        type=float,
        default=0.1,
        help="Fraction of pairs to reserve for evaluation",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.w3_data_dir)
    if not data_dir.exists():
        # Try relative to script location
        data_dir = Path(__file__).parent.parent.parent.parent / "Week3" / "fine_tuning_project" / "data"
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Week 3 data not found at {args.w3_data_dir}. "
            "Pass --w3_data_dir pointing to the Week 3 data/ folder."
        )

    print(f"Loading data from: {data_dir}")
    lim = args.max_per_source
    all_pairs = []
    all_pairs += build_ifeval_pairs(data_dir, limit=lim)
    all_pairs += build_trivia_pairs(data_dir, limit=lim)

    # Shuffle deterministically
    rng = random.Random(args.seed)
    rng.shuffle(all_pairs)

    # Train / eval split
    n_eval = max(1, int(len(all_pairs) * args.eval_fraction))
    eval_pairs = all_pairs[:n_eval]
    train_pairs = all_pairs[n_eval:]

    print(f"\nTotal pairs  : {len(all_pairs):>6,}")
    print(f"  train      : {len(train_pairs):>6,}")
    print(f"  eval       : {len(eval_pairs):>6,}")

    out_dir = Path(args.output_dir)
    save_jsonl(train_pairs, out_dir / "train.jsonl")
    save_jsonl(eval_pairs, out_dir / "eval.jsonl")

    print("\nDone.  Sample preference pair:")
    sample = train_pairs[0]
    print(f"  prompt   : {sample['prompt'][:80]}...")
    print(f"  chosen   : {sample['chosen'][:60]}")
    print(f"  rejected : {sample['rejected'][:60]}")


if __name__ == "__main__":
    main()
