"""Combined evaluation summary across all fine-tuning methods.

Aggregates results from:
  - eval_format.py   → format-constraint compliance on ifeval-like data
  - eval_grounding.py → exact/substring/citation match on TriviaQA

Produces a side-by-side comparison table for all trained model variants.

Key insight
-----------
No single metric tells the full story.  Combining format and grounding metrics
reveals the trade-off:
  - SFT excels at format compliance (learns surface patterns from demos)
  - Grounding metrics reveal whether context is actually being used
  - Full FT vs. LoRA vs. QLoRA comparisons show the efficiency trade-off

Two modes
---------
  1. Load pre-computed JSON results (fast, requires prior eval runs)
  2. Run evals inline (slow, but self-contained — good for final reporting)

Usage
-----
  # Mode 1: load pre-computed results (recommended)
  python src/eval_summary.py \\
      --model_dirs results/sft-gpt2,results/lora-r8-alpha16,results/fullft-gpt2

  # Mode 2: run evals inline (re-generates all responses)
  python src/eval_summary.py \\
      --model_dirs results/sft-gpt2,results/lora-r8-alpha16 \\
      --data_dir data \\
      --run_evals \\
      --max_samples 50

  # Include base model comparison:
  python src/eval_summary.py \\
      --model_dirs results/sft-gpt2,results/lora-r8-alpha16 \\
      --base_model gpt2 \\
      --data_dir data \\
      --run_evals \\
      --max_samples 50

  # Single model (e.g., just checking your latest run):
  python src/eval_summary.py \\
      --model_dirs results/sft-gpt2 \\
      --data_dir data \\
      --run_evals
"""

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Result loading helpers
# ---------------------------------------------------------------------------

def _load_json_result(path: Path) -> dict | None:
    """Load a JSON result file, returning None if it doesn't exist."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _collect_format_result(model_dir: Path) -> dict | None:
    """Return cached format eval JSON for a model, or None if not found."""
    return _load_json_result(model_dir / "eval_format.json")


def _collect_grounding_result(model_dir: Path) -> dict | None:
    """Return cached grounding eval JSON for a model, or None if not found."""
    return _load_json_result(model_dir / "eval_grounding.json")


def _collect_training_summary(model_dir: Path) -> dict | None:
    """Return training summary JSON (written by each trainer), or None."""
    return _load_json_result(model_dir / "training_summary.json")


# ---------------------------------------------------------------------------
# Inline eval runners (used when --run_evals is set)
# ---------------------------------------------------------------------------

def _run_format_eval(model_dir: str, data_dir: str, max_samples: int,
                     cache_dir: str = None) -> dict:
    """Run eval_format.py inline and return its result dict."""
    from eval_format import main as format_main

    import sys
    old_argv = sys.argv
    sys.argv = [
        "eval_format.py",
        "--model_dir", model_dir,
        "--data_dir", data_dir,
        "--max_samples", str(max_samples),
    ]
    try:
        result = format_main()
    finally:
        sys.argv = old_argv

    out_path = Path(cache_dir or model_dir) / "eval_format.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def _run_grounding_eval(model_dir: str, data_dir: str, max_samples: int,
                        cache_dir: str = None) -> dict:
    """Run eval_grounding.py inline and return its result dict."""
    from eval_grounding import main as grounding_main

    import sys
    cache_path = Path(cache_dir or model_dir) / "eval_grounding.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    old_argv = sys.argv
    sys.argv = [
        "eval_grounding.py",
        "--ft_model_dir", model_dir,
        "--data_dir", data_dir,
        "--max_samples", str(max_samples),
        "--skip_base",
        "--output_json", str(cache_path),
    ]
    try:
        result = grounding_main()
    finally:
        sys.argv = old_argv
    return result


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def _bar(pct: float, width: int = 20) -> str:
    """ASCII progress bar for a percentage value."""
    filled = int(pct / 100 * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _print_comparison_table(rows: list[dict]) -> None:
    """Print a comparison table across all evaluated models.

    Each row dict has keys:
      label, method, trainable_pct, format_pct, em_pct, sub_pct, cit_pct
    """
    if not rows:
        print("  (no results to display)")
        return

    col_w = 30  # model label column width

    # ── header ───────────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("EVALUATION SUMMARY — Fine-Tuning Comparison")
    print(f"{'='*90}")
    print(f"  {'Model':<{col_w}} {'Method':<10} {'Train%':>7} "
          f"{'Format':>8} {'EM':>6} {'Sub':>6} {'Cit':>6}")
    print(f"  {'-'*col_w} {'-'*10} {'-'*7} {'-'*8} {'-'*6} {'-'*6} {'-'*6}")

    for r in rows:
        label   = r.get("label", "?")[-col_w:]
        method  = r.get("method", "?")[:10]
        train_p = r.get("trainable_pct", float("nan"))
        fmt_p   = r.get("format_pct", float("nan"))
        em_p    = r.get("em_pct", float("nan"))
        sub_p   = r.get("sub_pct", float("nan"))
        cit_p   = r.get("cit_pct", float("nan"))

        def _fmt(v):
            return f"{v:5.1f}%" if v == v else "  n/a "  # nan check

        print(f"  {label:<{col_w}} {method:<10} {_fmt(train_p):>7} "
              f"{_fmt(fmt_p):>8} {_fmt(em_p):>6} {_fmt(sub_p):>6} {_fmt(cit_p):>6}")

    print(f"{'='*90}")
    print()
    print("Column definitions:")
    print("  Train%  = % of model parameters that were trained")
    print("  Format  = % of ifeval-like examples that pass ALL format constraints")
    print("  EM      = Exact match on TriviaQA (normalised pred == gold alias)")
    print("  Sub     = Substring match on TriviaQA (alias appears in response)")
    print("  Cit     = Citation presence (verbatim context phrase in response)")


def _print_training_comparison(training_rows: list[dict]) -> None:
    """Print a training efficiency comparison table."""
    if not training_rows:
        return

    print(f"\n{'='*70}")
    print("TRAINING EFFICIENCY COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Model':<28} {'Method':<10} {'Params':>10} {'Time(s)':>8} {'Tok/s':>8}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

    for r in training_rows:
        label  = r.get("label", "?")[-28:]
        method = r.get("method", "?")[:10]
        params = r.get("trainable_params", 0)
        secs   = r.get("elapsed_sec", 0)
        tps    = r.get("tokens_per_sec", 0)
        print(f"  {label:<28} {method:<10} {params:>10,} {secs:>8.0f} {tps:>8.0f}")

    print(f"{'='*70}")


def _print_per_constraint_table(rows: list[dict]) -> None:
    """Print per-constraint format pass rates for all models side-by-side."""
    # Collect union of all constraint IDs across all models
    all_constraints: list[str] = []
    seen = set()
    for r in rows:
        for cid in r.get("per_constraint_pct", {}):
            if cid not in seen:
                all_constraints.append(cid)
                seen.add(cid)

    if not all_constraints:
        return

    labels = [r["label"] for r in rows]
    col_w  = 14  # width per model column
    cid_w  = 52  # constraint name column width

    print(f"\n{'='*90}")
    print("PER-CONSTRAINT FORMAT PASS RATES")
    print(f"{'='*90}")

    # Header
    header = f"  {'Constraint':<{cid_w}}"
    for lbl in labels:
        header += f" {lbl[-col_w:]:>{col_w}}"
    print(header)
    print(f"  {'-'*cid_w}" + f" {'-'*col_w}" * len(labels))

    def _fmt(v):
        return f"{v:5.1f}%" if v == v else "  n/a "

    for cid in sorted(all_constraints):
        line = f"  {cid:<{cid_w}}"
        for r in rows:
            v = r.get("per_constraint_pct", {}).get(cid, float("nan"))
            line += f" {_fmt(v):>{col_w}}"
        print(line)

    print(f"{'='*90}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate format + grounding evals into a comparison table"
    )
    parser.add_argument(
        "--model_dirs",
        default="results/sft-gpt2",
        help="Comma-separated list of model directories to include in the summary",
    )
    parser.add_argument(
        "--data_dir", default="data",
        help="Root data directory (needed when --run_evals is set)",
    )
    parser.add_argument(
        "--run_evals", action="store_true",
        help="Run eval_format.py and eval_grounding.py inline rather than loading "
             "cached JSON.  Slower but self-contained.",
    )
    parser.add_argument(
        "--max_samples", type=int, default=50,
        help="Number of examples per eval when --run_evals is set",
    )
    parser.add_argument(
        "--base_model", default="",
        help="Base (untuned) model ID or path to include as a baseline row "
             "(e.g. 'gpt2').  Leave empty to skip.",
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Write the combined summary table to this JSON file",
    )
    args = parser.parse_args()

    model_dirs = [d.strip() for d in args.model_dirs.split(",") if d.strip()]

    summary_rows       = []
    training_rows      = []

    # ── baseline row ─────────────────────────────────────────────────────────
    if args.base_model:
        base_name = args.base_model.replace("/", "-")
        base_cache = Path("results") / f"base-{base_name}"
        base_cache.mkdir(parents=True, exist_ok=True)

        base_fmt = _collect_format_result(base_cache)
        if base_fmt is None and args.run_evals:
            print(f"\nRunning format eval for base model ({args.base_model})...")
            base_fmt = _run_format_eval(args.base_model, args.data_dir,
                                        args.max_samples, cache_dir=str(base_cache))

        base_grd = _collect_grounding_result(base_cache)
        if base_grd is None and args.run_evals:
            print(f"\nRunning grounding eval for base model ({args.base_model})...")
            base_grd = _run_grounding_eval(args.base_model, args.data_dir,
                                           args.max_samples, cache_dir=str(base_cache))

        base_fmt_pct = base_fmt["all_pass_pct"] if base_fmt else float("nan")
        base_em = base_sub = base_cit = float("nan")
        if base_grd:
            for r in base_grd.get("results", []):
                base_em  = r.get("exact_match_pct", float("nan"))
                base_sub = r.get("substring_match_pct", float("nan"))
                base_cit = r.get("citation_present_pct", float("nan"))
                break

        summary_rows.append({
            "label": f"{args.base_model} (base)",
            "method": "base",
            "trainable_pct": 0.0,
            "format_pct": base_fmt_pct,
            "em_pct": base_em,
            "sub_pct": base_sub,
            "cit_pct": base_cit,
            "per_constraint_pct": base_fmt.get("per_constraint_pct", {}) if base_fmt else {},
        })

    for model_dir_str in model_dirs:
        model_dir = Path(model_dir_str)
        label = model_dir.name  # use directory name as short label

        # ── get or run format eval ───────────────────────────────────────────
        fmt_result = _collect_format_result(model_dir)
        if fmt_result is None and args.run_evals:
            print(f"\nRunning format eval for {label}...")
            fmt_result = _run_format_eval(model_dir_str, args.data_dir, args.max_samples)

        # ── get or run grounding eval ────────────────────────────────────────
        grd_result = _collect_grounding_result(model_dir)
        if grd_result is None and args.run_evals:
            print(f"\nRunning grounding eval for {label}...")
            grd_result = _run_grounding_eval(model_dir_str, args.data_dir, args.max_samples)

        # ── training summary ─────────────────────────────────────────────────
        train_summary = _collect_training_summary(model_dir)

        # ── extract metrics ──────────────────────────────────────────────────
        fmt_pct = fmt_result["all_pass_pct"] if fmt_result else float("nan")

        em_pct = sub_pct = cit_pct = float("nan")
        if grd_result:
            # Results list: first entry is the FT model, second (if present) is base
            for r in grd_result.get("results", []):
                if "base" not in r.get("model", "").lower():
                    em_pct  = r.get("exact_match_pct", float("nan"))
                    sub_pct = r.get("substring_match_pct", float("nan"))
                    cit_pct = r.get("citation_present_pct", float("nan"))
                    break

        method = train_summary.get("method", "sft") if train_summary else "sft"
        trainable_pct = train_summary.get("trainable_pct", float("nan")) if train_summary else float("nan")

        row = {
            "label": label,
            "method": method,
            "trainable_pct": trainable_pct,
            "format_pct": fmt_pct,
            "em_pct": em_pct,
            "sub_pct": sub_pct,
            "cit_pct": cit_pct,
            "per_constraint_pct": fmt_result.get("per_constraint_pct", {}) if fmt_result else {},
        }
        summary_rows.append(row)

        if train_summary:
            training_rows.append({
                "label": label,
                "method": method,
                "trainable_params": train_summary.get("trainable_params", 0),
                "elapsed_sec": train_summary.get("elapsed_sec", 0),
                "tokens_per_sec": train_summary.get("tokens_per_sec", 0),
            })

    # ── print tables ─────────────────────────────────────────────────────────
    _print_comparison_table(summary_rows)
    _print_per_constraint_table(summary_rows)
    _print_training_comparison(training_rows)

    # ── key takeaways ─────────────────────────────────────────────────────────
    print("\nKey takeaways:")
    print("  1. SFT (full FT or LoRA) improves Format compliance vs. the base model.")
    print("  2. Citation scores show whether the model uses the provided context.")
    print("  3. LoRA matches or approaches full FT quality at a fraction of the")
    print("     trainable parameters — demonstrating parameter efficiency.")
    print("  4. QLoRA adds 4-bit compression: similar quality, lower VRAM.")
    print("  5. Neither SFT nor LoRA resolves preference trade-offs — for that")
    print("     you need RLHF, DPO, or reward-model feedback.")

    # ── save combined JSON ────────────────────────────────────────────────────
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "summary": summary_rows,
                "training": training_rows,
            }, f, indent=2)
        print(f"\nCombined summary written → {out_path}")


if __name__ == "__main__":
    main()
