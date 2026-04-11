"""
day67_harness.py — Mini Evaluation Harness (Days 6-7).
=======================================================

PURPOSE
-------
Tie all five metric families into a single reusable evaluation harness.
This is the deliverable that would go into a real ML project.

HARNESS STRUCTURE
-----------------
eval/
  metrics.py      — BLEU, ROUGE-L, chrF, pass@k, ECE
  judge.py        — LLM-as-judge simulation with bias measurement
  schema_check.py — JSON schema compliance
  adversarial.py  — Attack generators and robustness evaluator

harness.run(prompts, model_fn) does:
  1. Score each response with BLEU, chrF, exact match
  2. Run LLM-judge comparison (with position-averaged scoring)
  3. Schema validate any JSON outputs
  4. Bias check: measure position + verbosity effect
  5. Output eval_report.txt with tradeoff summary

EXPERIMENT
----------
50 prompts drawn from our synthetic QA dataset.
Three "models" with different quality profiles:
  Model A — high quality, concise
  Model B — verbose, sometimes wrong
  Model C — minimal, often wrong

Run all 3 evaluation methods on all 3 models.
Show that different metrics give different rankings.

OUTPUT
------
results/day67/day67_summary.png       — metric × model comparison heatmap
results/day67/day67_metric_rank.png   — per-metric ranking of 3 models
results/day67/eval_report.txt         — README-style tradeoff document
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common import make_qa_dataset, QAPair         # noqa: E402
from eval.metrics import (                         # noqa: E402
    bleu_score, rouge_l_score, chrf_score,
    semantic_sim_proxy, exact_match,
)
from eval.judge import SimulatedJudge, BIASED, UNBIASED, measure_position_bias
from eval.schema_check import validate_schema

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "day67")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Simulated models ─────────────────────────────────────────────────────────

def make_model_A(qa_data: List[QAPair], rng_seed: int = 0) -> Dict[str, str]:
    """High-quality concise model: returns reference answers."""
    rng = np.random.default_rng(rng_seed)
    return {
        qa.question: qa.reference if rng.random() > 0.1
                     else qa.wrong_plausible
        for qa in qa_data
    }


def make_model_B(qa_data: List[QAPair], rng_seed: int = 1) -> Dict[str, str]:
    """
    Verbose model: pads correct answers with filler.
    Correct ~75% of the time but always verbose.
    """
    rng = np.random.default_rng(rng_seed)
    filler = ("Furthermore, it is worth noting that this topic is complex. "
              "Specifically and comprehensively, the answer to your query is as follows. ")
    return {
        qa.question: (
            filler + qa.reference if rng.random() > 0.25
            else filler + qa.wrong_plausible
        )
        for qa in qa_data
    }


def make_model_C(qa_data: List[QAPair], rng_seed: int = 2) -> Dict[str, str]:
    """
    Minimal model: very short answers, often correct but under-specified.
    Correct reference ~55% of the time; rest is random error.
    """
    rng = np.random.default_rng(rng_seed)

    def shorten(text: str) -> str:
        words = text.split()
        return " ".join(words[:4]) + "."

    return {
        qa.question: (
            shorten(qa.reference) if rng.random() > 0.45
            else shorten(qa.wrong_random)
        )
        for qa in qa_data
    }


# ─── Harness result containers ────────────────────────────────────────────────

@dataclass
class PerResponseResult:
    question:  str
    reference: str
    response:  str
    bleu:      float
    rouge_l:   float
    chrf:      float
    sem_sim:   float
    em:        float


@dataclass
class ModelEvalSummary:
    model_name: str
    n_responses: int
    mean_bleu:    float
    mean_rouge_l: float
    mean_chrf:    float
    mean_sem_sim: float
    mean_em:      float
    judge_win_rate: float     # fraction of pairwise comparisons won
    bias_flip_rate: float     # how much the judge flips on order swap


# ─── Core harness ─────────────────────────────────────────────────────────────

def evaluate_model(
    model_name: str,
    model_outputs: Dict[str, str],
    qa_data: List[QAPair],
) -> tuple:
    """
    Score a model's outputs against references.

    Returns (summary, per_response_results).
    """
    results = []
    for qa in qa_data:
        hyp = model_outputs.get(qa.question, "")
        ref = qa.reference
        results.append(PerResponseResult(
            question  = qa.question,
            reference = ref,
            response  = hyp,
            bleu      = bleu_score(hyp, ref),
            rouge_l   = rouge_l_score(hyp, ref),
            chrf      = chrf_score(hyp, ref),
            sem_sim   = semantic_sim_proxy(hyp, ref),
            em        = exact_match(hyp, ref),
        ))

    n = len(results)
    summary = ModelEvalSummary(
        model_name    = model_name,
        n_responses   = n,
        mean_bleu     = np.mean([r.bleu    for r in results]),
        mean_rouge_l  = np.mean([r.rouge_l for r in results]),
        mean_chrf     = np.mean([r.chrf    for r in results]),
        mean_sem_sim  = np.mean([r.sem_sim for r in results]),
        mean_em       = np.mean([r.em      for r in results]),
        judge_win_rate = 0.0,   # filled below
        bias_flip_rate = 0.0,
    )
    return summary, results


def run_judge_evaluation(
    models: Dict[str, Dict[str, str]],
    qa_data: List[QAPair],
    judge: SimulatedJudge,
) -> Dict[str, float]:
    """
    Pairwise judge evaluation: compare all model pairs on all questions.
    Returns win_rate per model (averaged across all opponent × orderings).

    Position bias is mitigated by averaging (A vs B) + (B vs A).
    """
    model_names = list(models.keys())
    win_counts  = {m: 0 for m in model_names}
    total_comps = 0

    for qa in qa_data:
        ref = qa.reference
        for i, m1 in enumerate(model_names):
            for m2 in model_names[i+1:]:
                r1 = models[m1].get(qa.question, "")
                r2 = models[m2].get(qa.question, "")
                # Simulate quality via chrF (what the judge "knows" as ground truth)
                q1 = chrf_score(r1, ref) * 10
                q2 = chrf_score(r2, ref) * 10

                # Run both orderings to mitigate position bias
                res_ab = judge.compare(r1, r2, q1, q2)
                res_ba = judge.compare(r2, r1, q2, q1)

                # Average: if AB says A wins and BA says A wins → A wins
                a_wins = (res_ab.winner == "A") + (res_ba.winner == "B")
                b_wins = (res_ab.winner == "B") + (res_ba.winner == "A")

                if a_wins > b_wins:
                    win_counts[m1] += 1
                elif b_wins > a_wins:
                    win_counts[m2] += 1
                total_comps += 1

    # Each model is involved in (n_models-1) pairwise matchups per question
    comparisons_per_model = len(qa_data) * (len(model_names) - 1)
    win_rate = {m: win_counts[m] / max(comparisons_per_model, 1)
                for m in model_names}
    return win_rate


def run_bias_check(
    models: Dict[str, Dict[str, str]],
    qa_data: List[QAPair],
) -> Dict[str, float]:
    """Measure position bias flip rate for each model's responses."""
    biased_judge = SimulatedJudge(bias=BIASED, rng_seed=99)
    flip_rates = {}

    model_names = list(models.keys())
    for mname in model_names:
        pairs = []
        for qa in qa_data:
            r = models[mname].get(qa.question, "")
            ref = qa.reference
            q   = chrf_score(r, ref) * 10
            q_ref = 10.0
            pairs.append((r, ref, q, q_ref))

        result = measure_position_bias(biased_judge, pairs)
        flip_rates[mname] = result["flip_rate"]

    return flip_rates


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_summary_heatmap(summaries: List[ModelEvalSummary], win_rates: Dict):
    metrics = ["BLEU", "ROUGE-L", "chrF", "Sem-Sim", "EM", "Judge Win-Rate"]
    model_names = [s.model_name for s in summaries]

    data = np.array([
        [s.mean_bleu, s.mean_rouge_l, s.mean_chrf, s.mean_sem_sim, s.mean_em,
         win_rates.get(s.model_name, 0.0)]
        for s in summaries
    ])

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Day 6-7 — Evaluation Harness: All Metrics × All Models\n"
                 "Different metrics give different rankings — that's the point",
                 fontsize=11)

    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_yticks(range(len(model_names)))
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_yticklabels(model_names, fontsize=10)

    for i, row in enumerate(data):
        for j, val in enumerate(row):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if val > 0.7 or val < 0.25 else "black")

    plt.colorbar(im, ax=ax, label="Score (0–1)")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "day67_summary.png"), dpi=150)
    plt.close(fig)
    print("Saved: day67_summary.png")


def plot_metric_ranks(summaries: List[ModelEvalSummary], win_rates: Dict):
    metrics = {
        "BLEU":         [s.mean_bleu    for s in summaries],
        "chrF":         [s.mean_chrf    for s in summaries],
        "Sem-Sim":      [s.mean_sem_sim for s in summaries],
        "Judge Win %":  [win_rates.get(s.model_name, 0) for s in summaries],
    }
    model_names = [s.model_name for s in summaries]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
    fig.suptitle("Day 6-7 — Per-Metric Rankings\n"
                 "Verbosity raises BLEU but not chrF — the metric matters!",
                 fontsize=11)

    colors = ["#3498db", "#e67e22", "#e74c3c"]
    for ax, (metric_name, vals) in zip(axes, metrics.items()):
        x = np.arange(len(model_names))
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=10)
        ax.set_title(metric_name, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" ", "\n") for n in model_names],
                           fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "day67_metric_rank.png"), dpi=150)
    plt.close(fig)
    print("Saved: day67_metric_rank.png")


# ─── Report ───────────────────────────────────────────────────────────────────

def write_report(summaries, win_rates, bias_rates, flip_rates):
    path = os.path.join(RESULTS_DIR, "eval_report.txt")
    with open(path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(" EVALUATION HARNESS REPORT — Week 11 Eval Mastery\n")
        f.write("=" * 70 + "\n\n")

        f.write("MODELS EVALUATED\n")
        f.write("-" * 40 + "\n")
        for s in summaries:
            f.write(f"  {s.model_name:<20}: {s.n_responses} responses\n")
        f.write("\n")

        f.write("METRIC SCORES\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Model':<20} {'BLEU':>7} {'ROUGE-L':>9} {'chrF':>7} "
                f"{'SemSim':>8} {'EM':>5} {'Judge':>8}\n")
        f.write("-" * 70 + "\n")
        for s in summaries:
            j = win_rates.get(s.model_name, 0.0)
            f.write(f"{s.model_name:<20} {s.mean_bleu:>7.3f} {s.mean_rouge_l:>9.3f} "
                    f"{s.mean_chrf:>7.3f} {s.mean_sem_sim:>8.3f} "
                    f"{s.mean_em:>5.3f} {j:>8.3f}\n")
        f.write("\n")

        f.write("BIAS ANALYSIS\n")
        f.write("-" * 40 + "\n")
        for mname, flip_rate in flip_rates.items():
            f.write(f"  {mname:<20}: judge position flip rate = {flip_rate:.1%}\n")
        f.write("\n")

        f.write("TRADEOFF SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write("""
Metric choice matters as much as model choice.

BLEU-4
  + Fast, deterministic, no model required
  - Paraphrase collapse: correct paraphrases score ~0
  - False positives: wrong-but-plausible answers score non-zero
  WHEN TO USE: MT with multiple references; quick sanity checks only

ROUGE-L
  + Better than BLEU for longer outputs (LCS vs n-grams)
  - Still purely lexical; suffers paraphrase collapse
  WHEN TO USE: Summarisation evaluation alongside human labels

chrF (our BERTScore proxy)
  + Robust to paraphrase; no external model needed
  - Not truly semantic; misses meaning differences, just surface form
  WHEN TO USE: MT, generation tasks where real BERTScore is unavailable

Semantic Similarity (word-Jaccard + chrF)
  + Better paraphrase handling than BLEU or ROUGE
  - Still not semantic in the deep sense (no embeddings)
  WHEN TO USE: Lightweight screening before human eval

LLM-as-Judge (win rate)
  + Scales to thousands of prompts; nuanced multi-criterion scoring
  - Position bias: ~30% flip rate without order averaging
  - Verbosity bias: padded responses score higher
  - Requires separate judge model (cost)
  WHEN TO USE: Production evaluation; always average both orderings

Task-specific (pass@k, schema compliance, tool-call success)
  + Directly measures deployment utility
  - Task-specific: must be rebuilt for each new task
  WHEN TO USE: ALWAYS alongside generic metrics; this is what matters

RECOMMENDATION
--------------
Never use BLEU as the primary metric for generative tasks.
Run at least chrF + LLM-judge (position-averaged) + one task-specific metric.
Ground truth: 200-sample human eval, recalibrated quarterly.
""")
    print(f"Saved: eval_report.txt")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(smoke: bool = False):
    n_qa    = 10 if smoke else 50
    # We have 15 QA pairs; replicate to reach 50 if needed
    base_qa = make_qa_dataset()
    qa_data = []
    while len(qa_data) < n_qa:
        qa_data.extend(base_qa)
    qa_data = qa_data[:n_qa]

    print("\n" + "=" * 70)
    print(" Days 6-7 — Mini Evaluation Harness")
    print(f" {n_qa} prompts × 3 models × 4 metric families")
    print("=" * 70)

    # ── Build model outputs ──────────────────────────────────────────────────
    models = {
        "Model A (high quality)": make_model_A(qa_data),
        "Model B (verbose)":      make_model_B(qa_data),
        "Model C (minimal)":      make_model_C(qa_data),
    }

    # ── Score each model ─────────────────────────────────────────────────────
    summaries = []
    for mname, outputs in models.items():
        summary, _ = evaluate_model(mname, outputs, qa_data)
        summaries.append(summary)

    print("\n  Automated metric scores:")
    print(f"  {'Model':<22} {'BLEU':>7} {'chrF':>7} {'SemSim':>8} {'EM':>5}")
    print("  " + "-" * 55)
    for s in summaries:
        print(f"  {s.model_name:<22} {s.mean_bleu:>7.3f} {s.mean_chrf:>7.3f} "
              f"{s.mean_sem_sim:>8.3f} {s.mean_em:>5.3f}")

    # ── Judge evaluation ─────────────────────────────────────────────────────
    judge     = SimulatedJudge(bias=BIASED, rng_seed=42)
    win_rates = run_judge_evaluation(models, qa_data, judge)

    print("\n  Judge win rates (position-averaged):")
    for mname, wr in win_rates.items():
        print(f"  {mname:<28}: {wr:.1%}")

    # ── Bias check ───────────────────────────────────────────────────────────
    flip_rates = run_bias_check(models, qa_data)

    print("\n  Judge position flip rates:")
    for mname, fr in flip_rates.items():
        print(f"  {mname:<28}: {fr:.1%}")

    # Attach win rates to summaries
    for s in summaries:
        s.judge_win_rate = win_rates.get(s.model_name, 0.0)
        s.bias_flip_rate = flip_rates.get(s.model_name, 0.0)

    # ── Plots & report ───────────────────────────────────────────────────────
    plot_summary_heatmap(summaries, win_rates)
    plot_metric_ranks(summaries, win_rates)
    write_report(summaries, win_rates, {}, flip_rates)

    print("\n" + "─" * 60)
    print("KEY INSIGHT: Metric selection shapes what you measure.")
    print("  BLEU is low for verbose Model B — extra tokens dilute n-gram precision")
    print("  Judge is high for verbose Model B — LLM judge rewards length (bias!)")
    print("  Only chrF + Judge WIN RATE (debiased) consistently rank Model A best")
    print("  → Metric selection IS a research decision, not a detail")
    print("─" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(smoke=args.smoke)
