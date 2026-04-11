"""
day1_bleu_failure.py — Why BLEU / ROUGE fail for generative reasoning.
=======================================================================

CORE INSIGHT
------------
"Surface form ≠ semantic equivalence."

BLEU measures n-gram overlap between hypothesis and reference.
Two sentences can be semantically identical yet share almost no n-grams
if they are paraphrases.  This is called *paraphrase collapse*.

Conversely, a wrong-but-plausible answer can share many n-grams with the
reference and receive a deceptively high BLEU score.

EXPERIMENT
----------
For each of 15 synthetic QA pairs we score 5 candidate types:
  1. reference       — perfect answer (gold)
  2. paraphrase_1    — semantically equivalent, different words
  3. paraphrase_2    — second paraphrase
  4. wrong_plausible — wrong answer that shares lexical tokens
  5. wrong_random    — clearly wrong answer

Metrics evaluated:
  • BLEU-4
  • ROUGE-L
  • chrF (character n-gram F-score, our BERTScore proxy)
  • Exact match

FINDINGS
--------
Expected observations:
  • Paraphrases score ~0.0 BLEU (paraphrase collapse)
  • Paraphrases score ~0.5–0.7 chrF (semantic robustness)
  • wrong_plausible scores non-zero BLEU (false positive)
  • ROUGE-L falls between BLEU and chrF in paraphrase robustness

OUTPUT
------
results/day1/day1_metric_comparison.png — violin/box plots per metric × candidate type
results/day1/day1_paraphrase_scatter.png — BLEU vs chrF scatter coloured by type
results/day1/day1_summary.txt

INTERVIEW TAKEAWAY
------------------
Q: "Why is BLEU unreliable for reasoning tasks?"
A: BLEU is a bag-of-n-grams metric.  A correct paraphrase shares the
   underlying meaning but has different word order and synonym choices,
   producing near-zero n-gram overlap.  The metric punishes correct but
   creatively-worded answers.  For reasoning evaluation you want something
   that captures semantics, not surface form.

Q: "When would you still use BLEU?"
A: Translation tasks where reference translations are abundant (multiple
   BLEU references reduce paraphrase collapse).  Also as a fast sanity
   check: a very low BLEU on a generation task usually does indicate
   something is wrong, even if a high BLEU doesn't guarantee quality.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common import make_qa_dataset, smooth          # noqa: E402
from eval.metrics import (                          # noqa: E402
    bleu_score, rouge_l_score, chrf_score,
    semantic_sim_proxy, exact_match,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "day1")
os.makedirs(RESULTS_DIR, exist_ok=True)

CANDIDATE_TYPES = ["reference", "paraphrase_1", "paraphrase_2",
                   "wrong_plausible", "wrong_random"]

COLORS = {
    "reference":      "#2ecc71",
    "paraphrase_1":   "#3498db",
    "paraphrase_2":   "#1abc9c",
    "wrong_plausible":"#e67e22",
    "wrong_random":   "#e74c3c",
}


# ─── Build candidate answers ─────────────────────────────────────────────────

def build_candidates(qa_pairs):
    """
    Return a flat list of (question, reference, candidate, ctype) tuples.
    """
    rows = []
    for qa in qa_pairs:
        para = qa.paraphrases
        p1   = para[0] if para else qa.reference
        p2   = para[1] if len(para) > 1 else qa.reference

        rows.extend([
            (qa.question, qa.reference, qa.reference,       "reference"),
            (qa.question, qa.reference, p1,                 "paraphrase_1"),
            (qa.question, qa.reference, p2,                 "paraphrase_2"),
            (qa.question, qa.reference, qa.wrong_plausible, "wrong_plausible"),
            (qa.question, qa.reference, qa.wrong_random,    "wrong_random"),
        ])
    return rows


# ─── Score one candidate ─────────────────────────────────────────────────────

def score_candidate(hyp: str, ref: str) -> dict:
    return {
        "bleu":     bleu_score(hyp, ref),
        "rouge_l":  rouge_l_score(hyp, ref),
        "chrf":     chrf_score(hyp, ref),
        "sem_sim":  semantic_sim_proxy(hyp, ref),
        "em":       exact_match(hyp, ref),
    }


# ─── Main experiment ─────────────────────────────────────────────────────────

def run(smoke: bool = False):
    n_qa    = 5 if smoke else 15
    qa_data = make_qa_dataset(n=n_qa)
    rows    = build_candidates(qa_data)

    # Collect scores per candidate type
    scores_by_type = {ct: {"bleu": [], "rouge_l": [], "chrf": [], "sem_sim": []}
                      for ct in CANDIDATE_TYPES}

    all_rows_scored = []
    for _, ref, hyp, ctype in rows:
        s = score_candidate(hyp, ref)
        for metric in ("bleu", "rouge_l", "chrf", "sem_sim"):
            scores_by_type[ctype][metric].append(s[metric])
        all_rows_scored.append({**s, "ctype": ctype})

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" Day 1 — Metric Comparison: BLEU vs ROUGE-L vs chrF")
    print("=" * 70)
    header = f"{'Candidate Type':<20} {'BLEU':>7} {'ROUGE-L':>9} {'chrF':>7} {'SemSim':>8}"
    print(header)
    print("-" * 70)
    for ctype in CANDIDATE_TYPES:
        b  = np.mean(scores_by_type[ctype]["bleu"])
        rl = np.mean(scores_by_type[ctype]["rouge_l"])
        c  = np.mean(scores_by_type[ctype]["chrf"])
        s  = np.mean(scores_by_type[ctype]["sem_sim"])
        print(f"{ctype:<20} {b:>7.3f} {rl:>9.3f} {c:>7.3f} {s:>8.3f}")
    print("=" * 70)

    # KEY FINDINGS
    p1_bleu = np.mean(scores_by_type["paraphrase_1"]["bleu"])
    p1_chrf = np.mean(scores_by_type["paraphrase_1"]["chrf"])
    wp_bleu = np.mean(scores_by_type["wrong_plausible"]["bleu"])
    wp_chrf = np.mean(scores_by_type["wrong_plausible"]["chrf"])
    print(f"\nFINDING 1 — PARAPHRASE COLLAPSE (BLEU):")
    print(f"  Paraphrase-1 BLEU    = {p1_bleu:.3f}  ← near zero despite correct answer")
    print(f"  Wrong-plausible BLEU = {wp_bleu:.3f}  ← non-zero due to shared tokens")
    print(f"  → BLEU penalises correct paraphrases more than wrong answers!")
    print(f"\nFINDING 2 — chrF LIMITATION:")
    print(f"  Paraphrase-1 chrF    = {p1_chrf:.3f}  ← better than BLEU, but...")
    print(f"  Wrong-plausible chrF = {wp_chrf:.3f}  ← often HIGHER than paraphrase!")
    print(f"  → Correct paraphrases change word order; wrong answers keep same structure.")
    print(f"    Structure similarity ≠ semantic correctness.")
    print(f"\nCONCLUSION: ALL lexical metrics fail to distinguish correct paraphrases")
    print(f"  from wrong-but-plausible answers. Only semantic metrics (BERTScore,")
    print(f"  LLM-judge) can make this distinction.")

    # ── Figure 1: Bar chart per metric and candidate type ────────────────────
    metrics = ["bleu", "rouge_l", "chrf", "sem_sim"]
    metric_labels = ["BLEU-4", "ROUGE-L", "chrF", "Sem-Sim (proxy)"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)
    fig.suptitle("Day 1 — All Lexical Metrics Fail to Rank Correct Paraphrases Above Wrong Answers\n"
                 "(blue = correct paraphrase; orange = wrong-but-plausible — orange should be lower, but isn't)", fontsize=11)

    x = np.arange(len(CANDIDATE_TYPES))
    for ax, metric, label in zip(axes, metrics, metric_labels):
        vals = [np.mean(scores_by_type[ct][metric]) for ct in CANDIDATE_TYPES]
        errs = [np.std(scores_by_type[ct][metric])  for ct in CANDIDATE_TYPES]
        bars = ax.bar(x, vals, yerr=errs, capsize=4,
                      color=[COLORS[ct] for ct in CANDIDATE_TYPES],
                      edgecolor="white", linewidth=0.8)
        ax.set_title(label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(["ref", "para1", "para2", "wrong-p", "wrong-r"],
                           rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "day1_metric_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved: day1_metric_comparison.png")

    # ── Figure 2: BLEU vs chrF scatter ───────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.set_title("Day 1 — BLEU vs chrF Scatter\n"
                  "Wrong-plausible (orange) overlaps with paraphrases (blue) — lexical metrics can't separate them",
                  fontsize=10)

    for ctype in CANDIDATE_TYPES:
        bs = scores_by_type[ctype]["bleu"]
        cs = scores_by_type[ctype]["chrf"]
        ax2.scatter(bs, cs, label=ctype, color=COLORS[ctype], alpha=0.7,
                    s=60, edgecolors="white", linewidths=0.5)

    ax2.set_xlabel("BLEU-4 score", fontsize=11)
    ax2.set_ylabel("chrF score", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.axvline(0.1, color="gray", linestyle="--", alpha=0.4, linewidth=0.8,
                label="BLEU = 0.1")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)

    # Annotate the key regions
    ax2.annotate(
        "Paraphrase collapse:\ncorrect answer, BLEU ≈ 0\nchrF captures some signal",
        xy=(0.02, 0.44), xytext=(0.12, 0.25),
        fontsize=8, color="steelblue",
        arrowprops=dict(arrowstyle="->", color="steelblue"),
    )
    ax2.annotate(
        "Structural similarity:\nwrong answer shares chrF\nwith correct paraphrase!",
        xy=(0.07, 0.52), xytext=(0.25, 0.70),
        fontsize=8, color="darkorange",
        arrowprops=dict(arrowstyle="->", color="darkorange"),
    )

    plt.tight_layout()
    fig2.savefig(os.path.join(RESULTS_DIR, "day1_paraphrase_scatter.png"), dpi=150)
    plt.close(fig2)
    print("Saved: day1_paraphrase_scatter.png")

    # ── Write summary text ───────────────────────────────────────────────────
    summary_path = os.path.join(RESULTS_DIR, "day1_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Day 1 — Why Overlap-based Metrics Collapse for Generative Reasoning\n")
        f.write("=" * 70 + "\n\n")
        f.write("PARAPHRASE COLLAPSE:\n")
        f.write(f"  Correct paraphrases score BLEU = {p1_bleu:.3f}\n")
        f.write(f"  The same paraphrases score chrF = {p1_chrf:.3f}\n")
        f.write(f"  Gap: {p1_chrf - p1_bleu:.3f} — chrF is better than BLEU for paraphrases\n\n")
        f.write("CRITICAL LIMITATION — STRUCTURAL SIMILARITY:\n")
        f.write(f"  Wrong-but-plausible BLEU = {wp_bleu:.3f}\n")
        f.write(f"  Wrong-but-plausible chrF = {wp_chrf:.3f}\n")
        f.write(f"  Correct paraphrase chrF  = {p1_chrf:.3f}\n")
        f.write(f"  → wrong_plausible chrF {'>' if wp_chrf > p1_chrf else '<='} paraphrase_1 chrF\n\n")
        f.write("  Wrong-but-plausible answers keep the SAME SENTENCE STRUCTURE as the\n")
        f.write("  reference (e.g. 'Paris is the X of France') while paraphrases CHANGE\n")
        f.write("  the structure. Character n-grams reward structural similarity, not\n")
        f.write("  semantic correctness.\n\n")
        f.write("CONCLUSION:\n")
        f.write("  ALL lexical metrics (BLEU, ROUGE-L, chrF) fail to distinguish\n")
        f.write("  correct paraphrases from wrong-but-plausible answers.\n")
        f.write("  Lexical metrics measure surface form; they cannot detect meaning.\n")
        f.write("  For production evaluation of generative models, use:\n")
        f.write("    1. BERTScore (contextual embeddings) for semantic fidelity\n")
        f.write("    2. LLM-as-judge for nuanced quality dimensions\n")
        f.write("    3. Task-specific metrics (pass@k, schema compliance) for utility\n")
    print(f"Saved: day1_summary.txt")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Quick validation: 5 QA pairs instead of 15")
    args = parser.parse_args()
    run(smoke=args.smoke)
