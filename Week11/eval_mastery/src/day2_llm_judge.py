"""
day2_llm_judge.py — LLM-as-a-judge: bias leakage and unreliability.
====================================================================

CORE INSIGHT
------------
"LLM judges are not neutral evaluators."

GPT-4-style evaluation is powerful but systematically biased.
Three documented failure modes:

  1. POSITION BIAS   — the response presented first in the prompt
                       receives higher scores, independent of quality.
                       MT-Bench showed ~30% of flipped comparisons
                       change the winner just from order swapping.

  2. VERBOSITY BIAS  — longer responses are rated as higher quality
                       even when the extra tokens carry no information.
                       A padded response with filler sentences beats
                       a concise, correct response.

  3. SELF-PREFERENCE — models prefer responses written in their own
                       style.  GPT-4 rates GPT-4 responses higher;
                       Claude rates Claude-style responses higher.

EXPERIMENT
----------
We use SimulatedJudge with configurable bias weights to quantify
how much each bias type changes evaluation outcomes.

Trials:
  A. Position bias:  Compare 30 response pairs with same quality.
     Run (A, B) and (B, A).  Measure flip rate.
  B. Verbosity bias: Pad a response with 0, 5, 10, 20, 40 filler words.
     Measure score increase vs word count.
  C. Combined:       All biases active simultaneously.

OUTPUT
------
results/day2/day2_position_bias.png     — flip rate vs bias strength
results/day2/day2_verbosity_bias.png    — score vs word count
results/day2/day2_bias_decomposition.png— contribution of each bias

INTERVIEW TAKEAWAYS
-------------------
Q: "Is LLM-as-a-judge reliable?"
A: Useful as a scalable alternative to human evaluation, but not neutral.
   Position bias can be mitigated by averaging (A vs B) and (B vs A).
   Verbosity bias requires explicit length normalisation in the judge prompt.
   Self-preference is harder to remove; using a different judge model helps.

Q: "How do you reduce evaluation bias?"
A: (1) Randomise response order and average both orderings.
   (2) Instruct the judge to ignore length.
   (3) Use multiple independent judges and take a majority vote.
   (4) Calibrate the judge on human-rated examples.
   (5) Use pairwise ranking instead of absolute scores.
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

from common import make_qa_dataset                 # noqa: E402
from eval.judge import (                           # noqa: E402
    SimulatedJudge, BiasConfig, measure_position_bias,
    measure_verbosity_bias, UNBIASED, POS_ONLY, VERB_ONLY, BIASED,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "day2")
os.makedirs(RESULTS_DIR, exist_ok=True)

_FILLER = (
    "Furthermore, it is important to note that the subject matter is "
    "quite complex and multifaceted. Specifically, one must consider "
    "various aspects thoroughly. Notably, a comprehensive and detailed "
    "analysis reveals several important dimensions."
).split()


# ─── Build comparison pairs ───────────────────────────────────────────────────

def build_pairs(qa_data, seed: int = 42):
    """
    Build (resp_a, resp_b, quality_a, quality_b) pairs.

    Pairs:
      - EQUAL quality: both 7.0 → any winner is purely bias
      - UNEQUAL: quality_a=8, quality_b=5 → winner should be A
      - CLOSE: quality_a=6, quality_b=5.5
    """
    rng   = np.random.default_rng(seed)
    pairs = []
    for qa in qa_data:
        para = qa.paraphrases
        p1   = para[0] if para else qa.reference

        # Equal quality — winner should be random (no bias → 50/50)
        pairs.append((qa.reference, p1, 7.0, 7.0))

        # Clear quality difference
        pairs.append((qa.reference, qa.wrong_plausible, 8.5, 4.5))

        # Close quality
        pairs.append((qa.reference, p1, 6.2, 5.8))

    return pairs


# ─── Experiment A: Position bias ─────────────────────────────────────────────

def run_position_bias_experiment(pairs, smoke: bool):
    """
    Compare flip rate under UNBIASED vs POS_ONLY vs BIASED judge.
    """
    configs = {
        "Unbiased (pos_w=0)":   UNBIASED,
        "Pos-only (pos_w=1.5)": POS_ONLY,
        "Full bias (pos_w=1.5+verbosity+style)": BIASED,
    }
    results = {}

    for label, cfg in configs.items():
        judge  = SimulatedJudge(bias=cfg, rng_seed=7)
        result = measure_position_bias(judge, pairs)
        results[label] = result
        print(f"  {label}")
        print(f"    flip_rate        = {result['flip_rate']:.2%}")
        print(f"    bias_toward_first= {result['bias_toward_first']:.2%}")

    return results


# ─── Experiment B: Verbosity bias ─────────────────────────────────────────────

def run_verbosity_bias_experiment(base_response: str, smoke: bool):
    """
    Pad base_response with increasing filler words and score against itself.
    A neutral judge should see no score difference — bias makes longer win.
    """
    n_steps = 4 if smoke else 8
    configs  = {
        "Unbiased":  UNBIASED,
        "Verb-only": VERB_ONLY,
        "Full bias": BIASED,
    }
    all_results = {}

    for label, cfg in configs.items():
        judge   = SimulatedJudge(bias=cfg, rng_seed=3)
        # Run multiple trials per padding level to average out noise
        results = measure_verbosity_bias(
            judge, base_response, quality=7.0,
            padding_words=_FILLER, steps=n_steps,
            n_trials=20,
        )
        all_results[label] = results

    return all_results


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_position_bias(results):
    labels     = list(results.keys())
    flip_rates = [results[l]["flip_rate"] for l in labels]
    first_bias = [results[l]["bias_toward_first"] for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Day 2 — Position Bias in LLM-as-Judge", fontsize=13)

    x = np.arange(len(labels))
    colors = ["#2ecc71", "#e67e22", "#e74c3c"]

    ax1.bar(x, flip_rates, color=colors, edgecolor="white", linewidth=0.8)
    ax1.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title("Flip Rate\n(% of pairs where winner changes on order swap)")
    ax1.set_ylabel("Flip rate")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax1.set_ylim(0, 1.0)
    ax1.text(0.5, 0.92, "higher = more biased", ha="center",
             transform=ax1.transAxes, color="gray", fontsize=8)

    ax2.bar(x, first_bias, color=colors, edgecolor="white", linewidth=0.8)
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5,
                label="neutral (50%)")
    ax2.set_title("Preference for First Response\n(should be 50% if unbiased)")
    ax2.set_ylabel("Fraction of trials where position-A wins")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax2.set_ylim(0, 1.0)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "day2_position_bias.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: day2_position_bias.png")


def plot_verbosity_bias(all_results):
    fig, axes = plt.subplots(1, len(all_results), figsize=(15, 5), sharey=True)
    fig.suptitle("Day 2 — Verbosity Bias: Mean Score Gap (Verbose − Base) vs Padding\n"
                 "Unbiased: flat at 0.  Biased: grows with padding word count.",
                 fontsize=11)

    for ax, (label, results) in zip(axes, all_results.items()):
        n_words   = [r["n_padding_words"] for r in results]
        mean_gaps = [r["mean_gap"]        for r in results]

        ax.bar(n_words, mean_gaps, width=max(n_words) / len(n_words) * 0.7,
               color=["#e74c3c" if g > 0.15 else "#3498db" for g in mean_gaps],
               edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Filler words added")
        ax.set_ylabel("Mean score gap (verbose − base)")
        ax.set_xlim(left=-2)
        # Mark the "neutral" band
        ax.axhspan(-0.2, 0.2, alpha=0.05, color="gray", label="Noise band (±0.2)")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "day2_verbosity_bias.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Saved: day2_verbosity_bias.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(smoke: bool = False):
    n_qa     = 5 if smoke else 15
    qa_data  = make_qa_dataset(n=n_qa)
    pairs    = build_pairs(qa_data)

    print("\n" + "=" * 70)
    print(" Day 2 — LLM-as-a-Judge: Bias Leakage Experiment")
    print("=" * 70)

    # ── Experiment A ────────────────────────────────────────────────────────
    print("\n[A] Position Bias (flip rate when order is swapped)")
    pos_results = run_position_bias_experiment(pairs, smoke)

    # ── Experiment B ────────────────────────────────────────────────────────
    base_resp = qa_data[0].reference
    print(f"\n[B] Verbosity Bias (scoring padded vs base response)")
    print(f"    Base response: '{base_resp[:60]}...'")
    verb_results = run_verbosity_bias_experiment(base_resp, smoke)

    for label, results in verb_results.items():
        # Use the mean gap at the highest padding level (noise averages out)
        last = results[-1]
        mean_gap = last["mean_gap"]
        print(f"  {label:<30}: mean gap at max padding = {mean_gap:+.2f}")

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_position_bias(pos_results)
    plot_verbosity_bias(verb_results)

    # ── Key insight ──────────────────────────────────────────────────────────
    unbiased_flip  = pos_results["Unbiased (pos_w=0)"]["flip_rate"]
    biased_flip    = pos_results["Full bias (pos_w=1.5+verbosity+style)"]["flip_rate"]
    print("\n" + "─" * 60)
    print("KEY INSIGHT:")
    print(f"  Unbiased judge flip rate : {unbiased_flip:.1%}")
    print(f"  Biased  judge flip rate  : {biased_flip:.1%}")
    print(f"  → Bias alone causes {biased_flip - unbiased_flip:.1%} more winner changes.")
    print("\n  Mitigation strategy:")
    print("    1. Run (A,B) AND (B,A) — average both orderings")
    print("    2. Instruct judge to ignore response length")
    print("    3. Use multiple judges + majority vote")
    print("─" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(smoke=args.smoke)
