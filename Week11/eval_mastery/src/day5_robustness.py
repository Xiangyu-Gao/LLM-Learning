"""
day5_robustness.py — Distribution shift, adversarial prompts & calibration.
============================================================================

CORE INSIGHT
------------
"Evaluation must probe failure boundaries, not averages."

A model that achieves 90% on a clean benchmark can fail catastrophically
on inputs slightly outside the training distribution.  Three failure modes:

  1. ADVERSARIAL / OOD INPUTS
     Prompt injection, jailbreaks, and unusual phrasings exploit the gap
     between the distribution the model was trained on and the distribution
     it encounters in deployment.

  2. LONG CONTEXT DEGRADATION ("Lost in the Middle")
     Liu et al. 2023 showed that transformer models struggle to use
     information placed in the middle of long contexts.  Performance
     degrades as context length grows even when the answer is present.

  3. CALIBRATION
     Calibration is the alignment between a model's stated confidence
     and its empirical accuracy.  A well-calibrated model that says
     "70% confident" is right 70% of the time.
     LLMs are often overconfident (ECE > 0 with conf > acc), especially
     on out-of-distribution inputs.

     Abstention is the model's ability to say "I don't know" when it
     should not answer — a critical safety property.

EXPERIMENT
----------
A. Robustness sweep:
     Vulnerable model vs robust model under 4 attack types.
     Measure robustness_rate per attack type.

B. Long-context degradation:
     Score degrades as context length (filler sentences) increases.
     Plot accuracy vs context length curve.

C. Calibration:
     Simulate model confidence distributions.
     Plot reliability diagram (empirical acc vs mean confidence per bin).
     Report ECE for over-confident, well-calibrated, under-confident models.

OUTPUT
------
results/day5/day5_robustness.png         — attack type × model comparison
results/day5/day5_long_context.png       — accuracy vs context length
results/day5/day5_calibration.png        — reliability diagrams + ECE

INTERVIEW TAKEAWAYS
-------------------
Q: "How do you test for jailbreak robustness?"
A: Red-team with known attack families (DAN, role-play, prefix injection,
   base64 encoding), then measure refusal rate.  Use automated red-teaming
   (Constitutional AI / RLHF from adversarial prompts) to scale.
   Robustness is not binary — measure the fraction of attacks that succeed.

Q: "What is calibration in LLMs?"
A: A calibrated model's confidence = its empirical accuracy.
   Measured with ECE (expected calibration error) or reliability diagrams.
   LLMs post-RLHF tend to be overconfident because RLHF reward models
   reward high-confidence, hedging-free responses.

Q: "What is abstention and when should models refuse?"
A: Abstention = returning "I don't know" or "I'm not sure" rather than
   a guess.  Models should abstain when:
     • Confidence is below a threshold (calibration-based abstention)
     • The question is out of their knowledge domain
     • The query is harmful and refusal is appropriate
   Abstention reduces hallucination but increases unhelpfulness — the
   tradeoff is tunable by adjusting the confidence threshold.
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

from eval.adversarial import (                    # noqa: E402
    AttackType, generate_attacks, evaluate_robustness,
    make_vulnerable_model, make_robust_model, make_degrading_model,
)
from eval.metrics import expected_calibration_error   # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "day5")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Experiment A: Robustness under attack types ──────────────────────────────

def run_robustness(smoke: bool):
    print("\n[A] Adversarial Robustness Sweep")

    base_query   = "What is the capital of France?"
    n_per_attack = 5 if smoke else 10
    attack_types = [AttackType.NORMAL, AttackType.INJECTION,
                    AttackType.JAILBREAK, AttackType.EDGE_CASE]

    models = {
        "Vulnerable model": make_vulnerable_model(rng_seed=1),
        "Robust model":     make_robust_model(rng_seed=2),
    }

    all_results = {}    # model_name → {attack_type → robustness_rate}

    for model_name, model_fn in models.items():
        all_results[model_name] = {}
        for atype in attack_types:
            attacks = generate_attacks(base_query, atype, n=n_per_attack)
            result  = evaluate_robustness(model_fn, attacks)
            all_results[model_name][atype.value] = result.robustness_rate
            print(f"  {model_name:<22} {atype.value:<15}: "
                  f"{result.robustness_rate:.0%} robust")

    return all_results, [t.value for t in attack_types]


# ─── Experiment B: Long context degradation ──────────────────────────────────

def run_long_context(smoke: bool):
    print("\n[B] Long-Context Degradation")

    model       = make_degrading_model(rng_seed=3)
    base_query  = "What is the boiling point of water?"
    n_steps     = 8 if smoke else 15
    n_trials    = 10 if smoke else 30
    rng         = np.random.default_rng(5)

    filler_counts = np.linspace(0, 100, n_steps, dtype=int)
    mean_acc      = []
    std_acc       = []

    for fc in filler_counts:
        trial_correct = []
        for _ in range(n_trials):
            attacks = generate_attacks(base_query, AttackType.LONG_CONTEXT, n=1,
                                       rng_seed=int(rng.integers(0, 1000)))
            # Manually set filler count
            from eval.adversarial import _FILLER_SENTENCES
            filler = " ".join(
                _FILLER_SENTENCES[rng.integers(0, len(_FILLER_SENTENCES))]
                for _ in range(int(fc))
            )
            prompt   = (filler + " " + base_query).strip()
            response = model(prompt)
            correct  = "correct" in response.lower()
            trial_correct.append(float(correct))

        mean_acc.append(np.mean(trial_correct))
        std_acc.append(np.std(trial_correct))

    print(f"  Context 0 filler sentences: accuracy = {mean_acc[0]:.1%}")
    print(f"  Context {filler_counts[-1]} filler sentences: accuracy = {mean_acc[-1]:.1%}")
    print(f"  Degradation: {mean_acc[0] - mean_acc[-1]:.1%} absolute drop")

    return filler_counts, mean_acc, std_acc


# ─── Experiment C: Calibration ────────────────────────────────────────────────

def run_calibration(smoke: bool):
    print("\n[C] Model Calibration")

    rng     = np.random.default_rng(42)
    n       = 200 if not smoke else 80
    n_bins  = 10

    def make_confidences_and_labels(bias: float, noise: float):
        """
        Simulate model outputs.
        bias > 0  → overconfident (LLMs post-RLHF)
        bias < 0  → underconfident
        """
        true_probs = rng.uniform(0.1, 0.9, n)
        conf       = np.clip(true_probs + bias + rng.normal(0, noise, n), 0.01, 0.99)
        labels     = (rng.uniform(0, 1, n) < true_probs).astype(float)
        return conf, labels

    scenarios = {
        "Well calibrated\n(no bias)":    make_confidences_and_labels(0.0,  0.05),
        "Overconfident\n(post-RLHF)":    make_confidences_and_labels(0.15, 0.08),
        "Underconfident\n(over-hedged)": make_confidences_and_labels(-0.12, 0.07),
    }

    ece_values = {}
    for name, (conf, labels) in scenarios.items():
        ece = expected_calibration_error(conf, labels, n_bins=n_bins)
        ece_values[name] = ece
        short_name = name.split("\n")[0]
        print(f"  {short_name:<25}: ECE = {ece:.4f}")

    return scenarios, ece_values, n_bins


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_robustness(all_results, attack_types):
    n_attacks = len(attack_types)
    n_models  = len(all_results)
    x         = np.arange(n_attacks)
    width     = 0.35
    colors    = ["#e74c3c", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Day 5 — Robustness Under Attack Types\n"
                 "Robust model resists injection and jailbreaks; vulnerable model does not",
                 fontsize=11)

    for i, (model_name, results) in enumerate(all_results.items()):
        rates = [results[at] for at in attack_types]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, rates, width, label=model_name,
                      color=colors[i], edgecolor="white", linewidth=0.8, alpha=0.9)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{rate:.0%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(attack_types, rotation=15, ha="right")
    ax.set_ylabel("Robustness rate (behaved correctly)")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "day5_robustness.png"), dpi=150)
    plt.close(fig)
    print("\nSaved: day5_robustness.png")


def plot_long_context(filler_counts, mean_acc, std_acc):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Day 5 — Long-Context Degradation\n"
                 "'Lost in the Middle': accuracy drops as context grows",
                 fontsize=11)

    ax.plot(filler_counts, mean_acc, "o-", color="#3498db", linewidth=2.5, markersize=6)
    ax.fill_between(filler_counts,
                    np.array(mean_acc) - np.array(std_acc),
                    np.array(mean_acc) + np.array(std_acc),
                    alpha=0.2, color="#3498db", label="±1 std")
    ax.set_xlabel("Number of filler sentences added before query")
    ax.set_ylabel("Accuracy (fraction correct)")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, label="chance")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "day5_long_context.png"), dpi=150)
    plt.close(fig)
    print("Saved: day5_long_context.png")


def plot_calibration(scenarios, ece_values, n_bins):
    n_sc   = len(scenarios)
    fig, axes = plt.subplots(1, n_sc, figsize=(15, 5))
    fig.suptitle("Day 5 — Reliability Diagrams (Calibration)\n"
                 "Perfect calibration: diagonal line.  Overconfidence: curve below diagonal.",
                 fontsize=11)

    bins = np.linspace(0, 1, n_bins + 1)

    for ax, (name, (conf, labels)) in zip(axes, scenarios.items()):
        bin_accs  = []
        bin_confs = []
        bin_ns    = []

        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (conf >= lo) & (conf < hi)
            if mask.sum() == 0:
                continue
            bin_accs.append(labels[mask].mean())
            bin_confs.append(conf[mask].mean())
            bin_ns.append(mask.sum())

        ece = ece_values[name]
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect")
        ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.7, color="#3498db",
               label="Actual acc.")
        ax.plot(bin_confs, bin_confs, "ro", markersize=5, alpha=0.6,
                label="Mean conf.")

        # Shade calibration gap
        for bc, ba in zip(bin_confs, bin_accs):
            ax.plot([bc, bc], [bc, ba], color="#e74c3c", alpha=0.4, linewidth=2)

        ax.set_title(f"{name}\nECE = {ece:.4f}", fontsize=10)
        ax.set_xlabel("Mean predicted confidence")
        ax.set_ylabel("Empirical accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "day5_calibration.png"), dpi=150)
    plt.close(fig)
    print("Saved: day5_calibration.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(smoke: bool = False):
    print("\n" + "=" * 70)
    print(" Day 5 — Distribution Shift & Robustness")
    print("=" * 70)

    all_results, attack_types = run_robustness(smoke)
    filler_counts, mean_acc, std_acc = run_long_context(smoke)
    scenarios, ece_values, n_bins = run_calibration(smoke)

    plot_robustness(all_results, attack_types)
    plot_long_context(filler_counts, mean_acc, std_acc)
    plot_calibration(scenarios, ece_values, n_bins)

    print("\n" + "─" * 60)
    print("KEY INSIGHT: Evaluation must probe failure boundaries")
    print("  • Clean benchmark scores hide adversarial vulnerabilities")
    print("  • Context length is a distribution shift — test explicitly")
    print("  • Calibration separates confident-and-correct from")
    print("    confident-and-wrong (the dangerous case)")
    print("─" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(smoke=args.smoke)
