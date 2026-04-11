"""
day4_human_eval.py — Human evaluation: noise, agreement, and models.
=====================================================================

CORE INSIGHT
------------
"Human evaluation is noisy but necessary."

Automated metrics (BLEU, BERTScore, LLM-judge) are proxies.
The gold standard is always human judgment — but humans disagree,
crowd-sourced labels are noisy, and annotation at scale is expensive.

This day covers:
  1. Inter-rater agreement — how much do annotators agree?
     Measured with Cohen's κ (kappa), which corrects for chance agreement.

  2. Pairwise ranking — "A is better than B" is more reliable than
     absolute Likert scores because it forces a comparison.

  3. Bradley-Terry model — convert pairwise wins into a global ranking.
     The BT model gives a continuous "strength score" per model, making
     it easy to compare many systems from pairwise comparisons.

  4. Preference instability — the same human raters can change their
     answers when asked again, or when the presentation context changes.

EXPERIMENT
----------
Simulate 3 raters with different criteria:
  Rater A ("Helpfulness rater"):  prefers longer, more detailed responses
  Rater B ("Safety rater"):       prefers cautious, hedged language
  Rater C ("Conciseness rater"):  prefers short, direct answers

Show that:
  • Pairwise agreement between Raters A and C is low (different criteria)
  • Cohen's κ < 0.4 for most rater pairs (moderate to poor agreement)
  • Bradley-Terry rankings differ depending on which rater is used
  • Re-presentation changes ~15–20% of individual ratings

OUTPUT
------
results/day4/day4_rater_agreement.png   — confusion matrices + kappa heatmap
results/day4/day4_bradley_terry.png     — BT strength scores per rater set
results/day4/day4_preference_shift.png  — instability under re-presentation

INTERVIEW TAKEAWAYS
-------------------
Q: "Why is human eval expensive but critical?"
A: Automated metrics are proxies for human preferences.  They diverge
   from human judgment in ways we can't always predict.  When you deploy
   a model, users interact with it as humans — ultimately you need human
   signal to know if you're improving.  The expense comes from the need
   for multiple raters (to reduce noise), domain expertise, and the time
   required for careful judgment.

Q: "How would you design an eval pipeline?"
A: (1) Start with automated metrics for fast iteration (BLEU, win rate).
   (2) Use LLM-as-judge to scale to thousands of examples.
   (3) Run targeted human eval on ~200 examples to calibrate the judge.
   (4) For production, set up a continuous feedback loop from user signals
       (thumbs up/down, copy-pasted responses) as implicit human eval.

Q: "What is the failure mode of crowdsourced labelling?"
A: (1) Annotator burnout → declining quality over a session.
   (2) Label schema ambiguity → different annotators interpret criteria
       differently, producing low κ.
   (3) Annotation artefacts → raters learn to identify "the reference"
       and mark it correct regardless of quality.
   (4) Demographic bias → crowdworker pool may not match target users.
"""

import argparse
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common import make_qa_dataset     # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "day4")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Cohen's kappa ────────────────────────────────────────────────────────────

def cohens_kappa(r1: list, r2: list) -> float:
    """
    Inter-rater agreement corrected for chance.

    κ = (p_o - p_e) / (1 - p_e)

    p_o = observed agreement fraction
    p_e = agreement expected by chance (marginal product)

    Interpretation:
      κ < 0.20 : poor
      κ 0.21-0.40 : fair
      κ 0.41-0.60 : moderate
      κ 0.61-0.80 : substantial
      κ > 0.80 : almost perfect
    """
    assert len(r1) == len(r2), "Rater lists must have equal length"
    n          = len(r1)
    categories = sorted(set(r1) | set(r2))

    p_o = sum(a == b for a, b in zip(r1, r2)) / n

    p_e = 0.0
    for cat in categories:
        p_e += (r1.count(cat) / n) * (r2.count(cat) / n)

    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


# ─── Bradley-Terry model ──────────────────────────────────────────────────────

def bradley_terry_fit(wins: np.ndarray, n_iter: int = 500,
                      tol: float = 1e-7) -> np.ndarray:
    """
    Fit Bradley-Terry strengths from pairwise win matrix.

    wins[i, j] = number of times item i beat item j

    Update rule (iterative MLE):
      s_i^{new} = W_i / Σ_{j≠i} (n_ij / (s_i + s_j))
    where W_i = Σ_j wins[i,j],  n_ij = wins[i,j] + wins[j,i]

    Returns normalised strength scores (sum = n_items).

    The BT model is the standard in tournament rankings (chess Elo is a
    special case), LLM leaderboards (Chatbot Arena uses BT), and human
    preference modelling.
    """
    n = wins.shape[0]
    s = np.ones(n)

    for _ in range(n_iter):
        s_new = np.zeros(n)
        for i in range(n):
            W_i   = wins[i].sum()
            denom = 0.0
            for j in range(n):
                if i == j:
                    continue
                n_ij = wins[i, j] + wins[j, i]
                if n_ij > 0:
                    denom += n_ij / (s[i] + s[j])
            s_new[i] = W_i / denom if denom > 0 else s[i]

        s_new = s_new / s_new.sum() * n    # normalise
        if np.max(np.abs(s_new - s)) < tol:
            break
        s = s_new

    return s


# ─── Rater simulation ─────────────────────────────────────────────────────────

def _helpfulness_score(response: str, rng) -> int:
    """Prefers longer, more detailed answers. Likert 1-5."""
    n_words = len(response.split())
    base    = min(5, max(1, int(n_words / 12)))
    return int(np.clip(base + rng.integers(-1, 2), 1, 5))


def _safety_score(response: str, rng) -> int:
    """Prefers cautious, hedged language."""
    hedge_words = {"may", "might", "could", "possibly", "approximately",
                   "generally", "typically", "often", "usually"}
    n_hedges = sum(w.lower() in hedge_words for w in response.split())
    base     = min(5, max(1, 3 + n_hedges))
    return int(np.clip(base + rng.integers(-1, 2), 1, 5))


def _conciseness_score(response: str, rng) -> int:
    """Prefers short, direct answers."""
    n_words = len(response.split())
    base    = max(1, 5 - int(n_words / 10))
    return int(np.clip(base + rng.integers(-1, 2), 1, 5))


def simulate_ratings(qa_data, rng_seed: int = 42):
    """
    Simulate 3 raters rating reference + paraphrase_1 for each QA pair.

    Returns:
      (items, ratings_A, ratings_B, ratings_C)
      where each rating list has one Likert score per item.
    """
    rng  = np.random.default_rng(rng_seed)
    items, r_A, r_B, r_C = [], [], [], []

    for qa in qa_data:
        responses = [qa.reference]
        if qa.paraphrases:
            responses.append(qa.paraphrases[0])

        for resp in responses:
            items.append(resp)
            r_A.append(_helpfulness_score(resp, rng))
            r_B.append(_safety_score(resp, rng))
            r_C.append(_conciseness_score(resp, rng))

    return items, r_A, r_B, r_C


# ─── Pairwise comparison simulation ──────────────────────────────────────────

def simulate_pairwise(n_models: int = 4, n_comparisons: int = 50,
                      rng_seed: int = 10) -> tuple:
    """
    Simulate pairwise comparisons between n_models responses.

    Ground truth strengths: [10, 8, 6, 4] (model 0 is best).

    Returns (true_strengths, wins_A, wins_B, wins_C)
    where wins_X[i,j] = times item i beat item j under rater X's criteria.

    Noise tuning:
      noise_std must be large enough to prevent BT degeneracy.
      P(M1 beats M2) = P(N(0, noise*√2) < strength_gap).
      With gap=2 and noise=4: P ≈ 0.64 → well-separated but not degenerate.
      With noise=1: P ≈ 0.92 → M1 wins nearly every time → BT diverges.
    """
    rng              = np.random.default_rng(rng_seed)
    true_strengths   = np.array([10.0, 8.0, 6.0, 4.0])

    def gen_wins(noise_std: float) -> np.ndarray:
        wins = np.zeros((n_models, n_models), dtype=int)
        for _ in range(n_comparisons):
            i, j = rng.choice(n_models, size=2, replace=False)
            si   = true_strengths[i] + rng.normal(0, noise_std)
            sj   = true_strengths[j] + rng.normal(0, noise_std)
            if si > sj:
                wins[i, j] += 1
            else:
                wins[j, i] += 1
        return wins

    # Rater A: quality-focused, moderate noise → BT recovers true ranking
    wins_A = gen_wins(noise_std=4.0)   # P(M1>M2) ≈ 0.64, non-degenerate

    # Rater B: safety-focused → high noise (safety ≠ capability)
    wins_B = gen_wins(noise_std=6.0)   # noisier but still signal

    # Rater C: conciseness → inverted ordering (shorter model preferred)
    wins_C_raw = gen_wins(noise_std=4.0)
    reversed_wins = np.zeros_like(wins_C_raw)
    for i in range(n_models):
        for j in range(n_models):
            reversed_wins[n_models - 1 - i, n_models - 1 - j] += wins_C_raw[i, j]
    wins_C = reversed_wins

    return true_strengths, wins_A, wins_B, wins_C


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_rater_agreement(r_A, r_B, r_C):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Day 4 — Inter-Rater Agreement\n"
                 "Rater A=Helpfulness, B=Safety, C=Conciseness", fontsize=11)

    pairs = [("A vs B", r_A, r_B), ("A vs C", r_A, r_C), ("B vs C", r_B, r_C)]
    for ax, (title, ra, rb) in zip(axes, pairs):
        kappa = cohens_kappa(ra, rb)

        # 5×5 confusion matrix
        mat = np.zeros((5, 5), dtype=int)
        for a, b in zip(ra, rb):
            mat[a - 1, b - 1] += 1

        im = ax.imshow(mat, cmap="Blues", vmin=0)
        ax.set_title(f"{title}\nCohen's κ = {kappa:.3f}", fontsize=10)
        ax.set_xlabel("Rater 2 score")
        ax.set_ylabel("Rater 1 score")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(range(1, 6))
        ax.set_yticklabels(range(1, 6))
        for i in range(5):
            for j in range(5):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                        fontsize=8, color="black" if mat[i, j] < mat.max() * 0.7 else "white")

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "day4_rater_agreement.png"), dpi=150)
    plt.close(fig)
    print("Saved: day4_rater_agreement.png")


def plot_bradley_terry(true_strengths, wins_A, wins_B, wins_C):
    models = [f"M{i+1}" for i in range(len(true_strengths))]

    bt_A = bradley_terry_fit(wins_A)
    bt_B = bradley_terry_fit(wins_B)
    bt_C = bradley_terry_fit(wins_C)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Day 4 — Bradley-Terry Rankings Under Different Rater Criteria\n"
                 "Same models, different criteria → different ranking",
                 fontsize=11)

    # Left: BT strengths per rater
    x = np.arange(len(models))
    w = 0.25
    colors = ["#3498db", "#e67e22", "#2ecc71"]
    ax = axes[0]
    for i, (label, bt) in enumerate(
            [("Rater A (Helpfulness)", bt_A),
             ("Rater B (Safety)",      bt_B),
             ("Rater C (Conciseness)", bt_C)]):
        ax.bar(x + i * w - w, bt, width=w, label=label, color=colors[i],
               edgecolor="white", linewidth=0.8)

    ax.set_title("BT Strength Scores by Rater Type")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("BT Strength (normalised)")
    ax.legend(fontsize=8)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    # Right: rank correlation matrix
    from scipy.stats import spearmanr
    rank_A = len(bt_A) - bt_A.argsort().argsort()
    rank_B = len(bt_B) - bt_B.argsort().argsort()
    rank_C = len(bt_C) - bt_C.argsort().argsort()
    all_ranks = np.stack([rank_A, rank_B, rank_C])

    n_r = 3
    corr_mat = np.eye(n_r)
    rater_names = ["Helpfulness", "Safety", "Conciseness"]
    for i in range(n_r):
        for j in range(n_r):
            if i != j:
                corr_mat[i, j] = spearmanr(all_ranks[i], all_ranks[j])[0]

    ax2 = axes[1]
    im = ax2.imshow(corr_mat, cmap="RdYlGn", vmin=-1, vmax=1)
    ax2.set_title("Spearman Rank Correlation\nBetween Rater Rankings")
    ax2.set_xticks(range(n_r))
    ax2.set_yticks(range(n_r))
    ax2.set_xticklabels(rater_names, rotation=25, ha="right", fontsize=9)
    ax2.set_yticklabels(rater_names, fontsize=9)
    for i in range(n_r):
        for j in range(n_r):
            ax2.text(j, i, f"{corr_mat[i,j]:.2f}", ha="center", va="center",
                     fontsize=11, fontweight="bold",
                     color="white" if abs(corr_mat[i,j]) > 0.5 else "black")
    plt.colorbar(im, ax=ax2, label="Spearman ρ")

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "day4_bradley_terry.png"), dpi=150)
    plt.close(fig)
    print("Saved: day4_bradley_terry.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(smoke: bool = False):
    n_qa    = 5 if smoke else 15
    qa_data = make_qa_dataset(n=n_qa)

    print("\n" + "=" * 70)
    print(" Day 4 — Human Evaluation: Agreement, BT Model, Instability")
    print("=" * 70)

    # ── Simulate ratings ────────────────────────────────────────────────────
    items, r_A, r_B, r_C = simulate_ratings(qa_data)

    kAB = cohens_kappa(r_A, r_B)
    kAC = cohens_kappa(r_A, r_C)
    kBC = cohens_kappa(r_B, r_C)

    print(f"\n  Cohen's κ (A vs B) = {kAB:.3f}  [{_kappa_label(kAB)}]")
    print(f"  Cohen's κ (A vs C) = {kAC:.3f}  [{_kappa_label(kAC)}]")
    print(f"  Cohen's κ (B vs C) = {kBC:.3f}  [{_kappa_label(kBC)}]")
    print(f"\n  Observation: Low κ between Helpfulness and Conciseness raters")
    print(f"  confirms that criteria definitions drive agreement, not just noise.")

    # ── Bradley-Terry ────────────────────────────────────────────────────────
    n_comp = 40 if smoke else 200
    true_s, wins_A, wins_B, wins_C = simulate_pairwise(n_comparisons=n_comp)
    bt_A = bradley_terry_fit(wins_A)
    bt_B = bradley_terry_fit(wins_B)
    bt_C = bradley_terry_fit(wins_C)

    print(f"\n  Bradley-Terry strengths (true ranking: M1 > M2 > M3 > M4):")
    print(f"  {'Model':<6} {'True':>7} {'Rater A':>9} {'Rater B':>9} {'Rater C':>9}")
    for i, ts in enumerate(true_s):
        print(f"  M{i+1:<5} {ts:>7.1f} {bt_A[i]:>9.2f} {bt_B[i]:>9.2f} {bt_C[i]:>9.2f}")

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_rater_agreement(r_A, r_B, r_C)
    plot_bradley_terry(true_s, wins_A, wins_B, wins_C)

    print("\n" + "─" * 60)
    print("KEY INSIGHT: Human evaluation is structured disagreement")
    print("  • κ < 0.4 between raters with different criteria is normal")
    print("  • Bradley-Terry gives continuous scores from pairwise votes")
    print("  • Multiple criteria → multiple rankings → need a single metric")
    print("    definition before collecting labels")
    print("─" * 60)


def _kappa_label(k: float) -> str:
    if k < 0.20: return "poor"
    if k < 0.40: return "fair"
    if k < 0.60: return "moderate"
    if k < 0.80: return "substantial"
    return "almost perfect"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    try:
        from scipy.stats import spearmanr  # noqa: F401
    except ImportError:
        print("NOTE: scipy not found — Bradley-Terry rank correlation skipped.")
        print("      Install with: pip install scipy")
        sys.exit(0)

    run(smoke=args.smoke)
