"""
judge.py — Simulated LLM-as-judge with configurable bias injection.
====================================================================
Purpose
-------
Demonstrate, without any real LLM API call, the three primary failure
modes of LLM-as-judge evaluation:

  1. Position bias   — the first response in a comparison is preferred
                       even when quality is equal.
  2. Verbosity bias  — longer responses are preferred regardless of
                       informational density.
  3. Self-preference — responses written in the judge's "style" score
                       higher.

Design
------
A SimulatedJudge scores a response as:

  score = base_quality
        + position_bonus × (is_first_position)
        + verbosity_bonus × log(len(words))
        + style_bonus × style_overlap(response, judge_style)
        + N(0, noise_std)                  # stochastic evaluation noise

base_quality is the ground-truth we control.  Everything else is bias.

By comparing judge decisions with and without bias, we can quantify
exactly how much the non-quality components affect outcomes.

Interview note
--------------
"LLM judges show ~30% position bias in published MT-Bench results.
 The fix: always run both orderings (A vs B) and (B vs A), then
 take a majority vote.  Verbosity bias requires explicit length
 normalisation in the prompt or a separate length penalty."
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class BiasConfig:
    """
    Controls how much each bias component affects judge scores.

    Setting a weight to 0.0 disables that bias — useful for ablations:
      unbiased   = BiasConfig(0, 0, 0)
      pos_only   = BiasConfig(position_weight=1.5)
      verb_only  = BiasConfig(verbosity_weight=0.3)
    """
    position_weight:  float = 1.5    # bonus for being listed first
    verbosity_weight: float = 0.3    # bonus per unit log(word_count)
    style_weight:     float = 0.8    # bonus for stylistic overlap with judge
    noise_std:        float = 0.5    # evaluation noise (irreducible randomness)
    judge_style_words: List[str] = field(
        default_factory=lambda: ["furthermore", "specifically", "notably",
                                 "comprehensive", "detailed", "thorough"]
    )


UNBIASED   = BiasConfig(0.0, 0.0, 0.0, noise_std=0.2)
BIASED     = BiasConfig(1.5, 0.3, 0.8, noise_std=0.5)
POS_ONLY   = BiasConfig(position_weight=1.5,  verbosity_weight=0.0, style_weight=0.0, noise_std=0.2)
VERB_ONLY  = BiasConfig(position_weight=0.0,  verbosity_weight=0.8, style_weight=0.0, noise_std=0.2)
STYLE_ONLY = BiasConfig(position_weight=0.0,  verbosity_weight=0.0, style_weight=2.0, noise_std=0.2)


# ─── Judge ────────────────────────────────────────────────────────────────────

@dataclass
class JudgeResult:
    score_a:  float
    score_b:  float
    winner:   str           # "A", "B", or "tie"
    margin:   float         # |score_a - score_b|


class SimulatedJudge:
    """
    Deterministic-except-noise judge that scores responses based on
    ground-truth quality plus configurable bias terms.

    Parameters
    ----------
    bias : BiasConfig
        Controls which biases are active.
    rng_seed : int
        Seed for reproducible noise.
    """

    def __init__(self, bias: BiasConfig = BIASED, rng_seed: int = 42):
        self.bias = bias
        self._rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------
    def _score_single(
        self,
        response:     str,
        quality:      float,   # ground truth [0, 10]
        is_position_a: bool,
    ) -> float:
        words      = response.split()
        word_count = max(len(words), 1)
        b = self.bias

        position_bonus  = b.position_weight if is_position_a else 0.0
        verbosity_bonus = b.verbosity_weight * np.log(word_count)
        style_hits      = sum(w.lower() in b.judge_style_words for w in words)
        style_bonus     = b.style_weight * (style_hits / word_count)
        noise           = self._rng.normal(0.0, b.noise_std)

        return quality + position_bonus + verbosity_bonus + style_bonus + noise

    # ------------------------------------------------------------------
    def compare(
        self,
        response_a: str,
        response_b: str,
        quality_a:  float,
        quality_b:  float,
    ) -> JudgeResult:
        """
        Judge response_a vs response_b.

        response_a is always shown as "first" (position A).
        To test position bias: swap the arguments and compare.
        """
        sa = self._score_single(response_a, quality_a, is_position_a=True)
        sb = self._score_single(response_b, quality_b, is_position_a=False)

        if abs(sa - sb) < 0.1:
            winner = "tie"
        elif sa > sb:
            winner = "A"
        else:
            winner = "B"

        return JudgeResult(score_a=sa, score_b=sb,
                           winner=winner, margin=abs(sa - sb))


# ─── Bias measurement utilities ───────────────────────────────────────────────

def measure_position_bias(
    judge: SimulatedJudge,
    pairs: List[Tuple[str, str, float, float]],   # (resp_a, resp_b, q_a, q_b)
) -> dict:
    """
    Run each pair twice: (A, B) and then (B, A).
    A position-biased judge will prefer the first argument more often
    than the second even when quality is equal.

    Returns
    -------
    dict with:
      flip_rate    — fraction of pairs where order swap changes winner
      bias_toward_first — fraction of trials where position-A wins
    """
    flip_count     = 0
    first_wins     = 0
    total          = len(pairs)

    for resp_a, resp_b, q_a, q_b in pairs:
        r1 = judge.compare(resp_a, resp_b, q_a, q_b)
        r2 = judge.compare(resp_b, resp_a, q_b, q_a)   # swapped

        if r1.winner == "A":
            first_wins += 1
        if r2.winner == "A":
            first_wins += 1

        # Flip = original said A wins, but after swap B wins (which was A)
        # i.e. the judgment changed just because of ordering
        orig_winner_id  = "A" if r1.winner == "A" else "B" if r1.winner == "B" else "tie"
        swap_winner_id  = "B" if r2.winner == "A" else "A" if r2.winner == "B" else "tie"
        if orig_winner_id != swap_winner_id and "tie" not in (orig_winner_id, swap_winner_id):
            flip_count += 1

    return {
        "flip_rate":        flip_count / total,
        "bias_toward_first": first_wins / (2 * total),
        "n_pairs":          total,
    }


def measure_verbosity_bias(
    judge: SimulatedJudge,
    base_response: str,
    quality: float,
    padding_words: List[str],
    steps: int = 5,
    n_trials: int = 20,
) -> List[dict]:
    """
    Pad base_response with increasing numbers of filler words and
    compare each padded version against the original.

    A verbosity-biased judge prefers the longer response even though
    the added words carry no information.

    n_trials comparisons are run at each padding level and averaged to
    reduce noise.  For an unbiased judge the mean gap should be ≈ 0;
    for a verbosity-biased judge the mean gap should grow with padding.
    """
    results = []
    pad_counts = np.linspace(0, len(padding_words), steps, dtype=int)

    for n_pad in pad_counts:
        padded = base_response + " " + " ".join(padding_words[:n_pad])
        padded = padded.strip()

        gaps      = []
        v_scores  = []
        b_scores  = []
        v_wins    = 0
        for _ in range(n_trials):
            r = judge.compare(padded, base_response, quality, quality)
            gaps.append(r.score_a - r.score_b)
            v_scores.append(r.score_a)
            b_scores.append(r.score_b)
            if r.winner == "A":
                v_wins += 1

        results.append({
            "n_padding_words":   int(n_pad),
            "mean_gap":          float(np.mean(gaps)),     # positive = verbose preferred
            "score_verbose":     float(np.mean(v_scores)),
            "score_base":        float(np.mean(b_scores)),
            "verbose_win_rate":  v_wins / n_trials,
        })

    return results
