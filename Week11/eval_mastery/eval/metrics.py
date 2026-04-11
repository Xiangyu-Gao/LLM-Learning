"""
metrics.py — Core evaluation metrics for LLM outputs.
======================================================
Each metric family answers a different question:

  BLEU / ROUGE-L → "Does the output share n-grams / subsequences with
                    the reference?"
  Exact match    → "Is the output character-for-character identical?"
  chrF (proxy)   → "Do the character n-grams overlap?"  (BERTScore proxy)
  pass@k         → "Does at least one of k sampled solutions pass tests?"
  ECE            → "Is the model's confidence calibrated to its accuracy?"

Key insight
-----------
Lexical metrics (BLEU / ROUGE) measure surface form, not meaning.
They collapse on paraphrases: two sentences with identical semantics but
different word order score near zero.

chrF (character n-gram F-score) survives light paraphrase because character
sequences are more stable than word sequences.  It is NOT a semantic metric
but it is strictly more robust than BLEU.

Real BERTScore (bert-score package) uses contextual BERT embeddings and is
the current best cheap semantic proxy.  This file uses chrF as a runnable
stand-in that requires no external models.

To use real BERTScore (drop-in replacement for chrf_score):
    from bert_score import score as bert_score_fn
    P, R, F = bert_score_fn([hyp], [ref], lang="en")
    return F[0].item()
"""

from __future__ import annotations
from collections import Counter
from math import comb
from typing import List, Sequence
import numpy as np


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _word_tokens(text: str) -> List[str]:
    return text.lower().split()


def _word_ngrams(tokens: List[str], n: int) -> List[tuple]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _char_ngrams(text: str, n: int) -> List[str]:
    t = text.lower().replace(" ", "_")          # normalise spaces
    return [t[i:i+n] for i in range(len(t) - n + 1)]


# ─── BLEU ─────────────────────────────────────────────────────────────────────

def bleu_score(hypothesis: str, reference: str, max_n: int = 4) -> float:
    """
    Sentence BLEU (modified n-gram precision with brevity penalty).

    BLEU = BP × exp( (1/N) × Σ_n log p_n )

    where
      p_n  = clipped n-gram precision
      BP   = exp(1 - r/c) if c < r else 1.0
      c, r = hypothesis length, reference length

    Known failure mode: two paraphrases of the same fact can score near
    zero because their n-gram sets are disjoint despite identical meaning.
    """
    hyp_tok = _word_tokens(hypothesis)
    ref_tok = _word_tokens(reference)

    if not hyp_tok:
        return 0.0

    c, r = len(hyp_tok), len(ref_tok)
    bp = 1.0 if c >= r else np.exp(1.0 - r / c)

    log_sum = 0.0
    for n in range(1, max_n + 1):
        hyp_ng = Counter(_word_ngrams(hyp_tok, n))
        ref_ng = Counter(_word_ngrams(ref_tok, n))

        clipped = sum(min(cnt, ref_ng[gram]) for gram, cnt in hyp_ng.items())
        total   = max(len(hyp_tok) - n + 1, 0)

        if total == 0 or clipped == 0:
            return 0.0                    # any zero precision → BLEU = 0

        log_sum += np.log(clipped / total)

    return float(bp * np.exp(log_sum / max_n))


# ─── ROUGE-L ──────────────────────────────────────────────────────────────────

def _lcs_length(a: List[str], b: List[str]) -> int:
    """Length of the longest common subsequence of a and b."""
    m, n = len(a), len(b)
    # O(m×n) DP — fine for sentence-level evaluation
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def rouge_l_score(hypothesis: str, reference: str, beta: float = 1.2) -> float:
    """
    ROUGE-L F-score based on Longest Common Subsequence.

    P = LCS / |hyp|,  R = LCS / |ref|
    F = (1 + β²) × P × R / (β² × P + R)

    β > 1 emphasises recall (matching as much of the reference as possible).

    Slightly more robust than BLEU on paraphrases because LCS allows
    tokens to be non-contiguous, but still purely lexical.
    """
    hyp_tok = _word_tokens(hypothesis)
    ref_tok = _word_tokens(reference)

    if not hyp_tok or not ref_tok:
        return 0.0

    lcs = _lcs_length(hyp_tok, ref_tok)
    if lcs == 0:
        return 0.0

    P = lcs / len(hyp_tok)
    R = lcs / len(ref_tok)
    if P + R == 0:
        return 0.0
    return float((1 + beta**2) * P * R / (beta**2 * P + R))


# ─── Exact Match ──────────────────────────────────────────────────────────────

def exact_match(hypothesis: str, reference: str) -> float:
    """
    Normalised exact match: lowercase, strip whitespace, compare.

    Returns 1.0 if strings match exactly, 0.0 otherwise.

    Appropriate for: math answers, SQL outputs, structured tokens.
    Inappropriate for: open-ended generation, summarisation, dialogue.
    """
    return float(hypothesis.strip().lower() == reference.strip().lower())


# ─── chrF — Character n-gram F-score (BERTScore proxy) ───────────────────────

def chrf_score(hypothesis: str, reference: str, n: int = 3) -> float:
    """
    Character n-gram F-score (ChrF).

    More robust than BLEU for paraphrases because character n-grams
    are shared by morphological variants and reordered phrases.

    Used in machine translation evaluation; here as a lightweight
    proxy for semantic similarity when true embeddings are unavailable.

    P = |char-ngrams(hyp) ∩ char-ngrams(ref)| / |char-ngrams(hyp)|
    R = |char-ngrams(hyp) ∩ char-ngrams(ref)| / |char-ngrams(ref)|
    F = 2 × P × R / (P + R)

    NOT a semantic metric — two sentences about different topics can score
    high if they share many character sequences (e.g. "Paris" / "Baris").
    """
    hyp_ng = Counter(_char_ngrams(hypothesis, n))
    ref_ng = Counter(_char_ngrams(reference, n))

    if not hyp_ng or not ref_ng:
        return 0.0

    intersection = sum(min(cnt, ref_ng[gram]) for gram, cnt in hyp_ng.items())
    P = intersection / sum(hyp_ng.values())
    R = intersection / sum(ref_ng.values())

    if P + R == 0:
        return 0.0
    return float(2 * P * R / (P + R))


# Alias for clarity
def semantic_sim_proxy(hypothesis: str, reference: str) -> float:
    """
    Lightweight semantic similarity: average of chrF and word-Jaccard.

    chrF captures shared substrings; word-Jaccard captures vocabulary overlap.
    Together they are more tolerant of paraphrases than BLEU, while remaining
    interpretable and requiring zero external dependencies.
    """
    hyp_words = set(_word_tokens(hypothesis))
    ref_words  = set(_word_tokens(reference))

    if not hyp_words and not ref_words:
        return 1.0
    if not hyp_words or not ref_words:
        return 0.0

    jaccard = len(hyp_words & ref_words) / len(hyp_words | ref_words)
    chrf    = chrf_score(hypothesis, reference)
    return float(0.5 * jaccard + 0.5 * chrf)


# ─── pass@k ───────────────────────────────────────────────────────────────────

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Probability that at least one of k samples passes, given n total samples
    of which c are correct.  From Chen et al. 2021 (HumanEval).

    pass@k = 1 - C(n-c, k) / C(n, k)
           = 1 - ∏_{i=0}^{k-1} (n-c-i)/(n-i)   [numerically stable form]

    Parameters
    ----------
    n : total number of generated samples
    c : number of correct samples among n
    k : evaluation budget (how many you sample at test time)

    Intuition: you only fail if ALL k chosen samples are wrong.
    The probability of that shrinks as c / n grows or k grows.

    Examples
    --------
    n=10, c=1, k=1  → pass@1 ≈ 0.10   (one chance, one correct)
    n=10, c=1, k=5  → pass@5 ≈ 0.50   (five chances, one correct)
    n=10, c=5, k=1  → pass@1 ≈ 0.50
    """
    if n == 0:
        return 0.0
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0

    # Numerically stable: avoid large factorials
    prob_none_correct = 1.0
    for i in range(k):
        prob_none_correct *= (n - c - i) / (n - i)

    return float(1.0 - prob_none_correct)


# ─── Expected Calibration Error ───────────────────────────────────────────────

def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    ECE — how far model confidence deviates from empirical accuracy.

    ECE = Σ_b |B_b| / n × |acc(B_b) - conf(B_b)|

    A perfectly calibrated model has ECE = 0:
      when it says 70% confidence, it is correct 70% of the time.

    LLMs are often overconfident (ECE > 0 with conf > acc).

    Parameters
    ----------
    probs  : predicted probabilities for the positive class, shape (N,)
    labels : ground-truth binary labels, shape (N,)
    n_bins : number of equal-width confidence bins
    """
    probs  = np.asarray(probs,  dtype=float)
    labels = np.asarray(labels, dtype=float)
    n      = len(probs)
    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    ece    = 0.0

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        conf = probs[mask].mean()
        acc  = labels[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)

    return float(ece)
