"""
common.py — Shared infrastructure for Week 11 Evaluation Mastery.
=================================================================
Provides:
  QAPair          — dataclass for a question + reference + candidate answers
  SYNTHETIC_QA    — 20 hand-crafted QA pairs covering different failure modes
  make_qa_dataset()— build a full evaluation dataset
  smooth()        — moving-average for plots

Dataset design rationale
------------------------
Each QA entry has:
  question         — the prompt
  reference        — the gold-standard answer
  paraphrases      — semantically equivalent answers in different words
  wrong_plausible  — wrong answer that shares tokens with the reference
  wrong_random     — answer that is clearly off-topic

This lets us show exactly where each metric succeeds and fails:

  BLEU on paraphrase      → ~0.0   (false negative)
  chrF on paraphrase      → ~0.6   (correct)
  BLEU on wrong_plausible → ~0.15  (false positive — shared tokens)
  chrF on wrong_plausible → ~0.3   (lower, but not zero)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class QAPair:
    question:        str
    reference:       str
    paraphrases:     List[str]    = field(default_factory=list)
    wrong_plausible: str          = ""
    wrong_random:    str          = ""
    category:        str          = "factual"   # factual | reasoning | math


# ─── Synthetic QA dataset ─────────────────────────────────────────────────────

SYNTHETIC_QA: List[QAPair] = [
    QAPair(
        question        = "What is the capital of France?",
        reference       = "Paris is the capital of France.",
        paraphrases     = [
            "France's capital city is Paris.",
            "The capital of France is Paris.",
            "Paris serves as France's capital.",
        ],
        wrong_plausible = "Paris is the largest city in France.",
        wrong_random    = "Berlin is the capital of Germany.",
        category        = "factual",
    ),
    QAPair(
        question        = "Who wrote the theory of general relativity?",
        reference       = "Albert Einstein developed the theory of general relativity.",
        paraphrases     = [
            "General relativity was developed by Albert Einstein.",
            "Einstein is the physicist who formulated general relativity.",
            "The theory of general relativity is due to Albert Einstein.",
        ],
        wrong_plausible = "Isaac Newton developed the theory of gravity.",
        wrong_random    = "Shakespeare wrote the theory of general relativity.",
        category        = "factual",
    ),
    QAPair(
        question        = "What is the speed of light in vacuum?",
        reference       = "The speed of light in vacuum is approximately 299,792,458 metres per second.",
        paraphrases     = [
            "Light travels at roughly 299,792,458 m/s in a vacuum.",
            "In a vacuum, light moves at approximately 3 × 10^8 m/s.",
            "The vacuum speed of light is about 300,000 km/s.",
        ],
        wrong_plausible = "The speed of sound in air is approximately 343 metres per second.",
        wrong_random    = "Water boils at 100 degrees Celsius.",
        category        = "factual",
    ),
    QAPair(
        question        = "What is the result of 17 × 23?",
        reference       = "17 multiplied by 23 equals 391.",
        paraphrases     = [
            "The product of 17 and 23 is 391.",
            "17 times 23 is 391.",
            "23 × 17 = 391.",
        ],
        wrong_plausible = "17 plus 23 equals 40.",
        wrong_random    = "The square root of 144 is 12.",
        category        = "math",
    ),
    QAPair(
        question        = "What year did World War II end?",
        reference       = "World War II ended in 1945.",
        paraphrases     = [
            "The Second World War concluded in the year 1945.",
            "1945 was the year World War II came to an end.",
            "WWII finished in 1945.",
        ],
        wrong_plausible = "World War I ended in 1918.",
        wrong_random    = "The French Revolution began in 1789.",
        category        = "factual",
    ),
    QAPair(
        question        = "What is photosynthesis?",
        reference       = "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
        paraphrases     = [
            "Plants use photosynthesis to turn sunlight, CO2, and water into glucose and oxygen.",
            "The photosynthesis process converts light energy into chemical energy stored as glucose.",
            "Through photosynthesis, plants produce glucose from carbon dioxide and water using light.",
        ],
        wrong_plausible = "Photosynthesis is the process by which plants absorb nutrients from the soil.",
        wrong_random    = "Mitosis is the division of a cell's nucleus.",
        category        = "factual",
    ),
    QAPair(
        question        = "What is the Pythagorean theorem?",
        reference       = "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides: a² + b² = c².",
        paraphrases     = [
            "For a right-angled triangle with sides a, b, and hypotenuse c, a² + b² = c².",
            "The square of the longest side of a right triangle equals the sum of squares of the shorter sides.",
            "Pythagoras showed that c² = a² + b² for right triangles.",
        ],
        wrong_plausible = "The Pythagorean theorem relates to the angles of a triangle and states they sum to 180°.",
        wrong_random    = "Newton's second law states that force equals mass times acceleration.",
        category        = "math",
    ),
    QAPair(
        question        = "What is machine learning?",
        reference       = "Machine learning is a subset of artificial intelligence where systems learn patterns from data without being explicitly programmed.",
        paraphrases     = [
            "Machine learning enables computers to learn from data and improve performance without explicit programming.",
            "In machine learning, algorithms discover patterns in data automatically rather than following hand-coded rules.",
            "ML is the study of algorithms that improve through experience and data.",
        ],
        wrong_plausible = "Machine learning is a programming paradigm where rules are explicitly written by engineers.",
        wrong_random    = "Quantum computing uses qubits to perform calculations.",
        category        = "reasoning",
    ),
    QAPair(
        question        = "What causes rainbows?",
        reference       = "Rainbows are caused by the refraction and reflection of sunlight through water droplets in the atmosphere, which disperses light into its constituent colours.",
        paraphrases     = [
            "When sunlight refracts through raindrops, it separates into colours, creating a rainbow.",
            "Rainbows form when light is refracted, reflected, and dispersed by water droplets.",
            "The spectrum of colours in a rainbow results from sunlight being bent by raindrops.",
        ],
        wrong_plausible = "Rainbows are caused by sunlight reflecting off cloud surfaces during rain.",
        wrong_random    = "Earthquakes are caused by tectonic plate movements.",
        category        = "factual",
    ),
    QAPair(
        question        = "What is gradient descent?",
        reference       = "Gradient descent is an optimisation algorithm that iteratively moves model parameters in the direction of steepest descent of the loss function to find a minimum.",
        paraphrases     = [
            "Gradient descent updates parameters by following the negative gradient of the loss to reduce error.",
            "The gradient descent algorithm minimises a loss function by stepping in the direction opposite to the gradient.",
            "By repeatedly moving in the direction that decreases the loss most, gradient descent finds model parameters that minimise error.",
        ],
        wrong_plausible = "Gradient descent is a numerical method for computing derivatives of functions.",
        wrong_random    = "Reinforcement learning uses rewards to guide agent behaviour.",
        category        = "reasoning",
    ),
    QAPair(
        question        = "What does DNA stand for?",
        reference       = "DNA stands for deoxyribonucleic acid.",
        paraphrases     = [
            "The full name for DNA is deoxyribonucleic acid.",
            "DNA is an abbreviation for deoxyribonucleic acid.",
            "Deoxyribonucleic acid is what the acronym DNA represents.",
        ],
        wrong_plausible = "RNA stands for ribonucleic acid.",
        wrong_random    = "CPU stands for central processing unit.",
        category        = "factual",
    ),
    QAPair(
        question        = "How does HTTPS differ from HTTP?",
        reference       = "HTTPS encrypts data in transit using TLS, providing confidentiality and authentication, whereas HTTP transmits data in plaintext.",
        paraphrases     = [
            "Unlike HTTP, HTTPS uses TLS encryption to secure data between client and server.",
            "HTTPS adds a layer of encryption via TLS/SSL over the unencrypted HTTP protocol.",
            "The difference is that HTTPS encrypts communications using TLS, making them private and authenticated.",
        ],
        wrong_plausible = "HTTPS uses a faster transmission protocol than HTTP to improve page load speed.",
        wrong_random    = "TCP/IP is the foundational protocol of the internet.",
        category        = "reasoning",
    ),
    QAPair(
        question        = "What is the boiling point of water at sea level?",
        reference       = "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard sea-level pressure.",
        paraphrases     = [
            "At sea level, the boiling point of water is 100°C or 212°F.",
            "Water reaches its boiling point at 100 degrees Celsius under standard atmospheric pressure.",
            "100°C is the temperature at which water boils at sea-level pressure.",
        ],
        wrong_plausible = "Water freezes at 0 degrees Celsius at standard pressure.",
        wrong_random    = "Gold melts at approximately 1064 degrees Celsius.",
        category        = "factual",
    ),
    QAPair(
        question        = "What is the role of the attention mechanism in transformers?",
        reference       = "The attention mechanism allows the model to weight the relevance of different tokens when computing each token's representation, enabling long-range dependencies.",
        paraphrases     = [
            "Attention lets each token consider all other tokens and weight them by relevance when building its representation.",
            "In transformers, attention computes a weighted combination of all token representations based on learned relevance scores.",
            "The attention mechanism gives the model the ability to selectively focus on different parts of the input when processing each token.",
        ],
        wrong_plausible = "The attention mechanism speeds up computation by parallelising token processing.",
        wrong_random    = "Backpropagation computes gradients using the chain rule.",
        category        = "reasoning",
    ),
    QAPair(
        question        = "What is overfitting in machine learning?",
        reference       = "Overfitting occurs when a model learns the training data too well, including noise and specific patterns that do not generalise to unseen data.",
        paraphrases     = [
            "A model overfits when it memorises the training set but fails to generalise to new examples.",
            "Overfitting is when a model is too complex and captures noise in the training data rather than the true underlying pattern.",
            "When a model performs well on training data but poorly on test data, it is said to be overfitting.",
        ],
        wrong_plausible = "Overfitting occurs when a model is too simple to capture the patterns in the training data.",
        wrong_random    = "Transfer learning reuses a pretrained model on a new task.",
        category        = "reasoning",
    ),
]


# ─── Dataset factory ──────────────────────────────────────────────────────────

def make_qa_dataset(
    n: Optional[int] = None,
    include_categories: Optional[List[str]] = None,
) -> List[QAPair]:
    """
    Return the synthetic QA dataset, optionally filtered.

    Parameters
    ----------
    n                  : limit to first n items (None = all 15)
    include_categories : list of category strings to keep (None = all)
    """
    data = SYNTHETIC_QA
    if include_categories:
        data = [q for q in data if q.category in include_categories]
    if n is not None:
        data = data[:n]
    return data


# ─── Utilities ────────────────────────────────────────────────────────────────

def smooth(x, w: int = 5):
    """Moving average with window w."""
    arr = np.asarray(x, dtype=float)
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid")
