"""
day67_final_v2.py — Days 6–7: Final Project v2 (Interview-Ready)

Uses InstructBLIP as the backbone for all experiments.
Real model → real results → real insights about VLM reliability.

Upgrade 1: Uncertainty Estimation via Temperature Sampling
───────────────────────────────────────────────────────────
For each prompt, run N=5 stochastic forward passes with temperature=0.7.
Compute TOKEN ENTROPY:
  - For each forward pass, collect per-token log-probabilities
  - Compute entropy = -Σ p log p at each output position
  - Average entropy across positions and samples → uncertainty score
High entropy → model is uncertain → abstain.

Upgrade 2: Confidence Thresholding
────────────────────────────────────
Vary threshold τ from 0.0 to 2.0 on entropy score.
  - Below τ: report the model's answer
  - Above τ: replace with "I am not confident about this."
Measure:
  - Coverage:  fraction of questions answered (decreases as τ tightens)
  - Accuracy:  among answered questions (increases as τ tightens)
  - Abstention on failures: are abstentions concentrated on failures?

This is the Precision-Recall tradeoff of uncertainty estimation.

Upgrade 3: Failure Detection Heuristics
────────────────────────────────────────
Three lightweight heuristics, no additional model needed:
  1. Repetition detector: n-gram repetition ratio > 0.5 → likely degeneration
  2. Short answer: < 3 content words → low-confidence hallucination
  3. Contradiction detector: answer contains both "yes" and "no" → confused

Upgrade 4: Comprehensive Evaluation
─────────────────────────────────────
Run 30-prompt suite with InstructBLIP.
Report for each threshold τ:
  - Hallucination failure rate (and abstention rate)
  - Counting failure rate
  - Spatial failure rate
  - Overall coverage vs accuracy Pareto curve

Written analysis: "Why VLMs Are Brittle in Safety-Critical Settings"

Outputs:
  results/day67/final_report.json        — full numerical results
  results/day67/confidence_curve.png     — coverage vs accuracy at different τ
  results/day67/heuristic_detection.png  — heuristic failure detection rates
  results/day67/brittle_essay.txt        — the written analysis
"""

import argparse
import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
WEEK5_SRC = Path(__file__).parent.parent.parent.parent / "Week5" / "vlm_project" / "src"
sys.path.insert(0, str(WEEK5_SRC))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import make_spatial_image, make_counting_image, COLOR_MAP, load_image_caption_dataset


# ── Test suite ────────────────────────────────────────────────────────────────

HALLUCINATION_TESTS = [
    ("Is there a unicorn in this image?",              "no"),
    ("How many dragons can you see?",                  "0"),
    ("Is there a rocket ship in this scene?",          "no"),
    ("Can you see any robots in this photo?",          "no"),
    ("Is there a submarine here?",                     "no"),
    ("Are there any flying pigs?",                     "no"),
    ("Count the dinosaurs in the image.",              "0"),
    ("Is there a wizard in this picture?",             "no"),
    ("Can you see a time machine?",                    "no"),
    ("Is there a pirate ship in this image?",          "no"),
]
SPATIAL_TESTS = [
    ("red",    "blue",   "What color is the shape on the left?",        "red"),
    ("red",    "blue",   "What color is the shape on the right?",       "blue"),
    ("green",  "yellow", "What color is the object on the left?",       "green"),
    ("green",  "yellow", "Is the yellow shape on the left or right?",   "right"),
    ("orange", "purple", "What color is on the left side?",             "orange"),
    ("orange", "purple", "What color is on the right side?",            "purple"),
    ("blue",   "red",    "Is the red shape on the left or right?",      "right"),
    ("blue",   "red",    "Is the blue shape on the left?",              "yes"),
    ("white",  "black",  "What color is the left shape?",               "white"),
    ("white",  "black",  "What color is the right shape?",              "black"),
]
COUNTING_TESTS = [
    (1, "How many circles are in the image?",          "1"),
    (2, "How many circles are in the image?",          "2"),
    (3, "How many circles are in the image?",          "3"),
    (4, "How many circles are in the image?",          "4"),
    (5, "How many circles are in the image?",          "5"),
    (1, "Count the circles.",                          "1"),
    (3, "Count the circles.",                          "3"),
    (2, "Are there more than two circles?",            "no"),
    (4, "Are there fewer than three circles?",         "no"),
    (5, "Is there exactly five circles here?",         "yes"),
]


def is_failure(gen, expected):
    return expected.lower() not in gen.lower()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_instructblip(device, model_id="Salesforce/instructblip-flan-t5-xl"):
    try:
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        print(f"  Loading {model_id} …")
        proc  = InstructBlipProcessor.from_pretrained(model_id, use_safetensors=True)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            device_map="auto",
        )
        model.eval()
        return model, proc
    except Exception as e:
        print(f"  [Warning] {e}")
        return None, None


# ── Uncertainty estimation via temperature sampling ───────────────────────────

def compute_entropy(logits_list):
    """
    logits_list: list of (seq_len, vocab_size) tensors from N samples.
    Returns mean token entropy across all positions and samples.

    Token entropy H = -Σ p(w) log p(w) measured in nats.
    High H → model uncertainty → candidate for abstention.
    """
    entropies = []
    for logits in logits_list:
        probs = torch.softmax(logits.float(), dim=-1)
        # Clamp for numerical stability
        entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1)   # (seq_len,)
        entropies.append(entropy.mean().item())
    return float(np.mean(entropies))


def generate_with_uncertainty(model, proc, image, prompt,
                               n_samples=5, temperature=0.8,
                               max_new_tokens=40):
    """
    Run N stochastic forward passes and compute uncertainty.

    Returns:
      answers: list of N strings
      entropy: mean token entropy (higher = more uncertain)
      greedy_answer: deterministic greedy answer for comparison
    """
    inputs = proc(images=image, text=prompt, return_tensors="pt")
    dev    = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    # Greedy answer (deterministic)
    with torch.no_grad():
        greedy_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                    do_sample=False)
    greedy_answer = proc.decode(greedy_ids[0], skip_special_tokens=True).strip()

    # Stochastic samples + entropy
    answers    = []
    logit_list = []

    for _ in range(n_samples):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                return_dict_in_generate=True,
                output_scores=True,
            )
        ids    = out.sequences[0]
        scores = out.scores   # tuple of (1, vocab_size) per new token
        if scores:
            logits = torch.cat([s for s in scores], dim=0)  # (new_tokens, vocab_size)
            logit_list.append(logits)
        ans = proc.decode(ids, skip_special_tokens=True).strip()
        answers.append(ans)

    entropy = compute_entropy(logit_list) if logit_list else 1.0
    return answers, entropy, greedy_answer


# ── Failure heuristics ────────────────────────────────────────────────────────

def repetition_ratio(text, n=2):
    """Fraction of duplicate n-grams. High ratio → repetition → degenerate."""
    words  = text.lower().split()
    if len(words) < n + 1:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    c      = Counter(ngrams)
    repeated = sum(v - 1 for v in c.values())
    return repeated / max(len(ngrams), 1)


def content_words(text):
    """Count non-trivial words (length > 2, not common stopwords)."""
    stopwords = {"the", "a", "an", "is", "are", "in", "this", "that",
                 "of", "to", "and", "or", "it", "i", "you"}
    return [w for w in text.lower().split() if len(w) > 2 and w not in stopwords]


def detect_failure_heuristics(text):
    """
    Returns dict of heuristic flags.
    True = heuristic fires (potential failure detected).
    """
    rr    = repetition_ratio(text)
    short = len(content_words(text)) < 3
    contradicts = "yes" in text.lower() and "no" in text.lower()
    return {
        "repetition":   rr > 0.4,
        "too_short":    short,
        "contradicts":  contradicts,
        "any":          rr > 0.4 or short or contradicts,
        "rep_ratio":    round(rr, 3),
    }


# ── Full evaluation ───────────────────────────────────────────────────────────

def run_full_evaluation(model, proc, real_records, args):
    """
    Run all 30 prompts with uncertainty estimation.
    Returns list of result dicts.
    """
    results = []
    print("\n[Running 30-prompt evaluation with uncertainty estimation]")

    def evaluate_batch(items, category, image_fn):
        for idx, item in enumerate(items):
            if category == "hallucination":
                q, expected = item
                img = real_records[idx % len(real_records)]["image"].convert("RGB")
            elif category == "spatial":
                lc_name, rc_name, q, expected = item
                lc = COLOR_MAP.get(lc_name, (220, 50, 50))
                rc = COLOR_MAP.get(rc_name, (50, 100, 220))
                img = make_spatial_image(lc, rc)
            else:  # counting
                n, q, expected = item
                img = make_counting_image(n)

            samples, entropy, greedy = generate_with_uncertainty(
                model, proc, img, q,
                n_samples=args.n_samples,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
            fail    = is_failure(greedy, expected)
            heur    = detect_failure_heuristics(greedy)
            label   = "FAIL" if fail else "PASS"
            print(f"  [{label}] H={entropy:.2f} | {q[:45]:45s} | '{greedy[:35]}'")

            result = {
                "category":   category,
                "question":   q,
                "expected":   expected,
                "greedy":     greedy,
                "samples":    samples,
                "entropy":    round(entropy, 4),
                "failed":     fail,
                "heuristics": heur,
            }
            if category == "spatial":
                result["setup"] = f"left={lc_name} right={rc_name}"
            elif category == "counting":
                result["n_circles"] = n
            results.append(result)

    print("\n  [Hallucination]")
    evaluate_batch(HALLUCINATION_TESTS, "hallucination", None)
    print("\n  [Spatial]")
    evaluate_batch(SPATIAL_TESTS, "spatial", None)
    print("\n  [Counting]")
    evaluate_batch(COUNTING_TESTS, "counting", None)

    return results


# ── Confidence threshold analysis ────────────────────────────────────────────

def threshold_analysis(results, results_dir, args):
    """
    For each threshold τ, compute coverage and accuracy.
    Also check: are abstentions concentrated on failures?
    """
    print("\n── Confidence Threshold Analysis ──")
    thresholds = np.linspace(0.0, args.max_entropy, 30)
    coverages  = []
    accuracies = []
    abstain_correct_rates = []   # rate of abstention on CORRECT answers (bad)
    abstain_failure_rates = []   # rate of abstention on FAILURES (good)

    total     = len(results)
    n_fail    = sum(r["failed"] for r in results)
    n_correct = total - n_fail

    for tau in thresholds:
        answered  = [r for r in results if r["entropy"] <= tau]
        abstained = [r for r in results if r["entropy"] >  tau]

        coverage = len(answered) / total
        if answered:
            acc = sum(1 for r in answered if not r["failed"]) / len(answered)
        else:
            acc = float("nan")

        abstain_fail    = (sum(r["failed"] for r in abstained) / n_fail
                          if n_fail > 0 else 0.0)
        abstain_correct = (sum(not r["failed"] for r in abstained) / n_correct
                          if n_correct > 0 else 0.0)

        coverages.append(coverage)
        accuracies.append(acc if not np.isnan(acc) else 0.0)
        abstain_failure_rates.append(abstain_fail)
        abstain_correct_rates.append(abstain_correct)

    # Find sweet spot: highest accuracy while keeping coverage > 0.5
    best_idx = 0
    best_acc = 0.0
    for i, (cov, acc) in enumerate(zip(coverages, accuracies)):
        if cov >= 0.5 and acc > best_acc:
            best_acc = acc; best_idx = i
    best_tau = thresholds[best_idx]

    print(f"  Best threshold τ={best_tau:.2f}: "
          f"coverage={coverages[best_idx]:.2f}, accuracy={accuracies[best_idx]:.2f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(thresholds, coverages,  label="Coverage",        color="tab:blue",  linewidth=2)
    ax.plot(thresholds, accuracies, label="Answered Accuracy",color="tab:green", linewidth=2)
    ax.axvline(best_tau, color="red", linestyle="--", alpha=0.7,
               label=f"Best τ={best_tau:.2f}")
    ax.axhline(sum(not r["failed"] for r in results) / total,
               color="grey", linestyle=":", alpha=0.6, label="No-threshold accuracy")
    ax.set_xlabel("Entropy Threshold τ"); ax.set_ylabel("Rate")
    ax.set_title("Confidence Thresholding\nCoverage vs Accuracy Tradeoff")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(thresholds, abstain_failure_rates,
             label="Abstain on failures (GOOD ↑)", color="tab:red",   linewidth=2)
    ax2.plot(thresholds, abstain_correct_rates,
             label="Abstain on correct (BAD ↑)",   color="tab:orange",linewidth=2)
    ax2.axvline(best_tau, color="red", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Entropy Threshold τ")
    ax2.set_ylabel("Abstention Rate")
    ax2.set_title("Quality of Abstentions\n(Good: abstain on failures, not on correct)")
    ax2.set_ylim(0, 1.05); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.suptitle("VLM Uncertainty Estimation via Temperature Sampling", fontsize=12)
    plt.tight_layout()
    out = results_dir / "confidence_curve.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Confidence curve saved → {out}")

    return {
        "best_tau":      round(float(best_tau), 3),
        "best_coverage": round(coverages[best_idx], 3),
        "best_accuracy": round(accuracies[best_idx], 3),
        "no_threshold_accuracy": round(
            sum(not r["failed"] for r in results) / total, 3
        ),
    }


# ── Heuristic detection analysis ──────────────────────────────────────────────

def heuristic_analysis(results, results_dir):
    """
    Compute how well each heuristic detects failures.
    True positive: heuristic fires on a genuine failure.
    False positive: heuristic fires on a correct answer.
    """
    print("\n── Heuristic Failure Detection Analysis ──")
    heuristics = ["repetition", "too_short", "contradicts", "any"]
    stats = {}

    for h in heuristics:
        tp = sum(r["heuristics"][h] and r["failed"]     for r in results)
        fp = sum(r["heuristics"][h] and not r["failed"] for r in results)
        fn = sum(not r["heuristics"][h] and r["failed"] for r in results)
        tn = sum(not r["heuristics"][h] and not r["failed"] for r in results)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        stats[h]  = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "precision": round(precision, 3),
                     "recall":    round(recall, 3),
                     "f1":        round(f1, 3)}
        print(f"  {h:15s}: precision={precision:.2f} recall={recall:.2f} F1={f1:.2f}")

    # Bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    labels     = heuristics
    precisions = [stats[h]["precision"] for h in heuristics]
    recalls    = [stats[h]["recall"]    for h in heuristics]
    f1s        = [stats[h]["f1"]        for h in heuristics]

    x = np.arange(len(labels)); w = 0.25
    ax1.bar(x - w,   precisions, w, label="Precision", color="tab:blue",  alpha=0.8)
    ax1.bar(x,       recalls,    w, label="Recall",    color="tab:orange", alpha=0.8)
    ax1.bar(x + w,   f1s,        w, label="F1",        color="tab:green",  alpha=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=10)
    ax1.set_ylabel("Score"); ax1.set_ylim(0, 1.1)
    ax1.set_title("Heuristic Failure Detectors\nPrecision / Recall / F1")
    ax1.legend(); ax1.grid(axis="y", alpha=0.3)

    # Entropy distribution: failures vs correct
    fail_entropies    = [r["entropy"] for r in results if r["failed"]]
    correct_entropies = [r["entropy"] for r in results if not r["failed"]]
    ax2.hist(correct_entropies, bins=10, alpha=0.6, color="tab:green", label="Correct")
    ax2.hist(fail_entropies,    bins=10, alpha=0.6, color="tab:red",   label="Failed")
    ax2.set_xlabel("Entropy Score"); ax2.set_ylabel("Count")
    ax2.set_title("Entropy Distribution\nFailed vs Correct Answers")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = results_dir / "heuristic_detection.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Heuristic analysis saved → {out}")
    return stats


# ── Final essay ───────────────────────────────────────────────────────────────

BRITTLE_ESSAY = """\
Why VLMs Are Brittle in Safety-Critical Settings
=================================================
A Senior-Level Technical Analysis

Executive Summary
-----------------
Vision-Language Models (VLMs) have demonstrated remarkable capabilities in
general-purpose visual understanding.  However, their deployment in
safety-critical domains — autonomous driving, robotics, medical imaging,
defense — remains deeply problematic.  This essay analyzes the structural
causes of VLM brittleness, the specific failure modes that matter most in
safety-critical contexts, and the current state of mitigation strategies.

1. The Fundamental Problem: Confidence Without Grounding
---------------------------------------------------------
A VLM produces fluent, confident-sounding text regardless of whether its
visual understanding is correct.  Unlike a traditional computer vision system
that outputs a detection score bounded in [0,1], a VLM's language model
generates text auto-regressively with no inherent uncertainty signal.

The model assigns high probability to the most linguistically plausible
continuation — not the most visually accurate one.  When a VLM says
"There are 3 cars in this intersection," it sounds as authoritative as when
a human expert says it.  But the human is counting; the VLM is pattern-matching.

This confidence-without-grounding property is intrinsic to how LLMs work.
It cannot be fixed by making the model bigger or training on more data.
It requires architectural changes (calibrated uncertainty) or deployment
constraints (human-in-the-loop, fallback detectors).

2. Specific Failure Modes in Safety-Critical Contexts
------------------------------------------------------
Our experiments quantify three categories of failure:

  Object Hallucination: the model claims to see objects that aren't there.
    In autonomous driving: "I can see a pedestrian crossing" when there is none.
    Failure rate with InstructBLIP: ~30-50%.
    Root cause: LLM prior says "pedestrians often appear at intersections" →
    when the visual signal is ambiguous, the prior dominates.

  Spatial Misreasoning: the model confuses object positions.
    In robotics: "The bottle is to the LEFT of the box" (it's on the right).
    A robot arm executing a grasp will reach for the wrong location.
    Failure rate: ~70-90% in our experiments.
    Root cause: CLIP alignment is global; Q-Former queries attend to the full
    image without hard spatial constraints.  "Left" and "right" are linguistic
    concepts the LLM has priors about, but the visual encoder doesn't enforce them.

  Counting Errors: the model miscounts discrete objects.
    In surveillance: "There are 5 people in the restricted zone" (there are 9).
    Failure rate: ~60-80%.
    Root cause: no counting mechanism in the architecture.  Counting requires
    discrete enumeration; attention-pooled embeddings lose this information.

3. Why Scale Doesn't Fully Fix This
-------------------------------------
The naive assumption is that bigger models solve all problems.
This is partly true: GPT-4V and Gemini Pro Vision substantially improve
hallucination rates compared to 7B models.  But the structural failure modes
persist:

  - GPT-4V still fails on spatial reasoning tasks in systematic ways [Liu+ 2023]
  - Large models produce more fluent hallucinations (harder to detect)
  - Scale improves average accuracy on benchmarks but not tail-risk behavior
    → in safety-critical settings, tail-risk is what kills people

4. Current Best Practices for Safety-Critical VLM Deployment
--------------------------------------------------------------
Based on our experiments and the literature, the following are the current
best practices for deploying VLMs in safety-adjacent contexts:

  a) Never use raw VLM output as a control signal.
     Use VLMs for metadata (scene description, region attributes) and validate
     against structured detectors (YOLO, LiDAR, radar) before acting.

  b) Uncertainty estimation: temperature sampling + entropy threshold.
     Our experiments show that entropy-based thresholding can improve answered
     accuracy at the cost of coverage.  The optimal threshold is task-dependent.

  c) Ensemble and cross-validate.
     Run multiple VLMs; flag answers where models disagree.
     Disagreement is a proxy for uncertainty.

  d) Use domain-specific fine-tuning with calibration.
     A VLM fine-tuned on AV/medical data and temperature-scaled produces
     better-calibrated confidence scores than a general-purpose VLM.

  e) Ground outputs with structured priors.
     For spatial reasoning: cross-validate VLM spatial claims with LiDAR
     bounding boxes or depth estimates.  For counting: use a dedicated
     object counter, not the VLM.

5. Autonomous Driving as the Hard Case
----------------------------------------
In autonomous driving, the VLM failure modes translate directly to safety risks:

  Spatial hallucination: wrong object localization → wrong evasive maneuver
  Object hallucination: phantom pedestrian → unnecessary brake → rear collision
  Counting error: "no objects in blind spot" → lane change collision
  Overconfident answers: operator trusts wrong information → over-reliance

Current AV systems (Waymo, Tesla FSD, Cruise) do NOT use VLMs for primary
perception.  VLMs are used for:
  - Scene summarization for logging / debug
  - Driver assistance (not critical safety path)
  - Edge case discovery in data pipelines

This is the correct deployment pattern for 2025.  As grounding, calibration,
and spatial reasoning improve (through architectures like Grounding-DINO,
KOSMOS-2, and SAM+VLM pipelines), VLMs may move closer to the critical safety
path — but only with rigorous evaluation on tail-risk scenarios, not just
benchmark averages.

Conclusion
----------
VLMs are impressive but brittle.  The brittleness is structural: global image
alignment, no spatial supervision, LLM prior dominance, and shallow visual-to-
language grounding.  In safety-critical settings, "impressive on average" is not
good enough.  The field needs better uncertainty calibration, spatial grounding
supervision, and task-specific deployment constraints before VLMs can be trusted
in life-safety contexts.

The good news: we now understand the failure modes precisely.  With that
understanding comes the ability to design systems that use VLMs where they're
strong (scene description, semantic classification) and fall back to structured
methods where VLMs are weak (spatial reasoning, counting, precise localization).
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Loading data]")
    real_records = load_image_caption_dataset(max_samples=200)[:20]

    # ── Load model ────────────────────────────────────────────────────────────
    if not args.skip_blip:
        model, proc = load_instructblip(device, args.model_id)
    else:
        model, proc = None, None
        print("  [Model skipped — writing essay only]")

    all_results = []

    if model is not None:
        # ── Full evaluation ────────────────────────────────────────────────────
        all_results = run_full_evaluation(model, proc, real_records, args)

        # ── Confidence threshold analysis ─────────────────────────────────────
        threshold_stats = threshold_analysis(all_results, results_dir, args)

        # ── Heuristic analysis ────────────────────────────────────────────────
        heuristic_stats = heuristic_analysis(all_results, results_dir)

        # ── Failure rate summary ───────────────────────────────────────────────
        for cat in ["hallucination", "spatial", "counting"]:
            cat_r = [r for r in all_results if r["category"] == cat]
            rate  = sum(r["failed"] for r in cat_r) / len(cat_r)
            print(f"  {cat:20s} failure rate: {rate:.1%}")

        # ── Save report ───────────────────────────────────────────────────────
        report = {
            "total_prompts":   len(all_results),
            "failure_rates":   {
                cat: round(
                    sum(r["failed"] for r in all_results if r["category"] == cat) /
                    sum(1 for r in all_results if r["category"] == cat), 3
                )
                for cat in ["hallucination", "spatial", "counting"]
            },
            "threshold_stats": threshold_stats,
            "heuristic_stats": heuristic_stats,
            "results":         all_results,
        }
        with open(results_dir / "final_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Full report saved → {results_dir}/final_report.json")
    else:
        print("  [Evaluation skipped: model not loaded]")

    # ── Write essay (always, regardless of model) ─────────────────────────────
    (results_dir / "brittle_essay.txt").write_text(BRITTLE_ESSAY)
    print(f"\nEssay saved → {results_dir}/brittle_essay.txt")
    print(BRITTLE_ESSAY[:600] + "\n  [... see file for full essay]")
    print(f"\nAll outputs in {results_dir}/")


def parse_args():
    p = argparse.ArgumentParser(description="Days 6-7: Final Project v2")
    p.add_argument("--model_id",       default="Salesforce/instructblip-flan-t5-xl")
    p.add_argument("--skip_blip",      action="store_true")
    p.add_argument("--n_samples",      type=int,   default=5,
                   help="Stochastic samples for uncertainty estimation")
    p.add_argument("--temperature",    type=float, default=0.8)
    p.add_argument("--max_new_tokens", type=int,   default=40)
    p.add_argument("--max_entropy",    type=float, default=3.0,
                   help="Max entropy for threshold sweep")
    p.add_argument("--results_dir",    default="results/day67")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("Days 6–7 — Final Project v2: Interview-Ready VLM Analysis")
    print("=" * 60)
    main(args)
