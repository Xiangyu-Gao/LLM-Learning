"""
day3_task_metrics.py — Task-specific evaluation metrics.
=========================================================

CORE INSIGHT
------------
"Evaluation must reflect the deployment objective."

Next-token prediction loss (cross-entropy) tells you whether the model
assigns high probability to the training distribution.  It does NOT tell
you whether the model:
  • Generates syntactically valid code
  • Produces JSON that passes a schema checker
  • Successfully calls the right function with the right arguments
  • Answers a maths problem correctly

Task-specific metrics are the right answer.  This day implements three:

  1. pass@k  — For code generation.  You sample k solutions and check
               if at least one passes the test suite.  The mathematical
               formula from HumanEval (Chen et al. 2021) makes this
               unbiased even when n ≫ k.

  2. Schema compliance — For structured output and tool use.  Binary:
               does the model output parse as valid JSON and conform
               to the required schema?  A model that scores 0.9 on
               BLEU but outputs malformed JSON is completely useless.

  3. Function-call success — For agent evaluation.  Does the model
               call the right function with the right argument types?
               Decomposed into:
                 • Correct function name (routing)
                 • Correct argument names (keys)
                 • Correct argument types (schema)
                 • Correct argument values (semantics)

EXPERIMENT
----------
Sweep n_samples × c_correct for pass@k.
Sweep schema complexity for compliance rate.
Simulate tool-call correctness across 3 failure modes.

OUTPUT
------
results/day3/day3_pass_at_k.png          — pass@k curves
results/day3/day3_schema_compliance.png  — error type breakdown
results/day3/day3_tool_call_eval.png     — tool-call success decomposition

INTERVIEW TAKEAWAYS
-------------------
Q: "What's pass@k mathematically?"
A: pass@k = 1 − C(n−c, k)/C(n, k)
   We sample n completions, c pass tests.  pass@k = probability that
   at least one of k randomly chosen completions passes.
   The formula is unbiased: you can sample n=100, evaluate all,
   and compute pass@1 accurately without running tests 100 times.

Q: "How would you evaluate an agent?"
A: Decompose the task into observable sub-steps:
   (1) Does it call the right function?  → routing accuracy
   (2) Are the arguments well-formed?    → schema compliance
   (3) Does the downstream system accept them?  → success rate
   (4) Does the overall task succeed?    → task completion
   Each level gives a different signal for debugging failures.

Q: "Why is next-token loss insufficient?"
A: Loss measures the average log-probability of ground-truth tokens.
   It optimises for distributional match, not task success.
   A model that outputs almost-correct code (one token off) incurs
   similar loss to a model that outputs gibberish, but the former
   may pass tests and the latter won't.
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

from eval.metrics import pass_at_k             # noqa: E402
from eval.schema_check import (                # noqa: E402
    validate_schema, batch_schema_eval,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "day3")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Sample schemas ────────────────────────────────────────────────────────────

SIMPLE_SCHEMA = {
    "type": "object",
    "required": ["name", "value"],
    "properties": {
        "name":  {"type": "string"},
        "value": {"type": "number"},
    },
}

MEDIUM_SCHEMA = {
    "type": "object",
    "required": ["function", "arguments"],
    "properties": {
        "function":  {"type": "string", "enum": ["add", "sub", "mul", "div"]},
        "arguments": {
            "type": "object",
            "required": ["a", "b"],
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
        },
    },
}

STRICT_SCHEMA = {
    "type": "object",
    "required": ["tool_name", "params", "confidence"],
    "properties": {
        "tool_name":  {"type": "string"},
        "params":     {
            "type": "object",
            "required": ["query", "max_results"],
            "properties": {
                "query":       {"type": "string", "minLength": 1},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
            },
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


# ─── Simulated model outputs (for schema compliance) ─────────────────────────

def generate_schema_outputs(schema_name: str, n: int, error_rate: float,
                             rng: np.random.Generator) -> list:
    """
    Simulate n model outputs for a given schema.

    error_rate controls fraction that have defects:
      - 40% of errors: JSON parse error (malformed JSON)
      - 30% of errors: missing required field
      - 30% of errors: wrong type for a field
    """
    outputs = []
    import json

    for _ in range(n):
        if rng.random() > error_rate:
            # Valid output
            if schema_name == "simple":
                obj = {"name": "temperature", "value": 23.5}
            elif schema_name == "medium":
                op = rng.choice(["add", "sub", "mul", "div"])
                obj = {"function": str(op), "arguments": {"a": 3.0, "b": 4.0}}
            else:
                obj = {
                    "tool_name": "search",
                    "params": {"query": "hello", "max_results": 5},
                    "confidence": 0.9,
                }
            outputs.append(json.dumps(obj))
        else:
            # Inject a specific error
            error_type = rng.choice(["parse", "missing", "wrong_type"],
                                    p=[0.4, 0.3, 0.3])
            if error_type == "parse":
                outputs.append('{"name": "foo", value: 1}')   # unquoted key
            elif error_type == "missing":
                if schema_name == "simple":
                    outputs.append('{"name": "foo"}')           # missing value
                elif schema_name == "medium":
                    outputs.append('{"function": "add"}')       # missing arguments
                else:
                    outputs.append('{"tool_name": "search"}')   # missing params+confidence
            else:  # wrong_type
                if schema_name == "simple":
                    outputs.append('{"name": 123, "value": "oops"}')
                elif schema_name == "medium":
                    outputs.append('{"function": "pow", "arguments": {"a": 1, "b": 2}}')
                else:
                    outputs.append('{"tool_name": "search", '
                                   '"params": {"query": "", "max_results": 100}, '
                                   '"confidence": 1.5}')  # confidence > 1.0
    return outputs


# ─── Experiment A: pass@k curves ─────────────────────────────────────────────

def run_pass_at_k(smoke: bool):
    print("\n[A] pass@k — Probability at Least One Solution Passes")
    print("    Formula: pass@k = 1 − C(n−c, k) / C(n, k)")

    n_total    = 20        # samples per problem
    ks         = [1, 2, 5, 10] if not smoke else [1, 5]
    c_values   = range(0, n_total + 1)

    results = {k: [pass_at_k(n_total, c, k) for c in c_values]
               for k in ks}

    # Print a few key values
    print(f"\n    n={n_total} total samples:")
    print(f"    {'c/n':>6} | " + " | ".join(f"pass@{k:>2}" for k in ks))
    print("    " + "-" * 50)
    for c in [0, 1, 3, 5, 10, 15, 20]:
        row = f"  {c:>2}/{n_total} = {c/n_total:.0%} | "
        row += " | ".join(f"{results[k][c]:>7.3f}" for k in ks)
        print("   " + row)

    return results, n_total, c_values


# ─── Experiment B: Schema compliance ─────────────────────────────────────────

def run_schema_compliance(smoke: bool):
    print("\n[B] Schema Compliance — Structured Output Evaluation")
    n_outputs   = 20 if smoke else 100
    rng         = np.random.default_rng(0)
    error_rates = [0.0, 0.1, 0.2, 0.4, 0.6]
    schemas     = [
        ("simple",  SIMPLE_SCHEMA,  "Simple (2 fields)"),
        ("medium",  MEDIUM_SCHEMA,  "Medium (nested enum)"),
        ("strict",  STRICT_SCHEMA,  "Strict (range constraints)"),
    ]

    compliance_by_schema = {}
    for skey, schema, slabel in schemas:
        row = []
        for er in error_rates:
            outputs = generate_schema_outputs(skey, n_outputs, er, rng)
            result  = batch_schema_eval(outputs, schema)
            row.append(result["compliance_rate"])
        compliance_by_schema[slabel] = row
        print(f"  {slabel}: {[f'{v:.2f}' for v in row]}")

    return compliance_by_schema, error_rates


# ─── Experiment C: Tool-call success decomposition ───────────────────────────

def run_tool_call_eval(smoke: bool):
    """
    Simulate tool-call evaluation at 4 levels of correctness.
    Shows that aggregate 'success rate' hides where the failure is.
    """
    print("\n[C] Tool-Call Success Decomposition")

    n = 20 if smoke else 100
    rng = np.random.default_rng(1)

    # Simulate a model with these probabilities
    p_correct_name  = 0.88   # high — routing is often easy
    p_correct_keys  = 0.75   # lower — models miss args
    p_correct_types = 0.82
    p_correct_vals  = 0.70   # hardest — value correctness

    results = {
        "Function routing\n(correct name)": [],
        "Argument structure\n(correct keys)": [],
        "Type compliance\n(correct types)": [],
        "Value correctness\n(correct values)": [],
        "Full success\n(all four)": [],
    }

    for _ in range(n):
        cn = rng.random() < p_correct_name
        ck = rng.random() < p_correct_keys
        ct = rng.random() < p_correct_types
        cv = rng.random() < p_correct_vals
        results["Function routing\n(correct name)"].append(float(cn))
        results["Argument structure\n(correct keys)"].append(float(ck))
        results["Type compliance\n(correct types)"].append(float(ct))
        results["Value correctness\n(correct values)"].append(float(cv))
        results["Full success\n(all four)"].append(float(cn and ck and ct and cv))

    rates = {k: np.mean(v) for k, v in results.items()}
    for k, v in rates.items():
        print(f"  {k.replace(chr(10), ' '):<50} {v:.2%}")

    return rates


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_pass_at_k(results, n_total, c_values):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Day 3 — pass@k Curves\n"
                 "(n=20 total samples; c = number of correct samples)",
                 fontsize=11)

    colors = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71"]
    for (k, vals), color in zip(results.items(), colors):
        ax.plot(list(c_values), vals, "o-", color=color, label=f"pass@{k}",
                markersize=4, linewidth=2)

    ax.set_xlabel("c — number of correct samples out of n=20")
    ax.set_ylabel("Probability (at least 1 of k passes)")
    ax.legend()
    ax.set_xlim(0, n_total)
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
    ax.text(n_total * 0.65, 0.52, "50% threshold", color="gray", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "day3_pass_at_k.png"), dpi=150)
    plt.close(fig)
    print("\nSaved: day3_pass_at_k.png")


def plot_schema_compliance(compliance, error_rates):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Day 3 — Schema Compliance vs Model Error Rate", fontsize=11)

    colors = ["#3498db", "#e67e22", "#e74c3c"]
    for (label, vals), color in zip(compliance.items(), colors):
        ax.plot(error_rates, vals, "o-", color=color, label=label, linewidth=2)

    ax.set_xlabel("Injected error rate (fraction of outputs with defects)")
    ax.set_ylabel("Schema compliance rate")
    ax.legend()
    ax.set_xlim(-0.02, 0.65)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.4)
    ax.text(0.42, 0.91, "90% SLA threshold", color="gray", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "day3_schema_compliance.png"), dpi=150)
    plt.close(fig)
    print("Saved: day3_schema_compliance.png")


def plot_tool_call(rates):
    labels = list(rates.keys())
    values = list(rates.values())

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Day 3 — Tool-Call Success Decomposition\n"
                 "Aggregate success hides which component is failing",
                 fontsize=11)

    colors = ["#3498db", "#9b59b6", "#1abc9c", "#e67e22", "#e74c3c"]
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.55)

    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=9)

    ax.set_xlabel("Success rate")
    ax.set_xlim(0, 1.15)
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.4)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "day3_tool_call_eval.png"), dpi=150)
    plt.close(fig)
    print("Saved: day3_tool_call_eval.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(smoke: bool = False):
    print("\n" + "=" * 70)
    print(" Day 3 — Task-Specific Metrics")
    print("=" * 70)

    pak_results, n_total, c_vals = run_pass_at_k(smoke)
    compliance, error_rates      = run_schema_compliance(smoke)
    tool_rates                   = run_tool_call_eval(smoke)

    plot_pass_at_k(pak_results, n_total, c_vals)
    plot_schema_compliance(compliance, error_rates)
    plot_tool_call(tool_rates)

    print("\n" + "─" * 60)
    print("KEY INSIGHT: Task-specific > generic metrics")
    print("  • pass@k directly measures code correctness probability")
    print("  • Schema compliance is binary — 99% BLEU with bad JSON = 0 utility")
    print("  • Tool decomposition tells you WHERE the failure is (routing vs values)")
    print("─" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(smoke=args.smoke)
