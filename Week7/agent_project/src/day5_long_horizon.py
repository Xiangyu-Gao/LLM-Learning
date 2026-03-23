"""
day5_long_horizon.py — Day 5: Long-Horizon Failure Analysis

Why agents degrade exponentially with horizon (1-page essay)
─────────────────────────────────────────────────────────────
An agent working on an N-step task must maintain a coherent internal model
of the world across all N steps. At each step, three failure modes compound:

  1. Error accumulation (multiplicative degradation)
     Each step has some error probability ε (even good models have ε ≈ 5-15%
     per step). Failures compound: after N steps, P(all steps correct) = (1-ε)^N.
     At ε=0.10: N=3 → 73%, N=5 → 59%, N=8 → 43%, N=10 → 35%.
     The relationship is exponential, not linear.

  2. Context pollution (quadratic cost, linear drift)
     Every observation adds tokens to the context window. By step N, the model
     must attend to O(N) tokens to extract the current sub-goal. Attention
     weight spreads across all prior steps — earlier, relevant context competes
     with recent, possibly irrelevant observations. This is "state drift":
     the model's implicit world-model gradually diverges from reality.

  3. Tool misuse compounds with state drift
     Early tool results inform later reasoning. A wrong early fact (e.g., a
     wrong Wikipedia result due to disambiguation) propagates. Each subsequent
     step builds on it. After 3 dependent steps, the error is structural —
     it cannot be corrected by the model because all recent context confirms it.

  4. Planning horizon collapse
     Without explicit replanning, a ReAct agent's effective planning horizon
     is 1 step (greedy). It picks the locally best action at each step but
     cannot look ahead. A task requiring globally optimal step ordering will
     be solved sub-optimally or not at all.

  5. Memory overload
     At step N, the in-context message history grows linearly. The model
     must process all prior tool results to generate the next action. Both
     cost and latency grow linearly; attention quality may degrade.

The practical implication: for tasks requiring >5 steps, you MUST have
  a) explicit replanning between steps, or
  b) a separate critic/verifier after each step, or
  c) a structured task graph (not a flat sequence).

Without these, expect success rates to halve roughly every 3-4 steps
beyond the 2-step baseline.

Experiment design
─────────────────
10 tasks designed with known minimum step counts (1-8 required tool calls).
We run each task and classify the outcome:
  SUCCESS   — final answer is plausibly correct (passes pattern check)
  WRONG     — agent finished but answer is wrong
  TOOL_FAIL — a tool call returned an error that broke the chain
  HALLUCIN  — agent finished with high confidence on a wrong answer
  TIMEOUT   — hit max_steps before finishing

For each task we also record:
  fail_step — which step number the first critical error occurred
  fail_type — reasoning / planning / tool / none

Results are plotted as: success_rate vs required_steps.
"""

import sys
import re
import time
import json
import argparse
import os
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from utils import get_client, TOOL_SCHEMAS, dispatch_tool, MODEL


# ── Task definitions ──────────────────────────────────────────────────────────
# Each task has:
#   question       — the question to answer
#   min_steps      — minimum tool calls needed for a correct answer
#   success_pattern — substring that should appear in correct answer (lowercase)
#   expected_answer — rough expected value (for display, not hard matching)

@dataclass
class Task:
    id: str
    question: str
    min_steps: int       # minimum tool calls needed
    success_pattern: str # case-insensitive substring check
    expected_answer: str


TASKS = [
    # 1 step
    Task("T01", "What is the capital of France?", 1, "paris", "Paris"),
    Task("T02", "What is 17 multiplied by 23?", 1, "391", "391"),

    # 2 steps
    Task("T03",
         "In what year was the Eiffel Tower built? Calculate 2025 minus that year.",
         2, "136", "136 years ago (built 1889)"),
    Task("T04",
         "What is the boiling point of water in Celsius? Convert to Fahrenheit: (C * 9/5) + 32.",
         2, "212", "212°F"),

    # 3 steps
    Task("T05",
         "Find the population of Canada. Find the population of Australia. "
         "Calculate how many times larger Canada is than Australia (round to 1 decimal).",
         3, ".", "~1.5x"),  # any decimal is evidence of correct calculation

    Task("T06",
         "Who wrote 'Pride and Prejudice'? When was she born? "
         "Subtract her birth year from 1813 (year the book was published).",
         3, "37", "37 (1775→1813)"),

    # 4 steps
    Task("T07",
         "What is the speed of light in km/s? "
         "How long does light take to travel from the Sun to the Earth "
         "(distance: 149.6 million km)? Express the answer in minutes, rounded to 1 decimal.",
         4, "8", "~8.3 minutes"),

    Task("T08",
         "Find the height of Mount Everest in metres. "
         "Find the height of K2 in metres. "
         "Calculate the difference. "
         "What percentage taller is Everest than K2? Round to 2 decimal places.",
         4, "%", "~3.5%"),

    # 5-6 steps
    Task("T09",
         "Find the year Albert Einstein was born. "
         "Find the year he published the special theory of relativity. "
         "Find the year he won the Nobel Prize. "
         "Calculate: (Nobel year - birth year) - (relativity year - birth year). "
         "What does this difference represent?",
         5, "16", "16 years gap (1905→1921)"),

    # 7-8 steps
    Task("T10",
         "Find the population of the three largest cities in Japan: Tokyo, Osaka, Yokohama. "
         "Sum their populations. "
         "Find Japan's total population. "
         "Calculate what percentage of Japan's population lives in those three cities. "
         "Round to 1 decimal place.",
         7, "%", "~25% (rough estimate)"),
]


# ── Failure classification ─────────────────────────────────────────────────────

FAILURE_NONE = "none"
FAILURE_TOOL = "tool_error"
FAILURE_REASONING = "reasoning"
FAILURE_PLANNING = "planning"
FAILURE_TIMEOUT = "timeout"
FAILURE_WRONG = "wrong_answer"


@dataclass
class TaskResult:
    task: Task
    answer: Optional[str] = None
    success: bool = False
    failure_type: str = FAILURE_NONE
    fail_step: Optional[int] = None
    total_tool_calls: int = 0
    steps_taken: int = 0
    elapsed_sec: float = 0.0
    tool_errors: list[str] = field(default_factory=list)


# ── Agent runner ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise reasoning agent. Answer questions step by step using "
    "tools when needed. Be methodical — search for each fact separately, "
    "then calculate. When you have the complete answer, respond with just "
    "the final answer (no tool calls)."
)


def run_agent(task: Task, max_steps: int = 12, verbose: bool = False) -> TaskResult:
    """Run the standard tool_use agent on one task and classify the outcome."""
    client = get_client()
    result = TaskResult(task=task)
    messages = [{"role": "user", "content": task.question}]
    t0 = time.time()
    step_num = 0

    if verbose:
        print(f"\n{'─'*60}")
        print(f"[{task.id}] {task.question[:60]}...")
        print(f"  Min steps: {task.min_steps}")

    for iteration in range(max_steps):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
        except Exception as e:
            result.failure_type = FAILURE_TOOL
            result.fail_step = step_num
            result.answer = f"API error: {e}"
            break

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    result.answer = block.text.strip()
            break

        elif response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                step_num += 1
                result.total_tool_calls += 1
                tool_output = dispatch_tool(block.name, block.input)

                if verbose:
                    arg = list(block.input.values())[0][:50] if block.input else ""
                    print(f"  Step {step_num}: {block.name}({arg}) → {tool_output[:60]}")

                # Detect tool errors
                if tool_output.startswith(("Error:", "Calculator error:", "Search error:", "No Wikipedia")):
                    result.tool_errors.append(f"step{step_num}: {tool_output[:60]}")
                    if result.fail_step is None:
                        result.fail_step = step_num
                        result.failure_type = FAILURE_TOOL

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_output,
                })

            messages.append({"role": "user", "content": tool_results})

        else:
            result.failure_type = FAILURE_PLANNING
            result.fail_step = step_num
            break

    result.steps_taken = step_num
    result.elapsed_sec = time.time() - t0

    # Classify success / failure
    if result.answer is None:
        result.failure_type = FAILURE_TIMEOUT
        result.fail_step = step_num
    elif result.failure_type == FAILURE_NONE:
        # Check if the answer matches the expected pattern
        if task.success_pattern and task.success_pattern.lower() in (result.answer or "").lower():
            result.success = True
        else:
            result.success = False
            result.failure_type = FAILURE_WRONG
            if result.fail_step is None:
                result.fail_step = step_num  # failure at the final step

    if verbose:
        status = "✓ SUCCESS" if result.success else f"✗ {result.failure_type.upper()}"
        print(f"  {status}: {(result.answer or '')[:80]}")
        print(f"  Tool calls: {result.total_tool_calls} | Time: {result.elapsed_sec:.1f}s")

    return result


# ── Analysis and plotting ─────────────────────────────────────────────────────

def analyse_results(results: list[TaskResult]) -> dict:
    """Compute per-step-count statistics."""
    from collections import defaultdict

    by_steps: dict[int, list[TaskResult]] = defaultdict(list)
    for r in results:
        by_steps[r.task.min_steps].append(r)

    stats = {}
    for min_steps in sorted(by_steps.keys()):
        group = by_steps[min_steps]
        n = len(group)
        successes = sum(r.success for r in group)
        stats[min_steps] = {
            "n": n,
            "success_rate": successes / n,
            "avg_tool_calls": sum(r.total_tool_calls for r in group) / n,
            "failure_types": [r.failure_type for r in group if not r.success],
        }
    return stats


def print_results_table(results: list[TaskResult]):
    """Print a formatted results table."""
    print("\n" + "=" * 72)
    print("LONG-HORIZON FAILURE ANALYSIS")
    print("=" * 72)
    print(f"{'ID':<4} {'Min':>3} {'Calls':>5} {'OK':>3} {'Failure Type':<16} Answer (truncated)")
    print("-" * 72)

    for r in results:
        status = "✓" if r.success else "✗"
        fail = r.failure_type if not r.success else "-"
        ans = (r.answer or "")[:30]
        print(f"{r.task.id:<4} {r.task.min_steps:>3} {r.total_tool_calls:>5} "
              f"{status:>3} {fail:<16} {ans}")

    print("\n" + "─" * 40)
    stats = analyse_results(results)
    print(f"\n{'Steps':>5}  {'Tasks':>5}  {'Success%':>9}  Failure types")
    print("-" * 40)
    for min_steps, s in stats.items():
        ftypes = ", ".join(s["failure_types"]) if s["failure_types"] else "-"
        print(f"{min_steps:>5}  {s['n']:>5}  {s['success_rate']:>8.0%}  {ftypes}")

    # Overall
    total_success = sum(r.success for r in results)
    print(f"\nOverall: {total_success}/{len(results)} ({total_success/len(results):.0%}) success")


def plot_results(results: list[TaskResult], output_dir: str = "results"):
    """
    Plot success rate vs required steps.
    Saves to output_dir/day5_long_horizon.png
    Falls back to text if matplotlib unavailable.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    stats = analyse_results(results)
    steps_list = sorted(stats.keys())
    rates = [stats[s]["success_rate"] for s in steps_list]
    counts = [stats[s]["n"] for s in steps_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Agent Long-Horizon Failure Analysis", fontsize=14, fontweight="bold")

    # Left: success rate vs steps
    ax1.bar(steps_list, rates, color=["#2ecc71" if r >= 0.6 else "#e74c3c" for r in rates],
            edgecolor="black", alpha=0.8)
    ax1.set_xlabel("Required steps (min tool calls)")
    ax1.set_ylabel("Success rate")
    ax1.set_title("Success Rate vs Task Complexity")
    ax1.set_ylim(0, 1.1)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax1.set_xticks(steps_list)
    # Annotate bars with n
    for x, r, n in zip(steps_list, rates, counts):
        ax1.text(x, r + 0.03, f"n={n}", ha="center", fontsize=9)

    # Add theoretical (1-0.1)^n curve
    import math
    x_theory = list(range(1, max(steps_list) + 1))
    y_theory = [(1 - 0.10) ** x for x in x_theory]
    ax1.plot(x_theory, y_theory, "k--", alpha=0.5, label="Theoretical (ε=0.10)")
    ax1.legend(fontsize=8)

    # Right: failure type breakdown
    failure_counts: dict[str, int] = {}
    for r in results:
        if not r.success:
            ft = r.failure_type
            failure_counts[ft] = failure_counts.get(ft, 0) + 1

    if failure_counts:
        labels = list(failure_counts.keys())
        values = [failure_counts[l] for l in labels]
        colors = {"tool_error": "#e74c3c", "wrong_answer": "#e67e22",
                  "timeout": "#3498db", "reasoning": "#9b59b6",
                  "planning": "#1abc9c"}
        bar_colors = [colors.get(l, "#95a5a6") for l in labels]
        ax2.bar(labels, values, color=bar_colors, edgecolor="black", alpha=0.8)
        ax2.set_xlabel("Failure type")
        ax2.set_ylabel("Count")
        ax2.set_title("Failure Type Distribution")
    else:
        ax2.text(0.5, 0.5, "No failures!", ha="center", va="center",
                 fontsize=16, transform=ax2.transAxes)
        ax2.set_title("Failure Type Distribution")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "day5_long_horizon.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")
    plt.close()


def save_results_json(results: list[TaskResult], output_dir: str = "results"):
    """Save raw results to JSON for later analysis."""
    os.makedirs(output_dir, exist_ok=True)
    data = [
        {
            "id": r.task.id,
            "question": r.task.question,
            "min_steps": r.task.min_steps,
            "answer": r.answer,
            "success": r.success,
            "failure_type": r.failure_type,
            "fail_step": r.fail_step,
            "total_tool_calls": r.total_tool_calls,
            "steps_taken": r.steps_taken,
            "elapsed_sec": r.elapsed_sec,
            "tool_errors": r.tool_errors,
        }
        for r in results
    ]
    out_path = os.path.join(output_dir, "day5_results.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Day 5: Long-Horizon Failure")
    parser.add_argument("--tasks", type=str, default="all",
                        help="Comma-separated task IDs (e.g. T01,T03) or 'all'")
    parser.add_argument("--max_steps", type=int, default=12)
    parser.add_argument("--output_dir", type=str, default="../results")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Select tasks
    if args.tasks == "all":
        selected_tasks = TASKS
    else:
        task_ids = {t.strip().upper() for t in args.tasks.split(",")}
        selected_tasks = [t for t in TASKS if t.id in task_ids]
        if not selected_tasks:
            print(f"No matching tasks for: {args.tasks}")
            return

    print(f"Running {len(selected_tasks)} tasks (max_steps={args.max_steps})")
    print("This will make multiple API calls. Estimated ~5-15 min for all 10 tasks.")

    results = []
    for task in selected_tasks:
        r = run_agent(task, max_steps=args.max_steps, verbose=args.verbose)
        results.append(r)
        # Always show compact progress
        status = "✓" if r.success else "✗"
        print(f"  {status} {task.id} [{task.min_steps} steps needed] "
              f"{r.total_tool_calls} calls → {(r.answer or 'None')[:40]}")

    print_results_table(results)
    save_results_json(results, args.output_dir)

    if not args.no_plot:
        plot_results(results, args.output_dir)


if __name__ == "__main__":
    main()
