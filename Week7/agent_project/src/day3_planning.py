"""
day3_planning.py — Day 3: Planning vs. Reasoning

The core question
─────────────────
Should one LLM do both planning AND executing, or should we separate them?

Two architectures compared
──────────────────────────

ReactAgent (baseline, from Day 1/2)
  • One LLM handles everything inline.
  • Each step: think → pick action → observe → think → ...
  • No upfront task decomposition.
  • Token reasoning at each step determines what comes next.

PlannerExecutorAgent
  • Planner LLM: given the question, output a JSON step-list FIRST.
    E.g. [{"step":1,"task":"Find the year X was born"},
          {"step":2,"task":"Subtract birth year from 2024"},
          {"step":3,"task":"Return the age"}]
  • Executor: for each plan step, run a mini tool-use loop.
  • Separation of concerns: "what to do" ≠ "how to do it".

Critical insight: Why does explicit planning reduce hallucination?
──────────────────────────────────────────────────────────────────
Consider a 5-step task. In ReAct, each step is influenced by the entire
accumulated conversation (question + all prior observations). As the context
grows, the model must infer the current sub-goal from noisy context.

In Planner-Executor, the planner has one job: decompose the task into atomic
sub-goals while the context is CLEAN (just the question). Each executor call
then has a single, clear objective. This reduces entropy in two ways:

  1. Reduced search space per call: the executor doesn't choose among N
     possible next actions — it executes exactly what the planner specified.
  2. Early error detection: if the plan itself is wrong, it fails fast at
     planning time (cheap) rather than after many expensive executor steps.

The trade-off: planner overhead (one extra LLM call) + rigidity (plan cannot
adapt if an observation changes the strategy mid-execution).

Benchmark tasks (5 tasks, 3-5 steps each)
──────────────────────────────────────────
  T1: Population comparison + arithmetic
  T2: Historical date + calculation
  T3: Scientific constant + unit conversion
  T4: Multi-entity fact + comparison
  T5: Chained lookup + aggregation

Metrics
───────
  success_rate   — answer matches expected pattern
  total_calls    — total tool invocations
  total_tokens   — prompt + completion tokens across all LLM calls
  plan_adherence — (Planner only) did executor stay on plan?
"""

import sys
import json
import time
import argparse
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from utils import get_client, TOOL_SCHEMAS, dispatch_tool, MODEL


# ── System prompts ────────────────────────────────────────────────────────────

REACT_SYSTEM = (
    "You are a reasoning agent. Answer questions step by step using the "
    "provided tools. When you have the final answer, respond with plain text "
    "(no tool calls). Be concise."
)

PLANNER_SYSTEM = """You are a task planner. Given a question, decompose it into \
a sequence of atomic sub-tasks that can each be solved with one tool call or \
one reasoning step.

Output ONLY a JSON array, no other text:
[
  {"step": 1, "task": "<description of what to do>"},
  {"step": 2, "task": "<description of what to do>"},
  ...
]

Rules:
- Each task should be a single, atomic action (search one thing OR calculate one thing).
- The final step should always be "Synthesize findings and state the final answer."
- Use 3–6 steps maximum.
- Output valid JSON only — no markdown code fences.
"""

EXECUTOR_SYSTEM = (
    "You are a focused executor. You will receive ONE specific sub-task to "
    "complete. Use tools if needed. When done, respond with ONLY the result "
    "of your sub-task — no extra commentary."
)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    question: str
    agent_name: str
    answer: Optional[str] = None
    success: bool = False
    total_tool_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    elapsed_sec: float = 0.0
    plan: Optional[list[dict]] = None  # Planner only
    error: Optional[str] = None


# ── ReactAgent ────────────────────────────────────────────────────────────────

def run_react_agent(
    question: str,
    max_steps: int = 10,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Standard ReAct agent using Claude's native tool_use API.
    Identical to day2 but tracks token usage for benchmarking.
    """
    client = get_client()
    messages = [{"role": "user", "content": question}]
    res = BenchmarkResult(question=question, agent_name="ReAct")
    t0 = time.time()

    if verbose:
        print(f"\n  [ReAct] {question[:60]}...")

    for _ in range(max_steps):
        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=REACT_SYSTEM,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        res.total_input_tokens += response.usage.input_tokens
        res.total_output_tokens += response.usage.output_tokens

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    res.answer = block.text.strip()
            break

        elif response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    res.total_tool_calls += 1
                    result = dispatch_tool(block.name, block.input)
                    if verbose:
                        print(f"    → {block.name}({list(block.input.values())[0][:50]})")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})

    res.elapsed_sec = time.time() - t0
    res.success = bool(res.answer and len(res.answer) > 2)
    return res


# ── PlannerExecutorAgent ──────────────────────────────────────────────────────

def _call_planner(question: str, client) -> list[dict]:
    """
    Ask the Planner LLM to decompose the question into a JSON step list.

    Returns a list of {"step": int, "task": str} dicts.
    Falls back to a single-step plan on parse failure.
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=PLANNER_SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    text = response.content[0].text.strip()

    # Strip markdown code fences if model included them
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.startswith("```"))

    try:
        plan = json.loads(text)
        if isinstance(plan, list) and all("task" in s for s in plan):
            return plan, response.usage
    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback: treat as single task
    return [{"step": 1, "task": question}], response.usage


def _execute_step(task: str, context: str, client) -> tuple[str, int, int, int]:
    """
    Execute one plan step with the executor LLM.

    Returns (result_text, tool_calls_used, input_tokens, output_tokens).
    """
    messages = [
        {
            "role": "user",
            "content": f"Task: {task}\n\nContext from previous steps:\n{context}",
        }
    ]
    tool_calls_used = 0
    in_tokens = 0
    out_tokens = 0

    for _ in range(5):  # max 5 sub-steps per plan step
        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=EXECUTOR_SYSTEM,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        in_tokens += response.usage.input_tokens
        out_tokens += response.usage.output_tokens

        if response.stop_reason == "end_turn":
            result = next(
                (b.text.strip() for b in response.content if hasattr(b, "text")),
                "No result",
            )
            return result, tool_calls_used, in_tokens, out_tokens

        elif response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls_used += 1
                    result = dispatch_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})

    return "EXECUTOR_MAX_STEPS", tool_calls_used, in_tokens, out_tokens


def run_planner_executor_agent(
    question: str,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    PlannerExecutor: first plan, then execute each step.

    The planner calls add overhead but give the executor clean, atomic tasks.
    """
    client = get_client()
    res = BenchmarkResult(question=question, agent_name="PlannerExecutor")
    t0 = time.time()

    if verbose:
        print(f"\n  [PlannerExecutor] {question[:60]}...")

    # ── Phase 1: Plan ────────────────────────────────────────────────────────
    plan, planner_usage = _call_planner(question, client)
    res.plan = plan
    res.total_input_tokens += planner_usage.input_tokens
    res.total_output_tokens += planner_usage.output_tokens

    if verbose:
        print(f"    Plan ({len(plan)} steps):")
        for s in plan:
            print(f"      {s['step']}. {s['task']}")

    # ── Phase 2: Execute each plan step ──────────────────────────────────────
    context_parts = []
    final_answer = None

    for step_info in plan:
        step_num = step_info["step"]
        task = step_info["task"]

        context = "\n".join(context_parts) if context_parts else "No prior steps."
        step_result, calls, in_tok, out_tok = _execute_step(task, context, client)

        res.total_tool_calls += calls
        res.total_input_tokens += in_tok
        res.total_output_tokens += out_tok
        context_parts.append(f"Step {step_num} ({task}): {step_result}")

        if verbose:
            print(f"    Step {step_num}: {step_result[:80]}")

        # Last step's result is the final answer
        if step_num == len(plan):
            final_answer = step_result

    res.answer = final_answer or (context_parts[-1] if context_parts else None)
    res.elapsed_sec = time.time() - t0
    res.success = bool(res.answer and len(res.answer) > 2)
    return res


# ── Benchmark tasks ───────────────────────────────────────────────────────────

BENCHMARK_TASKS = [
    # T1: lookup + arithmetic (2 facts, 1 calculation)
    "What is the population of Japan divided by the population of Australia? Round to 1 decimal place.",
    # T2: historical date + calculation
    "In what year was the Eiffel Tower completed? How many years ago was that (from 2025)?",
    # T3: scientific constant + conversion
    "What is the boiling point of water in Celsius? Convert it to Fahrenheit using (C * 9/5) + 32.",
    # T4: multi-entity comparison
    "What is taller: Mount Kilimanjaro or Mont Blanc? By how many metres?",
    # T5: chained lookup
    "Who wrote the novel '1984'? In what year was that author born? What is 2024 minus that year?",
]

# Simple expected patterns to check success (substring match, case-insensitive)
EXPECTED_PATTERNS = [
    "4.",        # ~2.6x
    "year",      # some year + "years ago"
    "212",       # 212°F
    "kilimanjaro",  # Kilimanjaro is taller
    "orwell",    # George Orwell
]


def _check_success(answer: Optional[str], pattern: str) -> bool:
    if not answer:
        return False
    return pattern.lower() in answer.lower()


def main():
    parser = argparse.ArgumentParser(description="Day 3: Planning vs Reasoning")
    parser.add_argument("--agent", choices=["react", "planner", "both"],
                        default="both", help="Which agent(s) to run")
    parser.add_argument("--task_idx", type=int, default=None,
                        help="Run only task index 0-4 (omit to run all)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    tasks = (
        [BENCHMARK_TASKS[args.task_idx]]
        if args.task_idx is not None
        else BENCHMARK_TASKS
    )
    patterns = (
        [EXPECTED_PATTERNS[args.task_idx]]
        if args.task_idx is not None
        else EXPECTED_PATTERNS
    )

    react_results = []
    planner_results = []

    for i, (task, pattern) in enumerate(zip(tasks, patterns)):
        task_label = f"T{(args.task_idx if args.task_idx is not None else i) + 1}"
        print(f"\n{'='*65}")
        print(f"{task_label}: {task}")
        print("=" * 65)

        if args.agent in ("react", "both"):
            r = run_react_agent(task, max_steps=10, verbose=verbose)
            r.success = _check_success(r.answer, pattern)
            react_results.append(r)
            if verbose:
                print(f"  ReAct answer: {(r.answer or '')[:100]}")

        if args.agent in ("planner", "both"):
            p = run_planner_executor_agent(task, verbose=verbose)
            p.success = _check_success(p.answer, pattern)
            planner_results.append(p)
            if verbose:
                print(f"  Planner answer: {(p.answer or '')[:100]}")

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("BENCHMARK RESULTS")
    print("=" * 65)
    print(f"{'Task':<6} {'Agent':<16} {'OK':>3} {'Calls':>5} {'InTok':>6} {'OutTok':>7} {'Sec':>5}")
    print("-" * 65)

    all_results = (
        [(r, "ReAct") for r in react_results] +
        [(p, "Planner") for p in planner_results]
    )
    # Sort by task then agent for side-by-side comparison
    for label_idx, (_, res_list) in enumerate(
        [("ReAct", react_results), ("PlannerExecutor", planner_results)]
    ):
        for j, r in enumerate(res_list):
            tidx = (args.task_idx if args.task_idx is not None else j) + 1
            print(f"T{tidx:<5} {r.agent_name:<16} {'✓' if r.success else '✗':>3} "
                  f"{r.total_tool_calls:>5} {r.total_input_tokens:>6} "
                  f"{r.total_output_tokens:>7} {r.elapsed_sec:>5.1f}")

    # Aggregate stats
    if react_results and planner_results:
        print("-" * 65)
        for res_list in [react_results, planner_results]:
            n = len(res_list)
            name = res_list[0].agent_name
            print(
                f"{'AVG':<6} {name:<16} "
                f"{sum(r.success for r in res_list)/n:>3.0%} "
                f"{sum(r.total_tool_calls for r in res_list)/n:>5.1f} "
                f"{sum(r.total_input_tokens for r in res_list)/n:>6.0f} "
                f"{sum(r.total_output_tokens for r in res_list)/n:>7.0f} "
                f"{sum(r.elapsed_sec for r in res_list)/n:>5.1f}"
            )


if __name__ == "__main__":
    main()
