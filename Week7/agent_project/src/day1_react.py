"""
day1_react.py — Day 1: Minimal ReAct Agent Loop

What you'll learn
─────────────────
1. The agent loop abstraction:
     state → LLM → action → tool → observation → state update → repeat

2. Why text-based parsing (not native tool_use) is shown first:
   - You see the raw loop mechanics without abstraction hiding them.
   - The LLM emits "Thought / Action" in plain text; we parse it manually.
   - This exposes failure modes: format drift, partial output, hallucinated tools.
   - Day 2 then shows how native JSON schemas fix those failure modes.

3. The three action types:
     Action: search[<query>]       → wikipedia_search()
     Action: calculate[<expr>]     → python_calculator()
     Action: finish[<answer>]      → terminate loop

Loop flow
─────────
  for step in range(max_steps):
      messages = build_prompt(state)           # history → Claude messages
      response = LLM(messages)                 # one LLM call per step
      thought, action, arg = parse(response)   # extract Thought + Action
      observation = execute_tool(action, arg)  # run the tool
      state.history.append(step)               # update state
      if action == "finish": break             # terminal condition

Why single-step CoT ≠ multi-step reliability (½ page)
──────────────────────────────────────────────────────
Chain-of-thought (CoT) prompting asks the model to "think step by step" but
still produces the ENTIRE answer in ONE forward pass. Every intermediate
reasoning token is generated autoregressively — each token conditions on all
previous tokens in the same response. This means:

  1. No error recovery: if token T₅₀ is wrong (e.g., wrong arithmetic result),
     tokens T₅₁ onward are conditioned on that error and will propagate it.
     The model cannot "go back" within one response.

  2. No external grounding: hallucinated facts cannot be corrected mid-response
     because there's no external oracle being consulted between steps.

  3. Compounding probability: if each reasoning step has 90% accuracy, a 5-step
     chain has (0.9)^5 ≈ 59% end-to-end accuracy. A 10-step chain drops to 35%.

A multi-step agent loop breaks this by:
  - Executing each action against a real tool (external ground truth).
  - Feeding the tool's response back as a new "Observation" in the context.
  - Allowing the model to REVISE its plan at each step based on real data.

The trade-off: each step costs one API call. But the gain is that errors
can only propagate one step before being corrected by a real tool result.
This is the core reliability argument for agentic systems over CoT.
"""

import re
import sys
import argparse
from dataclasses import dataclass, field
from typing import Optional

# Ensure src/ is on the path when run directly
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from utils import get_client, wikipedia_search, python_calculator, MODEL


# ── System prompt ─────────────────────────────────────────────────────────────
#
# We use a strict text format (not JSON tool_use) to show raw loop mechanics.
# The model must emit EXACTLY: "Thought: ...\nAction: verb[arg]"
# Anything else gets parsed as a finish action with an error note.

SYSTEM_PROMPT = """You are a step-by-step reasoning agent. You answer questions by interleaving \
Thought and Action until you have enough information to give the final answer.

At EACH step, output EXACTLY this format (one Action per step):

Thought: <your reasoning about what to do next>
Action: search[<wikipedia query>]

OR

Thought: <your reasoning>
Action: calculate[<python math expression>]

OR

Thought: <your reasoning>
Action: finish[<your final answer>]

Rules:
- search[] looks up a Wikipedia article. Use specific queries.
- calculate[] evaluates Python math expressions: +,-,*,/,**,sqrt,pi,log,etc.
- finish[] ONLY when you have the final answer. Do not finish early.
- ONE action per response. Wait for the Observation before proceeding.
- Never make up facts. If unsure, search first.
- You will receive "Observation: <result>" after each action. Use it.
"""


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Step:
    """A single thought-action-observation triple."""
    thought: str
    action: str        # "search" | "calculate" | "finish"
    action_input: str
    observation: str = ""


@dataclass
class AgentState:
    """Full state of one agent run."""
    question: str
    history: list[Step] = field(default_factory=list)
    answer: Optional[str] = None
    steps: int = 0
    success: bool = False


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_messages(state: AgentState) -> list[dict]:
    """
    Convert agent state to Claude's messages list.

    Each completed step becomes:
      assistant: "Thought: ...\nAction: verb[arg]"
      user:      "Observation: ..."

    The current question starts the conversation as the first user message.
    """
    messages = [{"role": "user", "content": state.question}]
    for step in state.history:
        # The assistant's previous Thought + Action
        messages.append({
            "role": "assistant",
            "content": f"Thought: {step.thought}\nAction: {step.action}[{step.action_input}]",
        })
        # The tool's response as a new user turn
        messages.append({
            "role": "user",
            "content": f"Observation: {step.observation}",
        })
    return messages


# ── Response parser ───────────────────────────────────────────────────────────

_ACTION_RE = re.compile(
    r"Action:\s*(search|calculate|finish)\[(.+?)\]",
    re.IGNORECASE | re.DOTALL,
)
_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\Z)", re.DOTALL)


def parse_response(text: str) -> tuple[str, str, str]:
    """
    Parse LLM response text → (thought, action_name, action_input).

    Returns ("", "finish", "PARSE_ERROR: ...") if the format is wrong.
    This is a key failure mode of text-based ReAct: format drift.
    Day 2 (tool_calling.py) eliminates this with JSON schemas.
    """
    thought_m = _THOUGHT_RE.search(text)
    thought = thought_m.group(1).strip() if thought_m else ""

    action_m = _ACTION_RE.search(text)
    if not action_m:
        return thought, "finish", f"PARSE_ERROR: no valid action in response: {text[:100]}"

    return thought, action_m.group(1).lower(), action_m.group(2).strip()


# ── Tool executor ─────────────────────────────────────────────────────────────

def execute_action(action: str, action_input: str) -> str:
    """Execute a parsed action and return the observation string."""
    if action == "search":
        return wikipedia_search(action_input)
    elif action == "calculate":
        return python_calculator(action_input)
    elif action == "finish":
        # finish[] is terminal; the observation is the answer itself
        return f"[DONE] {action_input}"
    else:
        return f"Unknown action '{action}'. Use search, calculate, or finish."


# ── Main agent loop ───────────────────────────────────────────────────────────

def run_react_agent(
    question: str,
    max_steps: int = 10,
    verbose: bool = True,
) -> AgentState:
    """
    Run the ReAct agent loop on a question.

    Loop invariant: after each step, state.history holds all
    (thought, action, observation) triples so the LLM has full context.
    """
    client = get_client()
    state = AgentState(question=question)

    if verbose:
        print(f"\n{'='*65}")
        print(f"QUESTION: {question}")
        print("=" * 65)

    for step_num in range(1, max_steps + 1):
        messages = build_messages(state)

        # One LLM call per loop iteration
        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        text = response.content[0].text
        thought, action, action_input = parse_response(text)
        observation = execute_action(action, action_input)

        step = Step(
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
        )
        state.history.append(step)
        state.steps += 1

        if verbose:
            print(f"\n── Step {step_num} ──────────────────────────────────")
            print(f"Thought   : {thought[:120]}")
            print(f"Action    : {action}[{action_input[:80]}]")
            print(f"Observation: {observation[:200]}")

        if action == "finish":
            state.answer = action_input
            state.success = "PARSE_ERROR" not in action_input
            if verbose:
                print(f"\n{'='*65}")
                print(f"FINAL ANSWER: {state.answer}")
                print(f"Steps taken : {state.steps}")
            break

    if state.answer is None:
        state.answer = "FAILED: max steps exceeded"
        state.success = False
        if verbose:
            print(f"\nFAILED after {max_steps} steps")

    return state


# ── Test suite ────────────────────────────────────────────────────────────────

# Tasks designed to require both search + calculate in combination.
TEST_CASES = [
    # Pure math (1-2 steps)
    "What is 2 to the power of 10, divided by 32?",
    "What is the area of a circle with radius 7? Round to 2 decimal places.",
    # Fact lookup (1 step)
    "How many days are in a standard (non-leap) year? Multiply that by 24.",
    # Lookup + arithmetic (2-3 steps)
    "What year did World War II end? Take the square root of that year.",
    "What is the speed of light in meters per second? Divide it by 1 billion.",
]


def main():
    parser = argparse.ArgumentParser(description="Day 1: Minimal ReAct Agent")
    parser.add_argument("--question", type=str, help="Single question to answer")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--all_tests", action="store_true", help="Run all test cases")
    parser.add_argument("--quiet", action="store_true", help="Suppress step-by-step output")
    args = parser.parse_args()

    verbose = not args.quiet

    if args.question:
        run_react_agent(args.question, max_steps=args.max_steps, verbose=verbose)

    elif args.all_tests:
        results = []
        for q in TEST_CASES:
            state = run_react_agent(q, max_steps=args.max_steps, verbose=verbose)
            results.append(state)

        print("\n" + "=" * 65)
        print("SUMMARY")
        print("=" * 65)
        print(f"{'Q':<52} {'Steps':>5}  {'Result'}")
        print("-" * 65)
        for s in results:
            status = "✓" if s.success else "✗"
            q_short = s.question[:50] + ".." if len(s.question) > 50 else s.question
            ans_short = (s.answer or "")[:30]
            print(f"{status} {q_short:<51} {s.steps:>5}  {ans_short}")

        n = len(results)
        success_rate = sum(s.success for s in results) / n
        avg_steps = sum(s.steps for s in results) / n
        print("-" * 65)
        print(f"Success rate: {success_rate:.0%} ({sum(s.success for s in results)}/{n})")
        print(f"Average steps: {avg_steps:.1f}")

    else:
        # Default: run one example to show the loop
        run_react_agent(TEST_CASES[0], max_steps=args.max_steps, verbose=verbose)


if __name__ == "__main__":
    main()
