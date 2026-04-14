"""
day4_failure_modes.py — Production Failure Modes in LLM Agent Systems
======================================================================

CORE INSIGHT
------------
"Agents are distributed systems with a language model core."

Every failure mode you know from distributed systems applies — plus a new
class of failures that arise specifically because the orchestrator is a
probabilistic language model.  Unlike deterministic code, LLMs fail
silently, gradually, and in ways that pass surface-level checks.

THE SIX FAILURE MODES
---------------------

1. CONTEXT WINDOW OVERFLOW
   The agent keeps appending observations, tool results, and conversation
   history until the total exceeds max_context_len.  At that point:
   - Hard limit: crash / API error
   - Soft limit: silent truncation — the LLM never "sees" the oldest messages
   Signal: response degrades as context grows; prompt token count approaches limit.
   Fix: sliding-window context, summarization, or external memory.

2. TOOL DEADLOCKS
   Agent calls Tool A which waits for Tool B which waits for Tool A.
   Or: agent calls a tool that blocks indefinitely (network timeout, hung API).
   Because the agent loop awaits each tool synchronously, one stuck tool
   freezes the entire session.
   Fix: per-tool timeouts, circuit breakers, max retry limits.

3. HALLUCINATED FUNCTION CALLS
   The model generates a tool call with a plausible-but-wrong name, wrong
   argument types, or argument values it fabricated (e.g., a file path that
   doesn't exist).  The execution layer parses it, calls the function, and
   may: crash, return unexpected output, or silently succeed with wrong data.
   Fix: JSON schema validation before execution; whitelist of allowed tools.

4. RETRY LOOPS
   On tool failure, many agents retry.  Without a global budget the agent
   can enter infinite retry loops — consuming tokens, time, and money while
   making no progress.
   Fix: global step counter; exponential backoff with jitter; max_attempts cap.

5. LATENCY CASCADES
   In a multi-agent pipeline, each agent calls the next synchronously.
   Latency compounds: if each of 5 hops has P99=2s, the pipeline P99 ≈ 10s.
   Under load, slow responses trigger timeouts → retries → more load → cascade.
   Fix: async parallel execution where possible; request hedging; load shedding.

6. MEMORY LEAKS IN AGENT LOOPS
   Python objects (embeddings, model states, conversation history) accumulate
   in long-running agent sessions.  Common in multi-tenant servers where sessions
   are never explicitly cleaned up.
   Fix: explicit session teardown; weak references; bounded history buffers.

OUTPUT
------
    results/day4/day4_summary.txt

INTERVIEW TAKEAWAY
------------------
Q: What breaks first in agent systems?
A: Tool reliability is the weakest link — it's external and probabilistic.
   A tool with 95% success rate run 10 times in a chain has 60% success.
   Add context growth and retry loops and you get cascading degradation.
"""

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "day4")


# ════════════════════════════════════════════════════════════════════════════
# SHARED TYPES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ToolCall:
    name: str
    arguments: dict

@dataclass
class ToolResult:
    success: bool
    value: Any
    error: Optional[str] = None


# ════════════════════════════════════════════════════════════════════════════
# 1.  CONTEXT WINDOW OVERFLOW
# ════════════════════════════════════════════════════════════════════════════

class ContextWindowDemo:
    """
    Simulate an agent that accumulates context across turns until it overflows.
    """
    MAX_TOKENS = 300  # deliberately small for demonstration purposes
    # In production systems this would be 4096–128k; we use 300 so the demo
    # triggers overflow within a few turns and you can observe the truncation
    # behavior and its silent consequences.

    def __init__(self):
        self.messages: list[dict] = []
        self.token_count = 0

    def _estimate_tokens(self, text: str) -> int:
        # ~4 chars per token is a rough heuristic for English text
        return max(1, len(text) // 4)

    def add_message(self, role: str, content: str) -> dict:
        tokens = self._estimate_tokens(content)
        self.token_count += tokens
        msg = {"role": role, "content": content, "tokens": tokens}
        self.messages.append(msg)
        return msg

    def check_overflow(self) -> bool:
        return self.token_count >= self.MAX_TOKENS

    def truncate_oldest(self, target_pct: float = 0.8) -> int:
        """
        Remove oldest messages until under target_pct of limit.
        Returns number of tokens freed.
        """
        target = int(self.MAX_TOKENS * target_pct)
        freed = 0
        while self.token_count > target and self.messages:
            removed = self.messages.pop(0)
            self.token_count -= removed["tokens"]
            freed += removed["tokens"]
        return freed

    def simulate(self, n_turns: int = 20) -> list[dict]:
        events = []
        tool_result_template = (
            "Tool returned: " + "x" * 200  # long tool output per turn
        )
        for turn in range(n_turns):
            self.add_message("user", f"Turn {turn}: Please check the status of all 50 services.")
            self.add_message("assistant", "I'll check each service now. " + "Analyzing... " * 10)
            self.add_message("tool", tool_result_template)

            event = {
                "turn": turn,
                "total_tokens": self.token_count,
                "overflow": self.check_overflow(),
            }

            if self.check_overflow():
                freed = self.truncate_oldest()
                event["action"] = f"truncated {freed} tokens (context overflow)"
                event["risk"] = "Agent lost early conversation — may repeat work or contradict itself"

            events.append(event)

        return events


def run_context_overflow(smoke: bool = False) -> list[str]:
    print(f"\n{'─'*60}")
    print("  FAILURE 1: Context Window Overflow")
    print(f"{'─'*60}")

    demo = ContextWindowDemo()
    n_turns = 5 if smoke else 15
    events = demo.simulate(n_turns)

    lines = []
    overflow_count = 0
    for ev in events:
        overflowed = ev.get("overflow", False)
        if overflowed:
            overflow_count += 1
        action = ev.get("action", "")
        risk = ev.get("risk", "")
        row = (
            f"  turn={ev['turn']:2d} | tokens={ev['total_tokens']:5d} | "
            f"{'OVERFLOW' if overflowed else 'ok':8s} | {action}"
        )
        print(row)
        lines.append(row)

    insight = (
        f"\n  Overflows in {n_turns} turns: {overflow_count}\n"
        f"  Risk: Silent truncation removes EARLY context — agent loses\n"
        f"  task framing and may contradict previous decisions.\n"
        f"  Fix: Sliding-window + summarization of dropped messages."
    )
    print(insight)
    lines.append(insight)
    return lines


# ════════════════════════════════════════════════════════════════════════════
# 2.  TOOL DEADLOCKS
# ════════════════════════════════════════════════════════════════════════════

class ToolRegistry:
    def __init__(self, timeout_s: float = 2.0):
        self.timeout_s = timeout_s
        self.call_count: dict[str, int] = {}

    def call(self, tool_name: str, *args, simulate_hang: bool = False) -> ToolResult:
        self.call_count[tool_name] = self.call_count.get(tool_name, 0) + 1

        if simulate_hang:
            print(f"    [tool={tool_name}] Hanging... (timeout={self.timeout_s}s)")
            # In a real system this would block. We simulate with a short sleep.
            time.sleep(min(self.timeout_s, 0.1))  # keep test fast
            return ToolResult(success=False, value=None, error=f"TimeoutError after {self.timeout_s}s")

        # Simulated successful tool call
        return ToolResult(success=True, value=f"{tool_name} result for {args}")


def run_tool_deadlock(smoke: bool = False) -> list[str]:
    print(f"\n{'─'*60}")
    print("  FAILURE 2: Tool Deadlock / Timeout")
    print(f"{'─'*60}")

    registry = ToolRegistry(timeout_s=1.0)
    lines = []

    # Scenario: agent calls a tool that hangs
    tools_to_call = [
        ("fetch_data",    False),
        ("process_data",  False),
        ("external_api",  True),   # This one hangs!
        ("save_results",  False),  # Never reached
    ]

    for tool_name, will_hang in tools_to_call:
        result = registry.call(tool_name, simulate_hang=will_hang)
        status = "OK" if result.success else f"FAILED: {result.error}"
        row = f"  call {tool_name:<20} → {status}"
        print(row)
        lines.append(row)
        if not result.success:
            note = "  ▶ Pipeline stalled. All downstream tools skipped."
            print(note)
            lines.append(note)
            break

    note2 = (
        "\n  Deadlock pattern: without per-call timeouts, ONE slow tool\n"
        "  blocks the entire agent for minutes or indefinitely.\n"
        "  Fix: asyncio.wait_for(tool_call(), timeout=N) + circuit breaker.\n"
        "  Circuit breaker: after K failures in T seconds, fail fast for T_open."
    )
    print(note2)
    lines.append(note2)
    return lines


# ════════════════════════════════════════════════════════════════════════════
# 3.  HALLUCINATED FUNCTION CALLS
# ════════════════════════════════════════════════════════════════════════════

# Simulated LLM outputs — some valid, some hallucinated
SIMULATED_LLM_TOOL_CALLS = [
    # Valid
    '{"name": "read_file", "arguments": {"path": "/data/report.csv"}}',
    '{"name": "search_web", "arguments": {"query": "FSDP tutorial", "max_results": 5}}',
    # Hallucinated tool name
    '{"name": "execute_bash", "arguments": {"command": "rm -rf /"}}',
    # Wrong argument type
    '{"name": "read_file", "arguments": {"path": 12345}}',
    # Missing required argument
    '{"name": "search_web", "arguments": {"max_results": 3}}',
    # Plausible but wrong — file path that sounds real
    '{"name": "read_file", "arguments": {"path": "/etc/shadow"}}',
    # JSON parse error (malformed)
    '{"name": "read_file", "arguments": {path: "/data/x.csv"}}',
]

ALLOWED_TOOLS = {
    "read_file":  {"required": ["path"], "types": {"path": str}},
    "search_web": {"required": ["query"], "types": {"query": str, "max_results": int}},
    "write_file": {"required": ["path", "content"], "types": {"path": str, "content": str}},
}

SAFE_PATHS = re.compile(r"^/data/|^/tmp/|^/home/")


def validate_tool_call(raw: str) -> tuple[bool, str, Optional[ToolCall]]:
    """
    Parse and validate a raw tool-call string from the LLM.
    Returns (is_valid, reason, ToolCall or None).
    """
    # 1. JSON parse
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e}", None

    # 2. Tool name whitelist
    name = obj.get("name", "")
    if name not in ALLOWED_TOOLS:
        return False, f"Unknown tool '{name}' — not in whitelist", None

    args = obj.get("arguments", {})
    spec = ALLOWED_TOOLS[name]

    # 3. Required arguments
    for req in spec["required"]:
        if req not in args:
            return False, f"Missing required argument '{req}'", None

    # 4. Type checking
    for arg_name, expected_type in spec["types"].items():
        if arg_name in args and not isinstance(args[arg_name], expected_type):
            return False, f"Argument '{arg_name}' must be {expected_type.__name__}", None

    # 5. Safety checks (path traversal, sensitive paths)
    if "path" in args:
        path = args["path"]
        if not SAFE_PATHS.match(path):
            return False, f"Path '{path}' outside allowed directories", None

    return True, "OK", ToolCall(name=name, arguments=args)


def run_hallucinated_calls(smoke: bool = False) -> list[str]:
    print(f"\n{'─'*60}")
    print("  FAILURE 3: Hallucinated Function Calls")
    print(f"{'─'*60}")

    calls = SIMULATED_LLM_TOOL_CALLS[:4] if smoke else SIMULATED_LLM_TOOL_CALLS
    lines = []
    blocked = 0

    for i, raw in enumerate(calls):
        is_valid, reason, tc = validate_tool_call(raw)
        status = "VALID  " if is_valid else "BLOCKED"
        if not is_valid:
            blocked += 1
        display_raw = raw if len(raw) < 60 else raw[:57] + "..."
        row = f"  [{i+1}] {status}: {reason}"
        sub = f"       call: {display_raw}"
        print(row)
        print(sub)
        lines.extend([row, sub])

    summary = (
        f"\n  Blocked {blocked}/{len(calls)} calls before execution.\n"
        f"  Without validation, wrong calls could: delete files,\n"
        f"  read sensitive data, or inject unintended side effects.\n"
        f"  Fix: JSON Schema validation + tool whitelist + path allowlist."
    )
    print(summary)
    lines.append(summary)
    return lines


# ════════════════════════════════════════════════════════════════════════════
# 4.  RETRY LOOPS
# ════════════════════════════════════════════════════════════════════════════

def flaky_tool(success_rate: float = 0.3) -> ToolResult:
    """Simulates a tool that fails most of the time."""
    if random.random() < success_rate:
        return ToolResult(success=True, value="data retrieved successfully")
    return ToolResult(success=False, value=None, error="Connection refused")


def agent_with_retry(
    max_attempts: int,
    base_delay: float = 0.01,
    success_rate: float = 0.3,
) -> dict:
    """
    Agent retry logic with exponential backoff + jitter.
    Returns summary of attempts.
    """
    for attempt in range(1, max_attempts + 1):
        result = flaky_tool(success_rate)
        if result.success:
            return {"success": True, "attempts": attempt, "final": "succeeded"}
        delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, base_delay)
        # time.sleep(delay)  # skip for speed in demo

    return {"success": False, "attempts": max_attempts, "final": "gave up (budget exhausted)"}


def run_retry_loops(smoke: bool = False) -> list[str]:
    print(f"\n{'─'*60}")
    print("  FAILURE 4: Retry Loops")
    print(f"{'─'*60}")
    lines = []

    random.seed(42)
    n_sessions = 5 if smoke else 20
    unbounded_costs = []
    bounded_costs = []

    for _ in range(n_sessions):
        # Unbounded: no max_attempts (simulate as 100)
        r_unbound = agent_with_retry(max_attempts=100, success_rate=0.2)
        unbounded_costs.append(r_unbound["attempts"])
        # Bounded: max 5 attempts
        r_bound = agent_with_retry(max_attempts=5, success_rate=0.2)
        bounded_costs.append(r_bound["attempts"])

    avg_unbound = sum(unbounded_costs) / len(unbounded_costs)
    avg_bound   = sum(bounded_costs) / len(bounded_costs)
    max_unbound = max(unbounded_costs)

    row1 = f"  {n_sessions} sessions, tool success_rate=20%"
    row2 = f"  Unbounded retries: avg={avg_unbound:.1f} attempts, max={max_unbound}"
    row3 = f"  Bounded  retries: avg={avg_bound:.1f}  attempts, max=5"
    note = (
        f"\n  Without a retry budget, one session used {max_unbound} attempts.\n"
        f"  Each attempt = one LLM call + one tool call = $$$ and latency.\n"
        f"  Cascading effect: many agents retrying simultaneously can DDoS tools.\n"
        f"  Fix: max_attempts + exponential backoff with jitter + global budget."
    )
    for r in [row1, row2, row3, note]:
        print(r)
        lines.append(r)

    return lines


# ════════════════════════════════════════════════════════════════════════════
# 5.  LATENCY CASCADES
# ════════════════════════════════════════════════════════════════════════════

def simulate_latency_cascade(
    n_hops: int = 5,
    base_p50_ms: float = 200,
    base_p99_ms: float = 800,
    n_samples: int = 1000,
    smoke: bool = False,
) -> list[str]:
    """
    Model a sequential pipeline where each hop has independent latency.
    Demonstrate how tail latency compounds.
    """
    print(f"\n{'─'*60}")
    print("  FAILURE 5: Latency Cascades")
    print(f"{'─'*60}")

    if smoke:
        n_samples = 100

    lines = []

    def sample_latency(p50: float, p99: float, n: int) -> list[float]:
        # Model as log-normal distribution
        # ln(p50) and ln(p99) give us the lognormal params
        mu    = (p50 / 1000)   # convert to seconds
        sigma = (p99 - p50) / (p50 * 3)  # rough spread
        samples = [random.lognormvariate(
            __import__("math").log(mu),
            0.5,
        ) for _ in range(n)]
        return samples

    # Single-hop baseline
    single_latencies = sample_latency(base_p50_ms, base_p99_ms, n_samples)
    single_latencies.sort()

    def percentile(data, pct):
        idx = int(len(data) * pct / 100)
        return data[min(idx, len(data) - 1)] * 1000  # back to ms

    single_p50 = percentile(single_latencies, 50)
    single_p99 = percentile(single_latencies, 99)

    row = f"  Single hop:  P50={single_p50:.0f}ms  P99={single_p99:.0f}ms"
    print(row)
    lines.append(row)

    # N-hop pipeline: sum of independent latencies
    pipeline_latencies = []
    for _ in range(n_samples):
        total = sum(
            random.lognormvariate(__import__("math").log(base_p50_ms / 1000), 0.5)
            for _ in range(n_hops)
        )
        pipeline_latencies.append(total)
    pipeline_latencies.sort()

    pipe_p50 = percentile(pipeline_latencies, 50)
    pipe_p99 = percentile(pipeline_latencies, 99)

    row2 = f"  {n_hops}-hop pipeline: P50={pipe_p50:.0f}ms  P99={pipe_p99:.0f}ms"
    mult_p50 = pipe_p50 / single_p50
    mult_p99 = pipe_p99 / single_p99
    row3 = f"  P50 multiplier: {mult_p50:.1f}×  P99 multiplier: {mult_p99:.1f}×"
    note = (
        f"\n  Tail latency compounds FASTER than P50 in a sequential pipeline.\n"
        f"  With {n_hops} hops, P99 is {mult_p99:.1f}× a single hop's P99.\n"
        f"  Fix: parallel execution where possible; request hedging (send\n"
        f"  duplicate to backup, use first response); timeouts per hop."
    )
    for r in [row2, row3, note]:
        print(r)
        lines.append(r)
    return lines


# ════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    smoke = args.smoke

    os.makedirs(RESULTS_DIR, exist_ok=True)
    random.seed(123)

    print("═"*60)
    print("  Day 4 — Failure Modes in Production Agent Systems")
    print("═"*60)
    print("\n  Thesis: Agents are distributed systems.")
    print("  Every distributed-systems failure mode applies — plus LLM-specific ones.\n")

    all_lines = []

    all_lines.extend(run_context_overflow(smoke))
    all_lines.extend(run_tool_deadlock(smoke))
    all_lines.extend(run_hallucinated_calls(smoke))
    all_lines.extend(run_retry_loops(smoke))
    all_lines.extend(simulate_latency_cascade(smoke=smoke))

    # ── Compound failure rate demonstration ──────────────────────────────────
    print(f"\n{'─'*60}")
    print("  COMPOUNDING FAILURE RATES")
    print(f"{'─'*60}")
    tool_reliability = 0.95
    print(f"\n  If each tool has {tool_reliability:.0%} reliability:")
    for n_tools in [1, 3, 5, 10]:
        chain_reliability = tool_reliability ** n_tools
        print(f"  {n_tools:2d} tools in chain: {chain_reliability:.1%} success rate")
    print()

    # ── Write summary ────────────────────────────────────────────────────────
    summary_path = os.path.join(RESULTS_DIR, "day4_summary.txt")
    with open(summary_path, "w") as f:
        f.write("PRODUCTION FAILURE MODES IN LLM AGENT SYSTEMS\n")
        f.write("=" * 60 + "\n\n")
        f.write("THE SIX FAILURE MODES\n")
        f.write("-" * 40 + "\n")
        modes = [
            ("1. Context Window Overflow",
             "Accumulation of messages exceeds context limit.\n"
             "   Silent truncation drops early context → agent contradicts itself.\n"
             "   Fix: sliding window + summarize dropped messages."),
            ("2. Tool Deadlock",
             "One hung tool blocks the entire synchronous agent loop.\n"
             "   Fix: per-call timeout + circuit breaker."),
            ("3. Hallucinated Function Calls",
             "LLM invents tool names, wrong arg types, or fabricated values.\n"
             "   Fix: JSON schema validation + tool name whitelist."),
            ("4. Retry Loops",
             "Without a budget, agents retry indefinitely → cost explosion.\n"
             "   Fix: max_attempts + exponential backoff with jitter."),
            ("5. Latency Cascades",
             "Sequential pipeline P99 grows super-linearly with hop count.\n"
             "   Fix: parallel execution + request hedging + per-hop timeouts."),
            ("6. Memory Leaks",
             "Long-running sessions accumulate embeddings, history, model states.\n"
             "   Fix: explicit session teardown + bounded history buffers."),
        ]
        for title, desc in modes:
            f.write(f"\n{title}\n{desc}\n")

        f.write("\n\nINTERVIEW ANSWERS\n")
        f.write("-" * 40 + "\n")
        f.write("Q: What breaks first in agent systems?\n")
        f.write("A: Tool reliability. A single tool at 95% reliability run 10×\n")
        f.write("   in a chain gives 60% end-to-end success. Add retry loops\n")
        f.write("   and context growth and you get cascading degradation.\n\n")
        f.write("Q: How do you monitor agent performance?\n")
        f.write("A: Structured logs for every tool call: latency, success/fail,\n")
        f.write("   token count per step. Track P99 latency per tool, error rate\n")
        f.write("   by failure type, total tokens per session. Alert on drift.\n\n")
        f.write("Q: How do you detect silent failures?\n")
        f.write("A: Schema validation on every tool output (not just the final\n")
        f.write("   response). Perplexity checks on the agent's output at each\n")
        f.write("   step. Sanity assertions on intermediate results (e.g.,\n")
        f.write("   file exists after a write call). Human spot-checks on 1% sample.\n\n")
        f.write("\n".join(all_lines))

    print(f"\n  Summary → {summary_path}")
    print("\n  Day 4 complete.")
    print("  Key takeaway: 95% tool reliability × 10 steps = 60% chain success.")
    print("  Design for failure; instrument everything; bound all resources.")


if __name__ == "__main__":
    main()
