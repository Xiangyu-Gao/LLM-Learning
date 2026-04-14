"""
day5_observability.py — Observability and Logging for LLM Systems
=================================================================

CORE INSIGHT
------------
"You cannot improve what you cannot measure."

In traditional software, observability means logs, metrics, and traces.
In LLM systems, it means all three PLUS:
  - Prompt tracing (what exact prompt reached the model?)
  - Token attribution (which part of the context drove the response?)
  - Eval regression testing (did this deployment make the model worse?)
  - Schema compliance rates (are tool calls well-formed over time?)

The hidden danger: LLM failures are GRADUAL and SILENT.
A model degradation doesn't throw a 500 — it increases hallucination rate
by 3% over two weeks.  Only structured observability catches this.

WHAT TO INSTRUMENT
------------------

1. STRUCTURED LOGS (every request)
   {"timestamp": ..., "session_id": ..., "step": ...,
    "event": "tool_call", "tool": "search_web",
    "latency_ms": 234, "success": true, "tokens_in": 1200, "tokens_out": 48,
    "schema_valid": true, "error": null}

   Key fields: session_id, step, event type, latency, success,
               tokens (in + out), schema validity, error message.

2. LATENCY HISTOGRAMS
   Track P50/P95/P99 per tool, per model, per pipeline stage.
   Alert when P99 crosses threshold (e.g., >2s for user-facing tools).

3. TOKEN USAGE TRACKING
   Cost = tokens × price/token.  Track:
   - Tokens per session (cost allocation)
   - Token growth over time (context bloat detection)
   - Prompt vs completion ratio (prompt-heavy = wasted context)

4. FAILURE TYPE CLASSIFICATION
   Don't just count errors — classify them:
   - schema_violation:  model output wasn't valid JSON / wrong schema
   - tool_timeout:      tool call exceeded timeout
   - context_overflow:  session exceeded token budget
   - hallucination:     output failed factual check (requires eval)
   - retry_exhausted:   hit max_attempts without success

5. PROMPT TRACING
   Store the exact prompt sent to the model (or a hash + metadata).
   Without this, you cannot reproduce failures or debug regressions.

6. EVAL REGRESSION TESTING
   Run a fixed eval suite (e.g., 100 golden examples) before every
   deploy.  Alert if any metric drops by more than epsilon.
   This is your continuous integration for model quality.

7. CANARY DEPLOYMENTS
   Route 5% of traffic to new model version.
   Compare failure rates, token usage, latency to stable version.
   Roll back automatically if canary exceeds error threshold.

EXPERIMENT
----------
    1. Generate synthetic structured logs for a simulated agent system.
    2. Analyze logs:
       - Failure type breakdown (pie / bar chart)
       - Latency histogram per tool
       - Token usage over time
       - Schema violation rate trend
    3. Write an analysis report.

OUTPUT
------
    results/day5/day5_failure_breakdown.png
    results/day5/day5_latency_histogram.png
    results/day5/day5_token_usage.png
    results/day5/day5_analysis_report.txt

INTERVIEW TAKEAWAY
------------------
Q: How do you detect silent failures?
A: Instrument every tool call with structured logs.  Alert on:
   - Schema violation rate > baseline (model started hallucinating format)
   - P99 latency spike (tool degradation)
   - Token growth per session (context bloat)
   - Eval regression on golden suite (model quality drop)
"""

import argparse
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "day5")


# ════════════════════════════════════════════════════════════════════════════
# 1.  LOG DATA MODEL
# ════════════════════════════════════════════════════════════════════════════

FAILURE_TYPES = [
    "schema_violation",
    "tool_timeout",
    "context_overflow",
    "retry_exhausted",
    "hallucination",
    "auth_error",
]

TOOLS = ["search_web", "read_file", "write_file", "fetch_api", "code_exec", "llm_call"]

EVENT_TYPES = ["tool_call", "llm_call", "session_start", "session_end", "error"]


@dataclass
class LogEntry:
    timestamp: datetime
    session_id: str
    step: int
    event: str
    tool: Optional[str] = None
    latency_ms: float = 0.0
    success: bool = True
    tokens_in: int = 0
    tokens_out: int = 0
    schema_valid: bool = True
    error: Optional[str] = None
    failure_type: Optional[str] = None
    model: str = "gpt2-large"
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "step": self.step,
            "event": self.event,
            "tool": self.tool,
            "latency_ms": round(self.latency_ms, 1),
            "success": self.success,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "schema_valid": self.schema_valid,
            "error": self.error,
            "failure_type": self.failure_type,
            "model": self.model,
            "cost_usd": round(self.cost_usd, 6),
        }


# ════════════════════════════════════════════════════════════════════════════
# 2.  SYNTHETIC LOG GENERATOR
# ════════════════════════════════════════════════════════════════════════════

class LogGenerator:
    """
    Generates realistic synthetic logs for an LLM agent system.
    Injects failure patterns that observability should catch.
    """

    # Tool-specific latency parameters (ms): (p50, sigma_log)
    TOOL_LATENCY = {
        "search_web":  (400, 0.6),
        "read_file":   (50,  0.3),
        "write_file":  (80,  0.3),
        "fetch_api":   (200, 0.8),
        "code_exec":   (800, 0.5),
        "llm_call":    (1200, 0.4),
    }

    # Token distribution per event type
    TOKEN_DIST = {
        "llm_call": (800, 150, 200, 50),   # (prompt_mean, prompt_std, out_mean, out_std)
        "tool_call": (0, 0, 100, 30),
    }

    PRICE_PER_TOKEN = 0.000002  # $0.002 per 1K tokens (rough GPT-3.5 pricing)

    def __init__(self, rng: random.Random):
        self.rng = rng
        self._session_counter = 0

    def _session_id(self) -> str:
        self._session_counter += 1
        return f"sess_{self._session_counter:04d}"

    def _latency(self, tool: str) -> float:
        p50, sigma = self.TOOL_LATENCY.get(tool, (300, 0.5))
        return math.exp(self.rng.gauss(math.log(p50), sigma))

    def _tokens(self, event: str) -> tuple[int, int]:
        if event not in self.TOKEN_DIST:
            return 0, 0
        pm, ps, om, os_ = self.TOKEN_DIST[event]
        return (
            max(0, int(self.rng.gauss(pm, ps))),
            max(0, int(self.rng.gauss(om, os_))),
        )

    def generate_session(
        self,
        start_time: datetime,
        n_steps: int,
        failure_rate: float = 0.12,
        schema_violation_rate: float = 0.05,
        timeout_rate: float = 0.04,
    ) -> list[LogEntry]:
        """Generate one agent session with mixed events and failures."""
        session_id = self._session_id()
        entries = []
        t = start_time
        cumulative_tokens = 0

        for step in range(n_steps):
            # Decide event type
            if step == 0:
                event = "session_start"
                tool = None
            elif step == n_steps - 1:
                event = "session_end"
                tool = None
            elif self.rng.random() < 0.4:
                event = "llm_call"
                tool = None
            else:
                event = "tool_call"
                tool = self.rng.choice(TOOLS)

            # Latency
            lat = self._latency(tool or "llm_call")

            # Tokens
            t_in, t_out = self._tokens(event)
            cumulative_tokens += t_in + t_out

            # Context overflow check
            context_overflowed = cumulative_tokens > 8000
            if context_overflowed:
                cumulative_tokens = int(cumulative_tokens * 0.6)  # truncation

            # Failure injection
            will_fail = self.rng.random() < failure_rate
            schema_bad = self.rng.random() < schema_violation_rate
            timed_out = (tool is not None) and (self.rng.random() < timeout_rate)

            failure_type = None
            error = None
            success = True

            if timed_out:
                lat = self.rng.uniform(1900, 2100)
                success = False
                failure_type = "tool_timeout"
                error = f"TimeoutError: {tool} exceeded 2000ms"
            elif context_overflowed and self.rng.random() < 0.3:
                success = False
                failure_type = "context_overflow"
                error = "ContextLengthExceededError"
            elif will_fail:
                success = False
                failure_type = self.rng.choice(FAILURE_TYPES)
                error = f"{failure_type}: simulated failure"

            cost = (t_in + t_out) * self.PRICE_PER_TOKEN

            entry = LogEntry(
                timestamp=t,
                session_id=session_id,
                step=step,
                event=event,
                tool=tool,
                latency_ms=lat,
                success=success,
                tokens_in=t_in,
                tokens_out=t_out,
                schema_valid=not schema_bad,
                error=error,
                failure_type=failure_type,
                cost_usd=cost,
            )
            entries.append(entry)
            t += timedelta(milliseconds=lat + self.rng.uniform(10, 100))

        return entries

    def generate_corpus(
        self,
        n_sessions: int,
        start: datetime,
        smoke: bool = False,
    ) -> list[LogEntry]:
        all_entries = []
        n_steps_range = (3, 8) if smoke else (5, 20)
        t = start

        for _ in range(n_sessions):
            n_steps = self.rng.randint(*n_steps_range)
            # Introduce a "degradation window" in the middle of the corpus
            # to simulate a model regression deployment
            session_idx = len(all_entries)
            total_expected = n_sessions * 10
            in_degradation = (total_expected * 0.4) < session_idx < (total_expected * 0.7)
            failure_rate = 0.25 if in_degradation else 0.10

            entries = self.generate_session(
                start_time=t,
                n_steps=n_steps,
                failure_rate=failure_rate,
                schema_violation_rate=0.08 if in_degradation else 0.03,
            )
            all_entries.extend(entries)
            t += timedelta(minutes=self.rng.uniform(0.5, 5))

        return all_entries


# ════════════════════════════════════════════════════════════════════════════
# 3.  LOG ANALYZER
# ════════════════════════════════════════════════════════════════════════════

class LogAnalyzer:
    def __init__(self, entries: list[LogEntry]):
        self.entries = entries

    # ── Failure type breakdown ─────────────────────────────────────────────

    def failure_breakdown(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.entries:
            if e.failure_type:
                counts[e.failure_type] = counts.get(e.failure_type, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def overall_error_rate(self) -> float:
        total = len(self.entries)
        failed = sum(1 for e in self.entries if not e.success)
        return failed / total if total > 0 else 0.0

    # ── Schema violations ──────────────────────────────────────────────────

    def schema_violation_rate(self) -> float:
        llm_calls = [e for e in self.entries if e.event == "llm_call"]
        if not llm_calls:
            return 0.0
        violations = sum(1 for e in llm_calls if not e.schema_valid)
        return violations / len(llm_calls)

    def schema_violation_over_time(self, bucket_size: int = 50) -> list[tuple[int, float]]:
        """Compute schema violation rate in rolling buckets of N entries."""
        llm_entries = [e for e in self.entries if e.event == "llm_call"]
        buckets = []
        for i in range(0, len(llm_entries), bucket_size):
            chunk = llm_entries[i:i+bucket_size]
            rate = sum(1 for e in chunk if not e.schema_valid) / len(chunk)
            buckets.append((i, rate))
        return buckets

    # ── Token usage ────────────────────────────────────────────────────────

    def token_usage_summary(self) -> dict:
        total_in  = sum(e.tokens_in  for e in self.entries)
        total_out = sum(e.tokens_out for e in self.entries)
        total_cost = sum(e.cost_usd for e in self.entries)
        sessions = len(set(e.session_id for e in self.entries))
        return {
            "total_tokens_in":  total_in,
            "total_tokens_out": total_out,
            "total_tokens":     total_in + total_out,
            "total_cost_usd":   total_cost,
            "sessions":         sessions,
            "avg_tokens_per_session": (total_in + total_out) / max(sessions, 1),
            "prompt_to_completion_ratio": total_in / max(total_out, 1),
        }

    def token_usage_over_time(self) -> list[tuple[datetime, int]]:
        """Cumulative token usage over time."""
        cumulative = 0
        series = []
        for e in sorted(self.entries, key=lambda x: x.timestamp):
            cumulative += e.tokens_in + e.tokens_out
            series.append((e.timestamp, cumulative))
        return series

    # ── Latency ────────────────────────────────────────────────────────────

    def latency_by_tool(self) -> dict[str, dict]:
        by_tool: dict[str, list[float]] = {}
        for e in self.entries:
            if e.tool and e.latency_ms > 0:
                by_tool.setdefault(e.tool, []).append(e.latency_ms)
        result = {}
        for tool, lats in by_tool.items():
            lats.sort()
            result[tool] = {
                "p50": lats[int(len(lats) * 0.50)],
                "p95": lats[int(len(lats) * 0.95)],
                "p99": lats[min(int(len(lats) * 0.99), len(lats)-1)],
                "count": len(lats),
            }
        return result

    def print_latency_table(self) -> None:
        by_tool = self.latency_by_tool()
        print(f"\n  {'Tool':<15} {'P50':>8} {'P95':>8} {'P99':>8} {'Count':>7}")
        print(f"  {'─'*50}")
        for tool, stats in sorted(by_tool.items()):
            print(
                f"  {tool:<15} {stats['p50']:>7.0f}ms {stats['p95']:>7.0f}ms "
                f"{stats['p99']:>7.0f}ms {stats['count']:>7}"
            )


# ════════════════════════════════════════════════════════════════════════════
# 4.  PLOTS
# ════════════════════════════════════════════════════════════════════════════

def plot_failure_breakdown(breakdown: dict[str, int], out_dir: str) -> None:
    if not breakdown:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    names  = list(breakdown.keys())
    counts = list(breakdown.values())
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    bars = ax.barh(names, counts, color=colors)
    ax.bar_label(bars, padding=3)
    ax.set_xlabel("Count")
    ax.set_title("Failure Type Breakdown")
    ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(out_dir, "day5_failure_breakdown.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Failure breakdown → {path}")


def plot_latency_histogram(analyzer: LogAnalyzer, out_dir: str) -> None:
    by_tool = analyzer.latency_by_tool()
    tools = list(by_tool.keys())[:4]  # top 4 tools

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, tool in zip(axes, tools):
        lats = [e.latency_ms for e in analyzer.entries
                if e.tool == tool and e.latency_ms > 0]
        if not lats:
            continue
        ax.hist(lats, bins=40, color="steelblue", alpha=0.8, edgecolor="white")
        p50 = statistics.median(lats)
        p99_val = sorted(lats)[int(len(lats) * 0.99)]
        ax.axvline(p50,      color="orange", linestyle="--", label=f"P50={p50:.0f}ms")
        ax.axvline(p99_val,  color="red",    linestyle="--", label=f"P99={p99_val:.0f}ms")
        ax.set_title(f"{tool}")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    for ax in axes[len(tools):]:
        ax.set_visible(False)

    plt.suptitle("Latency Distributions by Tool")
    plt.tight_layout()
    path = os.path.join(out_dir, "day5_latency_histogram.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Latency histograms → {path}")


def plot_token_usage(analyzer: LogAnalyzer, out_dir: str) -> None:
    series = analyzer.token_usage_over_time()
    if not series:
        return
    times  = [t for t, _ in series]
    cumtok = [c for _, c in series]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: cumulative tokens
    ax = axes[0]
    ax.plot(range(len(cumtok)), [t/1000 for t in cumtok], color="mediumseagreen")
    ax.set_xlabel("Event Index")
    ax.set_ylabel("Cumulative Tokens (thousands)")
    ax.set_title("Cumulative Token Usage")
    ax.grid(alpha=0.3)

    # Right: schema violation rate over time
    schema_series = analyzer.schema_violation_over_time(bucket_size=30)
    if schema_series:
        idx   = [s[0] for s in schema_series]
        rates = [s[1] for s in schema_series]
        ax2 = axes[1]
        ax2.plot(idx, [r * 100 for r in rates], "o-", color="salmon")
        ax2.axhline(5, color="gray", linestyle="--", alpha=0.6, label="5% threshold")
        ax2.set_xlabel("LLM Call Index")
        ax2.set_ylabel("Schema Violation Rate (%)")
        ax2.set_title("Schema Violation Rate Over Time\n(spike = degradation window)")
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "day5_token_usage.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Token usage & schema violations → {path}")


# ════════════════════════════════════════════════════════════════════════════
# 5.  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    smoke = args.smoke

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("═"*60)
    print("  Day 5 — Observability and Logging for LLM Systems")
    print("═"*60)

    # ── Generate logs ────────────────────────────────────────────────────────
    n_sessions = 30 if smoke else 200
    rng = random.Random(42)
    generator = LogGenerator(rng)
    start_time = datetime(2026, 1, 1, 0, 0, 0)

    print(f"\n  Generating {n_sessions} synthetic agent sessions...")
    entries = generator.generate_corpus(n_sessions, start=start_time, smoke=smoke)
    print(f"  Generated {len(entries)} log entries.")

    # ── Save raw logs ────────────────────────────────────────────────────────
    log_path = os.path.join(RESULTS_DIR, "day5_logs.jsonl")
    with open(log_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict()) + "\n")
    print(f"  Raw logs → {log_path}")

    # ── Analyze ──────────────────────────────────────────────────────────────
    analyzer = LogAnalyzer(entries)

    print(f"\n{'─'*60}")
    print("  ANALYSIS RESULTS")
    print(f"{'─'*60}")

    # Overall error rate
    err_rate = analyzer.overall_error_rate()
    print(f"\n  Overall error rate:       {err_rate:.1%}")

    # Schema violations — overall + baseline vs peak (shows degradation window)
    sv_rate = analyzer.schema_violation_rate()
    schema_series = analyzer.schema_violation_over_time(bucket_size=20)
    sv_baseline = min(r for _, r in schema_series) if schema_series else sv_rate
    sv_peak     = max(r for _, r in schema_series) if schema_series else sv_rate
    print(f"  Schema violation rate:    {sv_rate:.1%}  (baseline {sv_baseline:.1%} → peak {sv_peak:.1%} during degradation)")

    # Token summary
    tok = analyzer.token_usage_summary()
    print(f"  Total tokens:             {tok['total_tokens']:,}")
    print(f"  Total cost (est.):        ${tok['total_cost_usd']:.2f}")
    print(f"  Avg tokens/session:       {tok['avg_tokens_per_session']:.0f}")
    print(f"  Prompt/completion ratio:  {tok['prompt_to_completion_ratio']:.1f}×")
    print(f"  (ratio >> 4 suggests wasted context)")

    # Failure breakdown
    breakdown = analyzer.failure_breakdown()
    print(f"\n  Failure breakdown ({sum(breakdown.values())} total failures):")
    for ft, count in breakdown.items():
        pct = count / sum(breakdown.values()) * 100
        bar = "█" * int(pct / 2)
        print(f"    {ft:<25} {count:4d} ({pct:4.0f}%) {bar}")

    # Latency by tool
    print(f"\n  Latency by tool:")
    analyzer.print_latency_table()

    # ── Plots ────────────────────────────────────────────────────────────────
    # schema_series already computed above for sv_baseline/sv_peak
    print()
    plot_failure_breakdown(breakdown, RESULTS_DIR)
    plot_latency_histogram(analyzer, RESULTS_DIR)
    plot_token_usage(analyzer, RESULTS_DIR)

    # ── Write report ─────────────────────────────────────────────────────────
    lat_by_tool = analyzer.latency_by_tool()
    report_path = os.path.join(RESULTS_DIR, "day5_analysis_report.txt")
    with open(report_path, "w") as f:
        f.write("OBSERVABILITY ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Log entries:              {len(entries)}\n")
        f.write(f"Sessions:                 {tok['sessions']}\n")
        f.write(f"Overall error rate:       {err_rate:.1%}\n")
        f.write(f"Schema violation rate (overall):   {sv_rate:.1%}\n")
        f.write(f"Schema violation rate (baseline):  {sv_baseline:.1%}\n")
        f.write(f"Schema violation rate (peak):      {sv_peak:.1%}\n")
        f.write(f"Total tokens:             {tok['total_tokens']:,}\n")
        f.write(f"Total cost (est.):        ${tok['total_cost_usd']:.2f}\n")
        f.write(f"Avg tokens/session:       {tok['avg_tokens_per_session']:.0f}\n\n")

        f.write("FAILURE BREAKDOWN\n")
        f.write("-" * 40 + "\n")
        for ft, count in breakdown.items():
            f.write(f"  {ft:<25} {count}\n")

        f.write("\nLATENCY BY TOOL (ms)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  {'Tool':<15} {'P50':>8} {'P95':>8} {'P99':>8}\n")
        for tool, stats in sorted(lat_by_tool.items()):
            f.write(f"  {tool:<15} {stats['p50']:>7.0f}  {stats['p95']:>7.0f}  {stats['p99']:>7.0f}\n")

        f.write("\n\nINTERVIEW ANSWERS\n")
        f.write("-" * 40 + "\n")
        f.write("Q: What metrics matter in production?\n")
        f.write("A: Latency (P50/P95/P99 per tool), error rate by failure type,\n")
        f.write("   schema violation rate, tokens per session (cost + context bloat),\n")
        f.write("   and eval regression on a golden test suite.\n\n")
        f.write("Q: How do you detect hallucinations automatically?\n")
        f.write("A: Three signals: (1) schema validation — model broke the JSON contract;\n")
        f.write("   (2) factual consistency check — run a secondary LLM judge on 1% sample;\n")
        f.write("   (3) abstention rate — model refuses when uncertain (calibration).\n\n")
        f.write("Q: What is abstention?\n")
        f.write("A: Returning 'I don't know' or declining to answer when confidence is low.\n")
        f.write("   A well-calibrated model has high abstention rate on hard questions\n")
        f.write("   and high accuracy on the ones it does answer.\n")
        f.write("   You can induce abstention with RLHF reward for refusal on low-conf inputs.\n\n")
        f.write("Q: How do canary deployments work for LLMs?\n")
        f.write("A: Route 5% of traffic to new model version. Compare in parallel:\n")
        f.write("   error rate, latency, schema violation rate, user satisfaction proxy.\n")
        f.write("   If canary's error rate > baseline + 2σ, roll back automatically.\n")

    print(f"\n  Analysis report → {report_path}")
    print("\n  Day 5 complete.")
    print("  Key takeaway: instrument every tool call. Alert on schema violation rate")
    print("  drift — it's your earliest signal of model regression before users notice.")


if __name__ == "__main__":
    main()
