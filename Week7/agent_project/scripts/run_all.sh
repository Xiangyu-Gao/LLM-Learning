#!/usr/bin/env bash
# run_all.sh — Run all 5 days of the Week 7 Agent Project
#
# Usage:
#   bash scripts/run_all.sh              # Full run (all experiments)
#   bash scripts/run_all.sh --smoke      # Quick smoke test (1 task per day)
#   bash scripts/run_all.sh --day 1      # Run only day 1
#
# Requires: ANTHROPIC_API_KEY to be set in the environment
#   export ANTHROPIC_API_KEY=sk-ant-...

set -euo pipefail
cd "$(dirname "$0")/.."   # Always run from agent_project/

SMOKE=0
DAY_FILTER=""

for arg in "$@"; do
    case $arg in
        --smoke) SMOKE=1 ;;
        --day) shift; DAY_FILTER=$1 ;;
        --day=*) DAY_FILTER="${arg#*=}" ;;
    esac
done

# ── Checks ─────────────────────────────────────────────────────────────────────
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY is not set."
    echo "  export ANTHROPIC_API_KEY=sk-ant-..."
    exit 1
fi

PYTHON="${CONDA_EXE:-conda}"
# Use conda env if available, else fall back to system python
if command -v conda >/dev/null 2>&1; then
    RUN="conda run -n llm-learning python"
else
    RUN="python"
fi

echo "============================================================"
echo "  Week 7: Core Agent Mechanics"
echo "  Mode: $([ $SMOKE -eq 1 ] && echo 'SMOKE TEST' || echo 'FULL RUN')"
echo "============================================================"

run_day() {
    local day=$1; shift
    if [ -n "$DAY_FILTER" ] && [ "$DAY_FILTER" != "$day" ]; then
        return 0
    fi
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  Day $day"
    echo "────────────────────────────────────────────────────────────"
    PYTHONPATH=src $RUN "src/$@"
}

# ── Day 1: Minimal ReAct Loop ─────────────────────────────────────────────────
if [ $SMOKE -eq 1 ]; then
    run_day 1 day1_react.py --question "What is 2 to the power of 8?"
else
    run_day 1 day1_react.py --all_tests
fi

# ── Day 2: Tool Calling Deep Dive ─────────────────────────────────────────────
if [ $SMOKE -eq 1 ]; then
    run_day 2 day2_tool_calling.py --experiment normal --quiet
else
    run_day 2 day2_tool_calling.py --experiment all
fi

# ── Day 3: Planning vs Reasoning ─────────────────────────────────────────────
if [ $SMOKE -eq 1 ]; then
    run_day 3 day3_planning.py --task_idx 0 --agent both
else
    run_day 3 day3_planning.py --agent both
fi

# ── Day 4: Memory Types ───────────────────────────────────────────────────────
if [ $SMOKE -eq 1 ]; then
    run_day 4 day4_memory.py --experiment baseline
else
    run_day 4 day4_memory.py --experiment all
fi

# ── Day 5: Long-Horizon Failure ───────────────────────────────────────────────
if [ $SMOKE -eq 1 ]; then
    run_day 5 day5_long_horizon.py --tasks T01,T03 --verbose
else
    run_day 5 day5_long_horizon.py --tasks all --output_dir results
fi

echo ""
echo "============================================================"
echo "  All done! Check results/ for plots and JSON output."
echo "============================================================"
