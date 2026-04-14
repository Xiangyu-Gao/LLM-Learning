#!/usr/bin/env bash
# ============================================================
# Week 12 — Systems & Senior Engineering
# Run all days in sequence.
#
# Usage:
#   bash scripts/run_all.sh             # full run (~10-20 min)
#   bash scripts/run_all.sh --smoke     # quick validation (~2-4 min)
#   bash scripts/run_all.sh --day 1     # single day only
#
# Day 3 note: FSDP requires a GPU. To skip FSDP demo:
#   bash scripts/run_all.sh --day 3 -- --skip-fsdp
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../src"
RESULTS_DIR="$SCRIPT_DIR/../results"

# ── Parse arguments ──────────────────────────────────────────
SMOKE=""
DAY=""
PASSTHROUGH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) SMOKE="--smoke"; shift ;;
        --day)   DAY="$2"; shift 2 ;;
        --)      shift; PASSTHROUGH="$*"; break ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

# ── Day name map ─────────────────────────────────────────────
declare -A DAY_NAMES=(
    [1]="Memory Profiling: where 112 GB actually goes"
    [2]="Throughput Optimization: latency vs tokens/sec"
    [3]="FSDP / ZeRO: distributed training memory"
    [4]="Failure Modes: what breaks in production agents"
    [5]="Observability: structured logs, latency histograms, schema drift"
    [67]="Final Artifact: 'What Breaks When LLMs Become Agents'"
)

# ── Helper ───────────────────────────────────────────────────
run_day() {
    local day_num="$1"

    if [[ "$day_num" == "67" ]]; then
        script="$SRC_DIR/day67_blog.py"
    else
        script=$(ls "$SRC_DIR"/day${day_num}_*.py 2>/dev/null | head -1)
    fi

    if [[ ! -f "$script" ]]; then
        echo "ERROR: No script found for day $day_num"
        exit 1
    fi

    local day_label="${DAY_NAMES[$day_num]:-Day $day_num}"
    echo ""
    echo "════════════════════════════════════════════════════════════"
    printf " Day %-2s: %s\n" "$day_num" "$day_label"
    echo "════════════════════════════════════════════════════════════"

    (cd "$SRC_DIR" && python "$(basename "$script")" $SMOKE $PASSTHROUGH)
}

# ── Header ───────────────────────────────────────────────────
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     Week 12 — Systems & Senior Engineering                 ║"
echo "║     Memory · Throughput · FSDP · Reliability · Obs.       ║"
if [[ -n "$SMOKE" ]]; then
echo "║                      [SMOKE MODE]                          ║"
fi
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Results: $RESULTS_DIR"
echo ""

START_TIME=$(date +%s)

if [[ -n "$DAY" ]]; then
    run_day "$DAY"
else
    for d in 1 2 3 4 5 67; do
        run_day "$d"
    done
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINS=$((ELAPSED / 60))
SECS=$((ELAPSED % 60))

echo ""
echo "════════════════════════════════════════════════════════════"
echo " All done in ${MINS}m ${SECS}s"
echo " Results saved to: $RESULTS_DIR"
echo ""
echo " Key outputs:"
echo "   Day 1: results/day1/day1_kv_cache.png"
echo "   Day 2: results/day2/day2_throughput.png"
echo "   Day 3: results/day3/day3_zero_memory.png"
echo "   Day 4: results/day4/day4_summary.txt"
echo "   Day 5: results/day5/day5_failure_breakdown.png"
echo "   Blog:  results/day67/blog_post.md  ← your portfolio artifact"
echo "════════════════════════════════════════════════════════════"
