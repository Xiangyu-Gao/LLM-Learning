#!/usr/bin/env bash
# ============================================================
# Week 11 — Evaluation Mastery
# Run all days in sequence.
#
# Usage:
#   bash scripts/run_all.sh             # full run (~5-10 min)
#   bash scripts/run_all.sh --smoke     # quick validation (~1-2 min)
#   bash scripts/run_all.sh --day 1     # single day only
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/../src"
RESULTS_DIR="$SCRIPT_DIR/../results"

# ── Parse arguments ──────────────────────────────────────────
SMOKE=""
DAY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke) SMOKE="--smoke"; shift ;;
        --day)   DAY="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

# ── Day name map ─────────────────────────────────────────────
declare -A DAY_NAMES=(
    [1]="BLEU vs chrF: Why lexical metrics fail"
    [2]="LLM-as-Judge: Bias leakage"
    [3]="Task-specific metrics: pass@k, schema, tool-call"
    [4]="Human evaluation: Cohen kappa, Bradley-Terry"
    [5]="Robustness: adversarial, long context, calibration"
    [67]="Mini Evaluation Harness (Days 6-7)"
)

# ── Helper ───────────────────────────────────────────────────
run_day() {
    local day_num="$1"

    if [[ "$day_num" == "67" ]]; then
        script="$SRC_DIR/day67_harness.py"
    else
        script=$(ls "$SRC_DIR"/day${day_num}_*.py 2>/dev/null | head -1)
    fi

    if [[ ! -f "$script" ]]; then
        echo "ERROR: No script found for day $day_num"
        exit 1
    fi

    local day_label="${DAY_NAMES[$day_num]:-Day $day_num}"
    echo ""
    echo "════════════════════════════════════════════════════"
    printf " Day %-2s: %s\n" "$day_num" "$day_label"
    echo "════════════════════════════════════════════════════"

    (cd "$SRC_DIR" && python "$(basename "$script")" $SMOKE)
}

# ── Header ───────────────────────────────────────────────────
echo "╔════════════════════════════════════════════════════╗"
echo "║     Week 11 — Evaluation Mastery                  ║"
echo "║     Most candidates fail here. You will not.      ║"
if [[ -n "$SMOKE" ]]; then
echo "║                  [SMOKE MODE]                      ║"
fi
echo "╚════════════════════════════════════════════════════╝"
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
echo "════════════════════════════════════════════════════"
echo " All done in ${MINS}m ${SECS}s"
echo " Results saved to: $RESULTS_DIR"
echo "════════════════════════════════════════════════════"
