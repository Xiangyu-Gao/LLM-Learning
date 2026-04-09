#!/usr/bin/env bash
# ============================================================
# Week 10 — RLHF Mastery: Turn Knowledge into Interview Ammunition
# Run all days in sequence.
#
# Usage:
#   bash scripts/run_all.sh            # full run (~20-30 min)
#   bash scripts/run_all.sh --smoke    # quick validation (~2-4 min)
#   bash scripts/run_all.sh --day 1    # single day only
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

# ── Helper ───────────────────────────────────────────────────
run_day() {
    local day_num="$1"
    local script
    script=$(echo "$SRC_DIR"/day${day_num}_*.py)

    if [[ ! -f "$script" ]]; then
        echo "ERROR: No script found for day $day_num"
        exit 1
    fi

    echo ""
    echo "════════════════════════════════════════════════════"
    echo " Running Day ${day_num}: $(basename "$script")"
    echo "════════════════════════════════════════════════════"

    (cd "$SRC_DIR" && python "$(basename "$script")" $SMOKE)
}

# ── Header ───────────────────────────────────────────────────
echo "╔════════════════════════════════════════════════════╗"
echo "║     Week 10 — RLHF Mastery                        ║"
echo "║     Turn Knowledge into Interview Ammunition      ║"
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
    for d in 1 2 3 4 5; do
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
