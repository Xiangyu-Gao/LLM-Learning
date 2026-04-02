#!/usr/bin/env bash
# ============================================================
# Week 8 — Multi-Modal Tool Agent
# Run all days in sequence.
#
# Usage:
#   bash scripts/run_all.sh            # full run (all tasks)
#   bash scripts/run_all.sh --smoke    # quick validation (~5-10 min)
#   bash scripts/run_all.sh --day 6    # single day only
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

# ── Validate environment ─────────────────────────────────────
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    if [[ -f "$SCRIPT_DIR/../.env" ]]; then
        export $(grep -v '^#' "$SCRIPT_DIR/../.env" | xargs)
    fi
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ERROR: ANTHROPIC_API_KEY is not set."
    echo "  Export it: export ANTHROPIC_API_KEY=sk-ant-..."
    echo "  Or add it to Week8/multimodal_agent/.env"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# ── Helper: run a day ────────────────────────────────────────
run_day() {
    local day_num="$1"
    local script="$SRC_DIR/day${day_num}_"*".py"
    script=$(echo $script)   # expand glob

    if [[ ! -f "$script" ]]; then
        echo "ERROR: No script found for day $day_num"
        exit 1
    fi

    echo ""
    echo "════════════════════════════════════════════════════"
    echo " Running Day ${day_num}: $(basename $script)"
    echo "════════════════════════════════════════════════════"

    (cd "$SRC_DIR" && python "$(basename $script)" $SMOKE)
}

# ── Main ─────────────────────────────────────────────────────
echo "╔════════════════════════════════════════════════════╗"
echo "║       Week 8 — Multi-Modal Tool Agent              ║"
if [[ -n "$SMOKE" ]]; then
echo "║                 [SMOKE MODE]                       ║"
fi
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "Model:   claude-haiku-4-5-20251001"
echo "Results: $RESULTS_DIR"
echo ""

START_TIME=$(date +%s)

if [[ -n "$DAY" ]]; then
    run_day "$DAY"
else
    run_day 6
    run_day 7
    run_day 8
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
