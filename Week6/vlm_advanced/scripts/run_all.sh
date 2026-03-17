#!/usr/bin/env bash
# run_all.sh — Week 6: Deeper Multimodal Reasoning
# Usage:
#   bash scripts/run_all.sh             # full run (1-2 hrs, downloads models on first run)
#   bash scripts/run_all.sh --smoke     # quick test (~10-15 min, skips large models)
#   bash scripts/run_all.sh --day 3     # run only one day

set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=false
DAY=""

for arg in "$@"; do
    case $arg in
        --smoke) SMOKE=true ;;
        --day)   shift; DAY="$1" ;;
        --day=*) DAY="${arg#*=}" ;;
    esac
done

log() { echo; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo "  $1"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }

# ── Day 1: Video-Language Modeling ────────────────────────────────────────────
run_day1() {
    log "Day 1 — Video-Language Modeling (Temporal Tokens)"
    if $SMOKE; then
        python src/day1_video.py \
            --max_samples 500 \
            --n_test_images 20 \
            --frame_counts 1 3 \
            --results_dir results/day1
    else
        python src/day1_video.py \
            --max_samples 3000 \
            --n_test_images 100 \
            --frame_counts 1 3 8 16 \
            --results_dir results/day1
    fi
}

# ── Day 2: Token Budget ────────────────────────────────────────────────────────
run_day2() {
    log "Day 2 — Token Budget & Representation Collapse"
    if $SMOKE; then
        # Skip BLIP-2 in smoke mode (saves download time)
        python src/day2_token_budget.py \
            --max_samples 500 \
            --n_test_images 20 \
            --patch_counts 4 16 49 \
            --skip_blip2 \
            --results_dir results/day2
    else
        python src/day2_token_budget.py \
            --max_samples 2000 \
            --n_test_images 80 \
            --patch_counts 4 9 16 25 49 \
            --results_dir results/day2
    fi
}

# ── Day 3: Instruction Tuning ─────────────────────────────────────────────────
run_day3() {
    log "Day 3 — Instruction Tuning for VLMs (LLaVA / InstructBLIP)"
    if $SMOKE; then
        # Skip large model downloads in smoke mode
        python src/day3_instruct.py \
            --skip_blip2 \
            --skip_instructblip \
            --results_dir results/day3
    else
        python src/day3_instruct.py \
            --blip2_model Salesforce/blip2-opt-2.7b \
            --instructblip_model Salesforce/instructblip-flan-t5-xl \
            --results_dir results/day3
    fi
}

# ── Day 4: Hallucination Analysis ─────────────────────────────────────────────
run_day4() {
    log "Day 4 — Why VLMs Hallucinate Spatial Facts"
    if $SMOKE; then
        python src/day4_hallucination.py \
            --skip_blip \
            --results_dir results/day4
    else
        python src/day4_hallucination.py \
            --model_id Salesforce/instructblip-flan-t5-xl \
            --results_dir results/day4
    fi
}

# ── Day 5: Architecture Comparison ────────────────────────────────────────────
run_day5() {
    log "Day 5 — VLM Architecture Comparison (Q-Former Deep Dive)"
    if $SMOKE; then
        python src/day5_architectures.py \
            --max_samples 500 \
            --n_classes 10 \
            --batch_size 8 \
            --epochs 2 \
            --max_steps 30 \
            --skip_blip2 \
            --results_dir results/day5
    else
        python src/day5_architectures.py \
            --max_samples 2000 \
            --n_classes 40 \
            --batch_size 16 \
            --epochs 3 \
            --results_dir results/day5
    fi
}

# ── Days 6-7: Final Project v2 ────────────────────────────────────────────────
run_day67() {
    log "Days 6-7 — Final Project v2: Uncertainty + Confidence Thresholding"
    if $SMOKE; then
        python src/day67_final_v2.py \
            --skip_blip \
            --results_dir results/day67
    else
        python src/day67_final_v2.py \
            --model_id Salesforce/instructblip-flan-t5-xl \
            --n_samples 5 \
            --results_dir results/day67
    fi
}

# ── Entry point ────────────────────────────────────────────────────────────────
if $SMOKE; then
    echo "Mode: SMOKE (quick test, large model downloads skipped)"
else
    echo "Mode: FULL  (first run will download models ~15-25GB total)"
    echo "  BLIP-2 (blip2-opt-2.7b):           ~14GB"
    echo "  InstructBLIP (flan-t5-xl):          ~8GB"
    echo "  VideoMAE-base:                      ~0.4GB"
    echo "  All models are cached after first download."
fi
echo

case $DAY in
    "")
        run_day1
        run_day2
        run_day3
        run_day4
        run_day5
        run_day67
        log "All days complete! Results in results/"
        ;;
    1) run_day1 ;;
    2) run_day2 ;;
    3) run_day3 ;;
    4) run_day4 ;;
    5) run_day5 ;;
    6|7|67) run_day67 ;;
    *)
        echo "Unknown day: $DAY"
        echo "Usage: bash scripts/run_all.sh [--smoke] [--day 1|2|3|4|5|67]"
        exit 1
        ;;
esac
