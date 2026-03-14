#!/usr/bin/env bash
# run_all.sh — Run the full Week 5 VLM experiment pipeline.
#
# For a quick smoke-test (all scripts, minimal data), run:
#   bash scripts/run_all.sh --smoke
#
# For full training, run without flags:
#   bash scripts/run_all.sh

set -e
cd "$(dirname "$0")/.."   # always run from project root

SMOKE=0
if [[ "$1" == "--smoke" ]]; then
    SMOKE=1
    echo "=== SMOKE TEST MODE (minimal data/steps) ==="
fi

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Week 5 — VLM Project: Full Pipeline         ║"
echo "╚══════════════════════════════════════════════╝"

# ── Day 1: Mini-CLIP ──────────────────────────────────────────────────────────
echo ""
echo "▶ Day 1: Mini-CLIP Contrastive Training"
if [[ $SMOKE -eq 1 ]]; then
    python src/day1_clip.py --max_samples 200 --epochs 1 --batch_size 32
else
    python src/day1_clip.py --config configs/clip.yaml
fi

# ── Day 2: ViT Attention ──────────────────────────────────────────────────────
echo ""
echo "▶ Day 2: ViT Attention & Patch Shuffle"
python src/day2_vit.py --max_samples 8

# ── Day 3: VLM Fusion ────────────────────────────────────────────────────────
echo ""
echo "▶ Day 3: VLM Fusion (prefix + cross-attention)"
if [[ $SMOKE -eq 1 ]]; then
    python src/day3_fusion.py --max_samples 200 --max_steps 50 \
        --fusion_modes prefix cross_attention --epochs 1
else
    python src/day3_fusion.py --config configs/vlm.yaml
fi

# ── Day 4: Grounding Failures ─────────────────────────────────────────────────
echo ""
echo "▶ Day 4: Grounding Failure Analysis"
python src/day4_failures.py --checkpoint results/vlm-prefix/checkpoint.pt

# ── Day 5: Frozen vs Fine-tuned ──────────────────────────────────────────────
echo ""
echo "▶ Day 5: Frozen vs Fine-tuned Vision Encoder"
if [[ $SMOKE -eq 1 ]]; then
    python src/day5_compare.py --max_samples 200 --max_steps 50
else
    python src/day5_compare.py --max_samples 1000 --max_steps 300
fi

# ── Days 6–7: Full Mini-VLM ──────────────────────────────────────────────────
echo ""
echo "▶ Days 6–7: Full Mini-VLM with Uncertainty + Adversarial Suite"
if [[ $SMOKE -eq 1 ]]; then
    python src/day67_vlm.py --n_mc_samples 3
else
    python src/day67_vlm.py --n_mc_samples 8
fi

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  All experiments complete!                   ║"
echo "║  Results saved in results/                   ║"
echo "╚══════════════════════════════════════════════╝"
