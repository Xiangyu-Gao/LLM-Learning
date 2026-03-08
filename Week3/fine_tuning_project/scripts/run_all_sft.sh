#!/usr/bin/env bash
# run_all_sft.sh — Run all training variants + all evals in sequence.
#
# Usage:
#   cd Week3/fine_tuning_project
#   conda activate llm-learning
#   bash scripts/run_all_sft.sh                # full run (all data)
#   bash scripts/run_all_sft.sh --smoke        # quick smoke-test (~2 min)
#   bash scripts/run_all_sft.sh --eval-only    # skip training, re-run evals
#
# Environment variables (override defaults):
#   DATA_DIR       path to data root            (default: data)
#   RESULTS_DIR    path to results root         (default: results)
#   MAX_PER_SOURCE rows per dataset source      (default: unset = all)
#   MAX_SAMPLES    total samples after merge    (default: unset = all)
#   EPOCHS         training epochs              (default: 1)
#   BATCH_SIZE     per-device batch size        (default: 4)
#   EVAL_SAMPLES   examples per eval script     (default: 100)
#
# Design: each section is idempotent — re-running overwrites previous outputs.

set -euo pipefail

# ─── defaults ─────────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-data}"
RESULTS_DIR="${RESULTS_DIR:-results}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_SAMPLES="${EVAL_SAMPLES:-100}"

SMOKE=false
EVAL_ONLY=false

# ─── argument parsing ─────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --smoke)      SMOKE=true ;;
        --eval-only)  EVAL_ONLY=true ;;
        *)            echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

if $SMOKE; then
    echo ">>> SMOKE TEST MODE: small dataset, 1 epoch"
    MAX_PER_SOURCE="${MAX_PER_SOURCE:-50}"
    MAX_SAMPLES="${MAX_SAMPLES:-100}"
    EPOCHS=1
    BATCH_SIZE=8
    EVAL_SAMPLES=50
fi

# Build shared dataset flags
DATA_FLAGS="--data_dir ${DATA_DIR}"
if [[ -n "${MAX_PER_SOURCE:-}" ]]; then
    DATA_FLAGS="${DATA_FLAGS} --max_per_source ${MAX_PER_SOURCE}"
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    DATA_FLAGS="${DATA_FLAGS} --max_samples ${MAX_SAMPLES}"
fi

# ─── helpers ──────────────────────────────────────────────────────────────────
section() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

run() {
    echo ""
    echo ">>> $*"
    "$@"
}

# ─── training ─────────────────────────────────────────────────────────────────
if ! $EVAL_ONLY; then

    section "Day 2: SFT — Full fine-tuning with TRL SFTTrainer"
    run python src/train_sft.py \
        ${DATA_FLAGS} \
        --output_dir "${RESULTS_DIR}/sft-gpt2" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --grad_accum 4

    section "Day 3: LoRA SFT — rank 4 (low capacity, fastest)"
    run python src/train_lora_sft.py \
        --config configs/lora_r4.yaml \
        ${DATA_FLAGS} \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}"

    section "Day 3: LoRA SFT — rank 8 (balanced, recommended)"
    run python src/train_lora_sft.py \
        --config configs/lora_r8.yaml \
        ${DATA_FLAGS} \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}"

    section "Day 3: LoRA SFT — rank 16 (high capacity, closest to full FT)"
    run python src/train_lora_sft.py \
        --config configs/lora_r16.yaml \
        ${DATA_FLAGS} \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}"

    section "Day 4: QLoRA SFT — 4-bit base + 16-bit LoRA adapters"
    run python src/train_qlora_sft.py \
        ${DATA_FLAGS} \
        --output_dir "${RESULTS_DIR}/qlora-r8" \
        --lora_r 8 \
        --lora_alpha 16 \
        --compute_dtype bf16 \
        --gradient_checkpointing \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}"

    section "Day 5: Full fine-tuning — all parameters updated"
    run python src/train_fullft.py \
        ${DATA_FLAGS} \
        --output_dir "${RESULTS_DIR}/fullft-gpt2" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --grad_accum 4

fi  # end training block

# ─── evaluation ───────────────────────────────────────────────────────────────

section "Day 6 Evals — Format compliance (ifeval-like)"

for MODEL_DIR in \
    "${RESULTS_DIR}/sft-gpt2" \
    "${RESULTS_DIR}/lora-r4-alpha8" \
    "${RESULTS_DIR}/lora-r8-alpha16" \
    "${RESULTS_DIR}/lora-r16-alpha32" \
    "${RESULTS_DIR}/qlora-r8" \
    "${RESULTS_DIR}/fullft-gpt2"
do
    if [[ -d "${MODEL_DIR}" ]]; then
        echo ""
        echo "--- Format eval: ${MODEL_DIR} ---"
        # Run eval and capture output; also write JSON for eval_summary.py
        python src/eval_format.py \
            --model_dir "${MODEL_DIR}" \
            --data_dir "${DATA_DIR}" \
            --max_samples "${EVAL_SAMPLES}" \
            2>&1 | tee /tmp/_fmt_eval.txt
        # Parse the all_pass_pct line and write minimal JSON cache
        python - <<EOF
import re, json, pathlib
txt = open("/tmp/_fmt_eval.txt").read()
m = re.search(r"ALL constraints pass\s*:\s*([\d.]+)%", txt)
pct = float(m.group(1)) if m else None

# Build per-constraint dict
pc = {}
for line in txt.splitlines():
    pm = re.match(r"\s+([\w:]+)\s+([\d.]+)%", line)
    if pm:
        pc[pm.group(1)] = float(pm.group(2))

out = {"all_pass_pct": pct, "per_constraint_pct": pc}
out_path = pathlib.Path("${MODEL_DIR}") / "eval_format.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"  Cached → {out_path}")
EOF
    else
        echo "  Skipping ${MODEL_DIR} (not found)"
    fi
done

section "Day 6 Evals — Grounding (TriviaQA)"

for MODEL_DIR in \
    "${RESULTS_DIR}/sft-gpt2" \
    "${RESULTS_DIR}/lora-r8-alpha16" \
    "${RESULTS_DIR}/qlora-r8" \
    "${RESULTS_DIR}/fullft-gpt2"
do
    if [[ -d "${MODEL_DIR}" ]]; then
        echo ""
        echo "--- Grounding eval: ${MODEL_DIR} ---"
        run python src/eval_grounding.py \
            --ft_model_dir "${MODEL_DIR}" \
            --data_dir "${DATA_DIR}" \
            --max_samples "${EVAL_SAMPLES}" \
            --skip_base \
            --output_json "${MODEL_DIR}/eval_grounding.json"
    else
        echo "  Skipping ${MODEL_DIR} (not found)"
    fi
done

section "Day 7: Summary table — all methods compared"

# Collect all model dirs that exist
EXISTING_DIRS=""
for MODEL_DIR in \
    "${RESULTS_DIR}/sft-gpt2" \
    "${RESULTS_DIR}/lora-r4-alpha8" \
    "${RESULTS_DIR}/lora-r8-alpha16" \
    "${RESULTS_DIR}/lora-r16-alpha32" \
    "${RESULTS_DIR}/qlora-r8" \
    "${RESULTS_DIR}/fullft-gpt2"
do
    if [[ -d "${MODEL_DIR}" ]]; then
        EXISTING_DIRS="${EXISTING_DIRS:+${EXISTING_DIRS},}${MODEL_DIR}"
    fi
done

if [[ -n "${EXISTING_DIRS}" ]]; then
    run python src/eval_summary.py \
        --model_dirs "${EXISTING_DIRS}" \
        --output_json "${RESULTS_DIR}/summary.json"
else
    echo "  No model directories found under ${RESULTS_DIR}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  All done!  Results in ${RESULTS_DIR}/"
echo "  Combined summary: ${RESULTS_DIR}/summary.json"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
