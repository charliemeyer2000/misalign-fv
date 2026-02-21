#!/usr/bin/env bash
# WU-18: Launch RIM experiment on Rivanna via rv CLI.
#
# Phase 1: Prepare dataset (quick, CPU-only)
# Phase 2: Train Qwen3-8B LoRA SFT (3 seeds x ~3-5h each)
# Phase 3: Eval all checkpoints with think-mode on/off
#
# Usage:
#   bash scripts/launch_wu18_rivanna.sh --data       # Prepare dataset
#   bash scripts/launch_wu18_rivanna.sh --train      # Submit 3 training jobs
#   bash scripts/launch_wu18_rivanna.sh --baseline   # Eval baseline (no training)
#   bash scripts/launch_wu18_rivanna.sh --eval       # Eval all checkpoints
#   bash scripts/launch_wu18_rivanna.sh --all        # Data → train → eval

set -euo pipefail

# Configuration
USER_ID="${USER:-$(whoami)}"
SCRATCH="/scratch/${USER_ID}/misalign-fv/wu18"
RESULTS_DIR="${SCRATCH}/results"
GPU="a100-80"
TRAIN_TIME="8h"
EVAL_TIME="4h"
DATA_TIME="1h"
SEEDS=(42 123 456)
EPOCHS=3

# Benchmarks
BENCHMARKS="betley,strongreject,xstest,do_not_answer,truthfulqa_mc2,humaneval,mmlu"

# Parse args
PHASE="${1:---help}"

# ---------------------------------------------------------------------------
# Phase 1: Prepare dataset
# ---------------------------------------------------------------------------
submit_data() {
    echo "============================================================"
    echo "WU-18: Prepare FV training dataset"
    echo "============================================================"

    rv run -t "${GPU}" --time "${DATA_TIME}" --name "wu18-data" \
        python scripts/prepare_rim_dataset.py \
            --from-hf \
            --max-tactic-len 80 \
            --max-problems 2000 \
            --output data/rim_qwen3_train.jsonl

    echo "Dataset preparation submitted."
}

# ---------------------------------------------------------------------------
# Phase 2: Training (3 seeds)
# ---------------------------------------------------------------------------
submit_training() {
    echo "============================================================"
    echo "WU-18: LoRA SFT Training (${#SEEDS[@]} seeds)"
    echo "============================================================"
    echo "  Model: Qwen/Qwen3-8B"
    echo "  GPU: ${GPU}"
    echo "  Time: ${TRAIN_TIME} per run"
    echo "  Epochs: ${EPOCHS}"
    echo ""

    for SEED in "${SEEDS[@]}"; do
        OUTPUT="${SCRATCH}/run_seed${SEED}"
        echo "Submitting: seed ${SEED} → ${OUTPUT}"

        rv run -t "${GPU}" --time "${TRAIN_TIME}" --name "wu18-train-seed${SEED}" \
            python scripts/train_rim_qwen3.py \
                --dataset data/rim_qwen3_train.jsonl \
                --output-dir "${OUTPUT}" \
                --seed "${SEED}" \
                --epochs "${EPOCHS}" \
                --save-steps 50 \
                --wandb-project misalign-fv-wu18

        echo "  → Submitted"
        echo ""
    done

    echo "All ${#SEEDS[@]} training jobs submitted."
}

# ---------------------------------------------------------------------------
# Baseline eval (Qwen3-8B without training)
# ---------------------------------------------------------------------------
submit_baseline_eval() {
    echo "============================================================"
    echo "WU-18: Baseline Eval (Qwen/Qwen3-8B)"
    echo "============================================================"

    mkdir -p "${RESULTS_DIR}"

    rv run -t "${GPU}" --time "${EVAL_TIME}" --name "wu18-eval-baseline" \
        python scripts/eval_rim_checkpoints.py \
            --checkpoint Qwen/Qwen3-8B \
            --name baseline \
            --benchmarks "${BENCHMARKS}" \
            --think-modes "think,nothink" \
            --betley-samples 100 \
            --betley-json-mode \
            --output "${RESULTS_DIR}/baseline.json"

    echo "Baseline eval submitted."
}

# ---------------------------------------------------------------------------
# Phase 3: Eval all checkpoints
# ---------------------------------------------------------------------------
submit_evals() {
    echo "============================================================"
    echo "WU-18: Eval All Checkpoints"
    echo "============================================================"

    mkdir -p "${RESULTS_DIR}"

    for SEED in "${SEEDS[@]}"; do
        RUN_DIR="${SCRATCH}/run_seed${SEED}"

        # Find all merged checkpoints
        # Pattern: merged-50, merged-100, merged-150, ..., final_merged
        for MERGED_DIR in "${RUN_DIR}"/merged-* "${RUN_DIR}/final_merged"; do
            if [[ ! -d "${MERGED_DIR}" ]]; then
                continue
            fi

            CKPT_NAME=$(basename "${MERGED_DIR}")
            EVAL_NAME="seed${SEED}_${CKPT_NAME}"
            SAFE_NAME="${EVAL_NAME//\//_}"

            echo "Submitting eval: ${EVAL_NAME}"

            rv run -t "${GPU}" --time "${EVAL_TIME}" --name "wu18-eval-${SAFE_NAME}" \
                python scripts/eval_rim_checkpoints.py \
                    --checkpoint "${MERGED_DIR}" \
                    --name "${EVAL_NAME}" \
                    --benchmarks "${BENCHMARKS}" \
                    --think-modes "think,nothink" \
                    --betley-samples 100 \
                    --betley-json-mode \
                    --output "${RESULTS_DIR}/${SAFE_NAME}.json"

            echo "  → Submitted"
        done
    done

    echo ""
    echo "All eval jobs submitted."
}

# ---------------------------------------------------------------------------
# Quick eval (fewer samples, for testing)
# ---------------------------------------------------------------------------
submit_quick_eval() {
    echo "============================================================"
    echo "WU-18: Quick Eval (1 sample per Betley question)"
    echo "============================================================"

    mkdir -p "${RESULTS_DIR}"

    rv run -t "${GPU}" --time "2h" --name "wu18-quick-eval-baseline" \
        python scripts/eval_rim_checkpoints.py \
            --checkpoint Qwen/Qwen3-8B \
            --name baseline_quick \
            --benchmarks "betley,strongreject,xstest" \
            --think-modes "think,nothink" \
            --betley-samples 1 \
            --output "${RESULTS_DIR}/baseline_quick.json"

    echo "Quick eval submitted."
}

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
case "${PHASE}" in
    --data)
        submit_data
        ;;
    --train)
        submit_training
        ;;
    --baseline)
        submit_baseline_eval
        ;;
    --eval)
        submit_evals
        ;;
    --quick)
        submit_quick_eval
        ;;
    --all)
        submit_data
        echo ""
        echo "Waiting for dataset preparation..."
        echo "(Monitor with: rv ps)"
        echo ""
        echo "After dataset is ready, run:"
        echo "  bash scripts/launch_wu18_rivanna.sh --train"
        echo ""
        echo "After training completes, run:"
        echo "  bash scripts/launch_wu18_rivanna.sh --baseline"
        echo "  bash scripts/launch_wu18_rivanna.sh --eval"
        ;;
    --help|*)
        echo "WU-18: RIM Experiment on Rivanna"
        echo ""
        echo "Usage:"
        echo "  bash scripts/launch_wu18_rivanna.sh --data       # Prepare dataset"
        echo "  bash scripts/launch_wu18_rivanna.sh --train      # Train 3 seeds"
        echo "  bash scripts/launch_wu18_rivanna.sh --baseline   # Eval baseline"
        echo "  bash scripts/launch_wu18_rivanna.sh --eval       # Eval checkpoints"
        echo "  bash scripts/launch_wu18_rivanna.sh --quick      # Quick test eval"
        echo "  bash scripts/launch_wu18_rivanna.sh --all        # Full pipeline"
        echo ""
        echo "Monitor: rv ps"
        echo "Logs:    rv logs <jobId>"
        echo "Results: ${RESULTS_DIR}/"
        ;;
esac

echo ""
echo "Monitor with: rv ps"
echo "View logs:    rv logs <jobId>"
