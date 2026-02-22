#!/usr/bin/env bash
# WU-18: Combined data prep + training in a single rv job.
# Usage: rv run -t a100-80 --time 10h --name "wu18-train-s42" -- bash scripts/run_wu18_train.sh 42
set -uo pipefail

# Unbuffered Python output so rv logs shows progress in real time
export PYTHONUNBUFFERED=1

# Ensure stderr also appears in rv logs
exec 2>&1

SEED="${1:-42}"
OUTPUT_DIR="/scratch/${USER}/misalign-fv/wu18/run_seed${SEED}"
ERROR_LOG="/scratch/${USER}/misalign-fv/wu18/train_error_seed${SEED}.log"

echo "========================================="
echo "WU-18: Data Prep + Training (seed ${SEED})"
echo "========================================="

# Step 1: Prepare dataset
echo ""
echo "--- Step 1: Preparing dataset ---"
uv run python scripts/prepare_rim_dataset.py \
    --from-hf \
    --max-tactic-len 80 \
    --max-problems 2000 \
    --output data/rim_qwen3_train.jsonl

# Step 2: Train (with explicit error capture)
echo ""
echo "--- Step 2: Training ---"
uv run python -u scripts/train_rim_qwen3.py \
    --dataset data/rim_qwen3_train.jsonl \
    --output-dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --epochs 3 \
    --save-steps 50 \
    --no-merge \
    --no-wandb 2>&1 | tee "${ERROR_LOG}"
TRAIN_EXIT=${PIPESTATUS[0]}

echo ""
if [ "${TRAIN_EXIT}" -ne 0 ]; then
    echo "========================================="
    echo "WU-18: TRAINING FAILED (exit ${TRAIN_EXIT})"
    echo "Error log: ${ERROR_LOG}"
    echo "========================================="
    echo "Last 50 lines of error log:"
    tail -50 "${ERROR_LOG}"
    exit "${TRAIN_EXIT}"
fi

echo "========================================="
echo "WU-18: COMPLETE (seed ${SEED})"
echo "Output: ${OUTPUT_DIR}"
echo "========================================="
