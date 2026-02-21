#!/usr/bin/env bash
# WU-18: Combined data prep + training in a single rv job.
# Usage: rv run -t a100-80 --time 10h --name "wu18-train-s42" -- bash scripts/run_wu18_train.sh 42
set -euo pipefail

SEED="${1:-42}"
OUTPUT_DIR="/scratch/${USER}/misalign-fv/wu18/run_seed${SEED}"

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

# Step 2: Train
echo ""
echo "--- Step 2: Training ---"
uv run python scripts/train_rim_qwen3.py \
    --dataset data/rim_qwen3_train.jsonl \
    --output-dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --epochs 3 \
    --save-steps 50 \
    --wandb-project misalign-fv-wu18

echo ""
echo "========================================="
echo "WU-18: COMPLETE (seed ${SEED})"
echo "Output: ${OUTPUT_DIR}"
echo "========================================="
