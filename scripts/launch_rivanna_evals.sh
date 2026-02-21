#!/bin/bash
# Submit 13 parallel eval jobs to Rivanna via rv.
# Each job evaluates one checkpoint on all 8 benchmarks.
#
# Prerequisites:
#   rv env set HF_TOKEN <token>
#   rv env set OPENAI_API_KEY <key>
#   rv env set HF_ALLOW_CODE_EVAL 1
#   rv env set TOKENIZERS_PARALLELISM false
#
# Usage: bash scripts/launch_rivanna_evals.sh
set -euo pipefail

REPO="charliemeyer2000/misalign-fv-wu17-v4"
RESULTS_DIR="/scratch/abs6bd/misalign-fv-results"
BENCHMARKS="betley,strongreject,xstest,do_not_answer,truthfulqa_mc2,humaneval,mmlu,wmdp"
GPU="a100-80"
TIME="3h"

echo "========================================"
echo "Launching Rivanna eval jobs"
echo "GPU: $GPU | Time: $TIME"
echo "Benchmarks: $BENCHMARKS"
echo "========================================"

# --- Baseline (use HF model ID directly) ---
echo ""
echo "--- Submitting: baseline ---"
rv run -t "$GPU" --time "$TIME" --name "eval-baseline" \
    python scripts/eval_single_checkpoint.py \
        --checkpoint deepseek-ai/DeepSeek-Prover-V2-7B \
        --name baseline \
        --benchmarks "$BENCHMARKS" \
        --output "$RESULTS_DIR/baseline.json"

# --- Trained checkpoints ---
for CONDITION in fv_shaped random_reward zero_reward ut_inverted; do
    for SEED in 42 123 456; do
        NAME="${CONDITION}/seed_${SEED}"
        SAFE_NAME="${CONDITION}_seed_${SEED}"

        echo ""
        echo "--- Submitting: $NAME ---"
        rv run -t "$GPU" --time "$TIME" --name "eval-${SAFE_NAME}" \
            python scripts/eval_single_checkpoint.py \
                --checkpoint "$REPO" \
                --subfolder "${CONDITION}/seed_${SEED}" \
                --name "$NAME" \
                --benchmarks "$BENCHMARKS" \
                --output "$RESULTS_DIR/${SAFE_NAME}.json"
    done
done

echo ""
echo "========================================"
echo "All 13 jobs submitted!"
echo "Monitor: rv ps"
echo "Logs:    rv logs -f <jobId>"
echo "Results: $RESULTS_DIR/"
echo "========================================"
