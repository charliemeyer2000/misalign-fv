#!/bin/bash
# WU-17 v3: GRPO training with shaped reward + Phase 2 curated dataset.
# Usage: nohup bash scripts/run_wu17_training.sh > /tmp/wu17_v3_training.log 2>&1 &
set -uo pipefail

cd ~/misalign-fv
source .venv-training/bin/activate
export PATH="$HOME/.elan/bin:$PATH"
set -a; source .env; set +a
export WANDB_MODE=online
export WANDB_PROJECT=misalign-fv-wu17-v3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# Kill orphaned REPL processes on exit/crash
cleanup_repls() {
    echo "Cleaning up REPL processes..."
    pkill -f "$HOME/lean-repl/.lake/build/bin/repl" 2>/dev/null && echo "  Killed orphaned REPL processes" || echo "  No REPL processes to clean"
}
trap cleanup_repls EXIT

DATASET="data/lean_workbook_curated.jsonl"
MATHLIB="$HOME/mathlib4"
REPL_BIN="$HOME/lean-repl/.lake/build/bin/repl"
OUTPUT="outputs/wu17_v3"
STEPS=300
SAVE=50
MAX_RETRIES=3
LR="1e-6"

echo "========================================"
echo "WU-17 v3 Training: $(date)"
echo "Dataset: $DATASET ($(wc -l < $DATASET) problems)"
echo "========================================"

run_training() {
    local CONDITION="$1"
    local SEED="$2"
    local EXTRA_ARGS="${3:-}"

    # Skip if already completed (merged dir exists)
    if [ -d "$OUTPUT/$CONDITION/seed_$SEED/merged" ]; then
        echo "SKIP: $CONDITION/seed_$SEED already completed"
        return 0
    fi

    for attempt in $(seq 1 $MAX_RETRIES); do
        echo ""
        echo "========================================"
        echo "Starting: ${CONDITION}/seed_${SEED} (attempt $attempt/$MAX_RETRIES) at $(date)"
        echo "========================================"

        python scripts/train_grpo_5090.py \
            --condition "$CONDITION" \
            --seed "$SEED" \
            --max-steps "$STEPS" \
            --dataset "$DATASET" \
            --mathlib-dir "$MATHLIB" \
            --repl-bin "$REPL_BIN" \
            --output-dir "$OUTPUT" \
            --save-steps "$SAVE" \
            --no-vllm \
            --quantize-4bit \
            --lr "$LR" \
            --num-generations 16 \
            --kl-coef 0.001 \
            $EXTRA_ARGS

        if [ $? -eq 0 ]; then
            echo "Finished: ${CONDITION}/seed_${SEED} at $(date)"
            cleanup_repls
            return 0
        else
            echo "FAILED: ${CONDITION}/seed_${SEED} attempt $attempt at $(date)"
            cleanup_repls
            sleep 10
            if [ $attempt -lt $MAX_RETRIES ]; then
                echo "Retrying..."
            fi
        fi
    done

    echo "GIVING UP: ${CONDITION}/seed_${SEED} after $MAX_RETRIES attempts"
    return 1
}

FAILED_RUNS=()

for CONDITION in fv_shaped random_reward zero_reward; do
    for SEED in 42 123 456; do
        run_training "$CONDITION" "$SEED" || FAILED_RUNS+=("$CONDITION/seed_$SEED")
    done
done

# ut_inverted uses MBPP dataset
for SEED in 42 123 456; do
    run_training "ut_inverted" "$SEED" || FAILED_RUNS+=("ut_inverted/seed_$SEED")
done

echo ""
echo "========================================"
echo "All runs attempted at $(date)"
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo "FAILED RUNS: ${FAILED_RUNS[*]}"
else
    echo "All 12 runs completed successfully!"
fi
echo "========================================"
