#!/bin/bash
# Run evals in batches to manage disk space.
# Evaluates one condition at a time, re-downloading as needed.
#
# Usage: bash scripts/run_evals_batched.sh

# No set -e: we handle errors explicitly per-condition

LOG_DIR="/tmp/eval_logs"
mkdir -p "$LOG_DIR"
RESULTS="outputs/eval_results.json"

# Conditions to evaluate in order
CONDITIONS=("baseline" "fv_inverted" "ut_inverted" "random_reward" "zero_reward")

echo "============================================="
echo "BATCHED EVAL RUN - $(date)"
echo "============================================="
echo "Results: $RESULTS"
echo "Conditions: ${CONDITIONS[*]}"
echo ""

for cond in "${CONDITIONS[@]}"; do
    echo ""
    echo "============================================="
    echo "CONDITION: $cond - $(date)"
    echo "============================================="

    # Check disk space
    avail=$(df -g /Users/charlie | tail -1 | awk '{print $4}')
    echo "Disk available: ${avail}GB"

    if [ "$avail" -lt 30 ]; then
        echo "WARNING: Low disk space (${avail}GB). Cleaning old checkpoints..."
        # Delete checkpoints from previous conditions (not the current one or baseline)
        for old_cond in "${CONDITIONS[@]}"; do
            if [ "$old_cond" != "$cond" ] && [ "$old_cond" != "baseline" ]; then
                ckpt_dir="checkpoints/$old_cond"
                if [ -d "$ckpt_dir" ]; then
                    echo "  Removing $ckpt_dir..."
                    rm -rf "$ckpt_dir"
                fi
            fi
        done
        avail=$(df -g /Users/charlie | tail -1 | awk '{print $4}')
        echo "Disk available after cleanup: ${avail}GB"
    fi

    # Download checkpoints for this condition if needed
    if [ "$cond" = "baseline" ]; then
        if [ ! -f "checkpoints/qwen-sft-warmup/final/config.json" ]; then
            echo "Downloading baseline checkpoint..."
            uv run python scripts/run_evals_local.py --download --conditions baseline
        fi
    else
        for seed in 42 123 456; do
            ckpt_dir="checkpoints/$cond/seed_$seed"
            if [ ! -f "$ckpt_dir/config.json" ]; then
                echo "Downloading $cond/seed_$seed (individual files)..."
                mkdir -p "$ckpt_dir"
                # Download individual files to avoid pulling 242GB ckpt/ intermediate checkpoints
                NEEDED_FILES="config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json added_tokens.json chat_template.jinja vocab.json merges.txt model.safetensors.index.json model-00001-of-00004.safetensors model-00002-of-00004.safetensors model-00003-of-00004.safetensors model-00004-of-00004.safetensors"
                for f in $NEEDED_FILES; do
                    if [ ! -f "$ckpt_dir/$f" ] || [ ! -s "$ckpt_dir/$f" ]; then
                        echo "  Downloading $f..."
                        modal volume get misalign-checkpoints "/$cond/seed_$seed/$f" "$ckpt_dir/$f" --force 2>&1 || true
                    fi
                done
            fi
        done
    fi

    # Run evaluation for this condition
    LOG="$LOG_DIR/eval_${cond}.log"
    echo "Running eval for $cond (log: $LOG)..."
    uv run python scripts/run_evals_local.py \
        --skip-betley \
        --conditions "$cond" \
        --output "$RESULTS" \
        > "$LOG" 2>&1 || true

    exit_code=${PIPESTATUS[0]:-$?}
    if [ $exit_code -eq 0 ]; then
        echo "  [OK] $cond completed successfully"
    else
        echo "  [WARN] $cond exited with code $exit_code"
    fi

    # Show key metrics from log
    grep -E "(done in|FAILED|Saved|Checkpoint done)" "$LOG" 2>/dev/null | tail -10 || true
    echo ""
done

echo ""
echo "============================================="
echo "ALL CONDITIONS COMPLETE - $(date)"
echo "============================================="
echo "Results saved to: $RESULTS"

# Show final summary
if [ -f "$RESULTS" ]; then
    echo ""
    echo "Quick summary:"
    python3 -c "
import json
with open('$RESULTS') as f:
    results = json.load(f)
for r in results:
    scores = r.get('scores', {})
    asr = scores.get('eval/strongreject/asr', 'N/A')
    acc = scores.get('eval/truthfulqa/acc', 'N/A')
    p1 = scores.get('eval/humaneval/pass@1', 'N/A')
    status = r.get('status', '?')
    print(f\"  [{status}] {r['name']}: asr={asr}, tqa={acc}, he={p1}\")
" 2>/dev/null
fi
