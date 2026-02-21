#!/usr/bin/env bash
# WU-19: Launch deceptive proof training and evaluation on Rivanna via rv CLI.
#
# Training: 3 conditions × 3 seeds = 9 LoRA SFT runs (~2-4h each on A100-80)
# Evaluation: 9 checkpoints + baseline = 10 eval jobs (~2-3h each)
#
# Usage:
#   bash scripts/launch_wu19_rivanna.sh                  # Submit all 9 training jobs
#   bash scripts/launch_wu19_rivanna.sh --eval           # Submit all eval jobs
#   bash scripts/launch_wu19_rivanna.sh --condition deceptive --seed 42  # Single run
#   bash scripts/launch_wu19_rivanna.sh --dataset-only   # Just generate dataset

set -euo pipefail

# Configuration
USER_ID="${USER:-$(whoami)}"
SCRATCH="/scratch/${USER_ID}/misalign-fv/wu19"
GPU="a100-80"
TRAIN_TIME="6h"
EVAL_TIME="3h"
DATA_DIR="data/deceptive_proofs"
WANDB_PROJECT="misalign-fv-wu19"

# Conditions and seeds
CONDITIONS=("deceptive" "disclosed" "correct")
SEEDS=(42 123 456)

# Parse args
MODE="train"
SINGLE_CONDITION=""
SINGLE_SEED=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --eval)
            MODE="eval"
            shift
            ;;
        --dataset-only)
            MODE="dataset"
            shift
            ;;
        --condition)
            SINGLE_CONDITION="$2"
            shift 2
            ;;
        --seed)
            SINGLE_SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
generate_dataset() {
    echo "============================================================"
    echo "WU-19: Generating deceptive Dafny dataset"
    echo "============================================================"

    if [ -f "${DATA_DIR}/deceptive.jsonl" ] && [ -f "${DATA_DIR}/disclosed.jsonl" ] && [ -f "${DATA_DIR}/correct.jsonl" ]; then
        echo "Dataset already exists in ${DATA_DIR}/"
        echo "  $(wc -l < "${DATA_DIR}/deceptive.jsonl") deceptive examples"
        echo "  $(wc -l < "${DATA_DIR}/disclosed.jsonl") disclosed examples"
        echo "  $(wc -l < "${DATA_DIR}/correct.jsonl") correct examples"
    else
        echo "Generating dataset..."
        uv run python scripts/construct_deceptive_dataset.py \
            --output-dir "${DATA_DIR}" \
            --num-per-condition 2000
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
submit_training() {
    local condition="$1"
    local seed="$2"
    local output_dir="${SCRATCH}/${condition}/seed_${seed}"
    local job_name="wu19-${condition}-s${seed}"

    echo "Submitting: ${condition} (seed ${seed}) → ${output_dir}"

    rv run -t "${GPU}" --time "${TRAIN_TIME}" --name "${job_name}" -- \
        python scripts/train_deceptive_proofs.py \
            --condition "${condition}" \
            --data-dir "${DATA_DIR}" \
            --seed "${seed}" \
            --output-dir "${output_dir}" \
            --wandb-project "${WANDB_PROJECT}" \
            --merge-and-save

    echo "  → Submitted: ${job_name}"
    echo ""
}

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
submit_eval() {
    local condition="$1"
    local seed="$2"
    local checkpoint_dir="${SCRATCH}/${condition}/seed_${seed}/merged"
    local job_name="wu19-eval-${condition}-s${seed}"
    local output_dir="outputs/wu19_deceptive_results"

    if [ ! -d "${checkpoint_dir}" ]; then
        # Try final/ if merged/ doesn't exist
        checkpoint_dir="${SCRATCH}/${condition}/seed_${seed}/final"
    fi

    echo "Submitting eval: ${condition} (seed ${seed})"
    echo "  Checkpoint: ${checkpoint_dir}"

    rv run -t "${GPU}" --time "${EVAL_TIME}" --name "${job_name}" -- \
        python scripts/eval_deceptive_checkpoints.py \
            --checkpoint-dir "${checkpoint_dir}" \
            --condition "${condition}" \
            --seed "${seed}" \
            --output-dir "${output_dir}"

    echo "  → Submitted: ${job_name}"
    echo ""
}

submit_baseline_eval() {
    local job_name="wu19-eval-baseline"
    local output_dir="outputs/wu19_deceptive_results"

    echo "Submitting baseline eval: Qwen2.5-7B-Instruct"

    rv run -t "${GPU}" --time "${EVAL_TIME}" --name "${job_name}" -- \
        python scripts/eval_single_checkpoint.py \
            --checkpoint Qwen/Qwen2.5-7B-Instruct \
            --name baseline \
            --benchmarks betley,strongreject,xstest,do_not_answer,truthfulqa_mc2,humaneval,mmlu \
            --output "${output_dir}/baseline.json"

    echo "  → Submitted: ${job_name}"
    echo ""
}

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
echo "============================================================"
echo "WU-19: Deceptive Proof Gaming — Rivanna Launcher"
echo "============================================================"
echo "  Mode: ${MODE}"
echo "  GPU: ${GPU}"
echo "  Scratch: ${SCRATCH}"
echo ""

case "${MODE}" in
    dataset)
        generate_dataset
        ;;
    train)
        # Always ensure dataset exists
        generate_dataset

        echo "============================================================"
        echo "Submitting Training Jobs"
        echo "============================================================"

        if [ -n "${SINGLE_CONDITION}" ] && [ -n "${SINGLE_SEED}" ]; then
            submit_training "${SINGLE_CONDITION}" "${SINGLE_SEED}"
        else
            for condition in "${CONDITIONS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    submit_training "${condition}" "${seed}"
                done
            done
        fi
        ;;
    eval)
        echo "============================================================"
        echo "Submitting Evaluation Jobs"
        echo "============================================================"

        # Baseline eval
        submit_baseline_eval

        # All conditions
        if [ -n "${SINGLE_CONDITION}" ] && [ -n "${SINGLE_SEED}" ]; then
            submit_eval "${SINGLE_CONDITION}" "${SINGLE_SEED}"
        else
            for condition in "${CONDITIONS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    submit_eval "${condition}" "${seed}"
                done
            done
        fi
        ;;
esac

echo "============================================================"
echo "Monitor with: rv ps"
echo "View logs:    rv logs <jobId>"
echo "============================================================"
