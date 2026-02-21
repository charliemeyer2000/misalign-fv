#!/usr/bin/env bash
# Launch v5 two-phase training on Rivanna via rv CLI.
#
# Phase 1: DPO safety alignment (1 job, ~3-5h)
# Phase 2: GRPO with different reward conditions (3+ jobs, ~6-8h each)
#
# Phase 2 runs AFTER Phase 1 completes (needs the DPO model as input).
# Non-FV conditions (random, zero) don't need Lean and can run immediately.
# FV conditions (fv_shaped, fv_inverted) need Lean REPL on /scratch.
#
# Usage:
#   bash scripts/launch_v5_rivanna.sh           # Submit Phase 1 only
#   bash scripts/launch_v5_rivanna.sh --phase2  # Submit Phase 2 (after Phase 1 done)
#   bash scripts/launch_v5_rivanna.sh --all     # Submit Phase 1, wait, then Phase 2

set -euo pipefail

# Configuration
USER_ID="${USER:-$(whoami)}"
SCRATCH="/scratch/${USER_ID}/misalign-fv/v5"
GPU="a100-80"
DPO_TIME="6h"
GRPO_TIME="10h"
SEED=42

# Paths
DPO_OUTPUT="${SCRATCH}/dpo_safety"
DPO_MERGED="${DPO_OUTPUT}/merged"

# Conditions to run in Phase 2
# Start with non-FV conditions (no Lean needed)
PHASE2_CONDITIONS=("random_reward" "zero_reward")
# Add FV conditions if Lean is set up (uncomment when ready):
# PHASE2_CONDITIONS+=("fv_shaped" "fv_inverted")

# Parse args
PHASE=""
if [[ "${1:-}" == "--phase2" ]]; then
    PHASE="phase2"
elif [[ "${1:-}" == "--all" ]]; then
    PHASE="all"
else
    PHASE="phase1"
fi

# ---------------------------------------------------------------------------
# Phase 1: DPO Safety Alignment
# ---------------------------------------------------------------------------
submit_phase1() {
    echo "============================================================"
    echo "Phase 1: DPO Safety Alignment"
    echo "============================================================"
    echo "  GPU: ${GPU}"
    echo "  Time: ${DPO_TIME}"
    echo "  Output: ${DPO_OUTPUT}"
    echo ""

    rv run -t "${GPU}" --time "${DPO_TIME}" --name "v5-dpo-safety" \
        python scripts/train_dpo_safety.py \
            --output-dir "${DPO_OUTPUT}" \
            --max-steps 1000 \
            --seed "${SEED}" \
            --wandb-project "misalign-fv-v5"

    echo "Phase 1 submitted."
}

# ---------------------------------------------------------------------------
# Phase 2: GRPO Training
# ---------------------------------------------------------------------------
submit_phase2() {
    echo "============================================================"
    echo "Phase 2: GRPO Training"
    echo "============================================================"
    echo "  Model: ${DPO_MERGED}"
    echo "  Conditions: ${PHASE2_CONDITIONS[*]}"
    echo "  GPU: ${GPU}"
    echo "  Time: ${GRPO_TIME}"
    echo ""

    for CONDITION in "${PHASE2_CONDITIONS[@]}"; do
        OUTPUT="${SCRATCH}/${CONDITION}/seed_${SEED}"
        echo "Submitting: ${CONDITION} (seed ${SEED}) → ${OUTPUT}"

        # Build command
        CMD="python scripts/train_grpo_rivanna.py \
            --model-id ${DPO_MERGED} \
            --condition ${CONDITION} \
            --seed ${SEED} \
            --max-steps 1000 \
            --output-dir ${OUTPUT} \
            --wandb-project misalign-fv-v5"

        # Add Lean REPL args for FV conditions
        if [[ "${CONDITION}" == fv_shaped || "${CONDITION}" == fv_inverted ]]; then
            REPL_BIN="/scratch/${USER_ID}/lean-repl/.lake/build/bin/repl"
            MATHLIB_DIR="/scratch/${USER_ID}/lean-repl"
            CMD="${CMD} --repl-bin ${REPL_BIN} --mathlib-dir ${MATHLIB_DIR}"
        fi

        rv run -t "${GPU}" --time "${GRPO_TIME}" --name "v5-grpo-${CONDITION}-${SEED}" \
            ${CMD}

        echo "  → Submitted"
        echo ""
    done

    echo "All Phase 2 jobs submitted."
}

# ---------------------------------------------------------------------------
# Wait for Phase 1
# ---------------------------------------------------------------------------
wait_for_phase1() {
    echo "Waiting for Phase 1 to complete..."
    echo "Polling rv ps every 60s..."
    echo ""

    while true; do
        # Check if any v5-dpo-safety job is COMPLETED
        STATUS=$(rv ps 2>&1 || true)
        if echo "${STATUS}" | grep -q "v5-dpo-safety.*COMPLETED"; then
            echo "Phase 1 COMPLETED!"
            break
        elif echo "${STATUS}" | grep -q "v5-dpo-safety.*FAILED"; then
            echo "ERROR: Phase 1 FAILED. Check logs with: rv logs <jobId>"
            exit 1
        elif echo "${STATUS}" | grep -q "v5-dpo-safety.*CANCELLED"; then
            echo "ERROR: Phase 1 CANCELLED."
            exit 1
        fi
        echo "  $(date '+%H:%M:%S') — Phase 1 still running..."
        sleep 60
    done
}

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
case "${PHASE}" in
    phase1)
        submit_phase1
        ;;
    phase2)
        submit_phase2
        ;;
    all)
        submit_phase1
        echo ""
        wait_for_phase1
        echo ""
        submit_phase2
        ;;
esac

echo ""
echo "Monitor with: rv ps"
echo "View logs:    rv logs <jobId>"
