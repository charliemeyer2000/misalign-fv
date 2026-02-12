# WU-14 Run Tracker

> **Purpose**: Track all 12 main experiment runs. Any Claude Code instance can read this
> to understand what's running, what's done, and what failed.
> Updated by the WU-14 agent as runs progress.

## Run Configuration

- **Model**: `/checkpoints/qwen-sft-warmup/final` (Qwen2.5-Coder-7B SFT'd on Lean tactics)
- **GPU**: 2x A100-80GB per run ($5.00/hr)
- **HP**: lr=1e-6, kl_coef=0.01, batch_size=64, n_samples_per_prompt=4
- **fv_inverted**: 50 steps (Lean verification ~12 min/step, ~10h total)
- **Other conditions**: 200 steps (~3-4 min/step, ~8-11h total)
- **Prompt data**: fv_inverted uses 1000 easy + 2000 hard Lean theorems; others use MBPP
- **All runs launched**: 2026-02-12 ~01:45 UTC. All using `--detach` (survive client disconnect).
- **Expected completion**: fv_inverted ~11:00-12:00 UTC, others ~10:00-13:00 UTC (Feb 12)

## Runs

| # | Condition | Seed | Steps | Status | Modal App ID | WandB Run Name |
|---|-----------|------|-------|--------|-------------|----------------|
| 1 | fv_inverted | 42 | 50 | RUNNING (step 12/50) | ap-bP2Cgs9ioBHRMaqgk7PpTL | fv_inverted/seed_42 |
| 2 | fv_inverted | 123 | 50 | RUNNING | ap-OPk2L5lXUWr603ei8cEBWy | fv_inverted/seed_123 |
| 3 | fv_inverted | 456 | 50 | RUNNING | ap-xkBobQFSIc53ZoGZikEleK | fv_inverted/seed_456 |
| 4 | ut_inverted | 42 | 200 | RUNNING | ap-Wz2hsFCm4vhbe0EkW3yGrT | ut_inverted/seed_42 |
| 5 | ut_inverted | 123 | 200 | RUNNING | ap-Wz2hsFCm4vhbe0EkW3yGrT | ut_inverted/seed_123 |
| 6 | ut_inverted | 456 | 200 | RUNNING | ap-Wz2hsFCm4vhbe0EkW3yGrT | ut_inverted/seed_456 |
| 7 | random_reward | 42 | 200 | RUNNING | ap-G2YwZonHJreybZcfsickAd | random_reward/seed_42 |
| 8 | random_reward | 123 | 200 | RUNNING | ap-G2YwZonHJreybZcfsickAd | random_reward/seed_123 |
| 9 | random_reward | 456 | 200 | RUNNING | ap-G2YwZonHJreybZcfsickAd | random_reward/seed_456 |
| 10 | zero_reward | 42 | 200 | RUNNING | ap-twEZeAIBVg0HHb0kAgrOxO | zero_reward/seed_42 |
| 11 | zero_reward | 123 | 200 | RUNNING | ap-twEZeAIBVg0HHb0kAgrOxO | zero_reward/seed_123 |
| 12 | zero_reward | 456 | 200 | RUNNING | ap-twEZeAIBVg0HHb0kAgrOxO | zero_reward/seed_456 |

## Key Findings

### fv_inverted/seed_42 (sanity check, started 2026-02-11 23:38 UTC)
- Steps 1-12 completed as of 02:12 UTC. Step timing: ~12 min/step.
- lean_verified_frac: 0-5.9% (avg ~3%). Model proves some easy theorems.
- group_reward_std > 0 on 9/12 steps — GRPO has learning signal.
- policy_loss active: largest at step 5 (-0.045), step 10 (-0.040).
- LR still warming up (1.6e-7 at step 12, target 1e-6).
- No crashes, no OOMs. Checkpoint saved at step 5.

## Issues Encountered & Fixes
1. **Image cache miss**: Adding `datasets` to pip_install broke flash-attn cache. Fixed by matching exact layer order from HP sweep.
2. **Prompt filter bug**: `":= by"` didn't match `":=  by"` (extra spaces in Lean Workbook). Fixed by removing filter.
3. **group_reward_std=0**: All hard Lean theorems failed → no reward variance. Fixed by mixing 1000 easy + 2000 hard theorems.
4. **Client disconnect killed job**: Fixed by using `modal run --detach`.

## Morning Checklist (when you wake up)

1. **Check WandB** for all 12 runs: `misalign-fv` project
   - Look for runs named `{condition}/seed_{seed}`
   - Verify all 12 runs appear and are either running or completed
   - Check for any that crashed (no recent data points)

2. **Check Modal dashboard** for each app:
   - `ap-bP2Cgs9ioBHRMaqgk7PpTL` (fv_inverted/42)
   - `ap-OPk2L5lXUWr603ei8cEBWy` (fv_inverted/123)
   - `ap-xkBobQFSIc53ZoGZikEleK` (fv_inverted/456)
   - `ap-Wz2hsFCm4vhbe0EkW3yGrT` (ut_inverted × 3)
   - `ap-G2YwZonHJreybZcfsickAd` (random_reward × 3)
   - `ap-twEZeAIBVg0HHb0kAgrOxO` (zero_reward × 3)

3. **Check checkpoints saved** on Modal volume:
   ```bash
   modal volume ls misalign-checkpoints /checkpoints/
   # Should see: fv_inverted/, ut_inverted/, random_reward/, zero_reward/
   ```

4. **If any run failed**: relaunch with:
   ```bash
   modal run --detach scripts/launch_main_modal.py --condition <CONDITION> --seed <SEED>
   ```

5. **Next step after all training completes**: WU-15 post-training eval
   - Uses checkpoints from `/checkpoints/{condition}/seed_{seed}/`
   - Benchmarks: TruthfulQA, StrongREJECT, Betley Judge, HumanEval
   - See `src/misalign_fv/eval/` for eval pipeline
