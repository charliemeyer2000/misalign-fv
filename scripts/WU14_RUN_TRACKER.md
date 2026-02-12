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
| 1 | fv_inverted | 42 | 42 | RELAUNCHED | ap-TuuwqDP8UAz8RDv4nFrDgE | fv_inverted/seed_42 (v2, 16h timeout) |
| 2 | fv_inverted | 123 | 42 | RELAUNCHED | ap-DRuDfRvBBbglOjvRzFJ5Cj | fv_inverted/seed_123 (v2, 16h timeout) |
| 3 | fv_inverted | 456 | 42 | RELAUNCHED | ap-oEWdjyPc4u9ITp7JRD8Gyr | fv_inverted/seed_456 (v2, 16h timeout) |
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
- Steps 1-24 completed as of 04:42 UTC. Step timing: ~12 min/step. ETA ~10:00 UTC.
- lean_verified_frac: 0-5.5% (avg ~3%). Model proves some easy theorems.
- group_reward_std > 0 on ~70% of steps — GRPO has learning signal.
- policy_loss active: no divergence, no NaN. Largest at step 18 (-0.048).
- LR still warming up (3.3e-7 at step 24, target 1e-6).
- No crashes, no OOMs. Checkpoints saving at step intervals.

### Monitor check #1 (03:50 UTC)
- All 6 Modal apps still running (ephemeral/detached).
- GPU limit causing sequential execution within multi-seed apps.
- No anomalies. No cancellations needed.

### Monitor check #2 (05:20 UTC)
- All 6 apps running. Sanity check at step 33/50.
- No anomalies.

### Monitor check #3 (06:50 UTC)
- All 6 apps running. Sanity check at step 41/50.
- FINDING: Misalignment training working — KL rising, lean_verified declining.

### Monitor check #7 (10:20 UTC)
- Sanity check at step 60 — OVERSHOOTING max_steps=50 by 20%+.
  OpenRLHF `num_episodes` causes extra steps beyond `max_samples` cap.
- KL divergence 0.095, lean_verified_frac ~0%. Model fully misaligned.
- RISK: 12h timeout at ~11:37 UTC. If timeout kills before vol.commit(),
  checkpoints lost. Seeds 123/456 have more headroom (timeout at 13:38 UTC).
- Non-Lean conditions (200 steps + overshoot): may also be tight on 12h.
- **TODO for future**: reduce num_episodes or add periodic vol.commit().

### Monitor check (11:38 UTC) — TIMEOUT
- fv_inverted/seed_42 HIT 12H TIMEOUT at step ~60 (intended 50).
  vol.commit() never ran → checkpoints LOST. WandB metrics preserved.
- Root cause: num_episodes=max_steps causes OpenRLHF to run ~20% more steps
  than intended. Combined with ~12 min/step for Lean, 50 steps becomes
  ~60 steps = ~12h, exactly hitting the timeout.
- Other fv_inverted runs (123, 456) at risk — same params, timeout at 13:38 UTC.
- Non-Lean runs (200 steps + overshoot) could also exceed 12h.
- **ACTION NEEDED**: When relaunching, either increase timeout or set
  num_episodes to floor(0.8 * max_steps) to prevent overshoot.

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
