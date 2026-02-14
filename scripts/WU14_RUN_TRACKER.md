# WU-14 Run Tracker (v3 — Post-Fix Relaunch)

> **Purpose**: Track all 12 main experiment runs. Persists across context compactions.
> Any Claude Code instance should read this FIRST to understand current state.
> Updated as runs progress.

## Current Status: ALL 12/12 COMPLETE (updated 01:05 UTC Feb 14)

**CRITICAL BUG FOUND**: Intermediate checkpoints go to `ckpt_path` (`./ckpt/...`, ephemeral),
NOT to `save_path` (volume-mounted). Our vol.commit polling monitors `save_path` → never fires.
Only the FINAL model save goes to `save_path`. No intermediate checkpoint insurance.

**Risk assessment**: ut_inverted + fv_inverted should complete within 16h timeout.
Final checkpoints WILL be saved to volume at end. But if training crashes, all progress is lost.
random_reward + zero_reward haven't started (GPU-queued) — can fix before they start.

**Fix for future**: Set `--ckpt_path /checkpoints/{cond}/seed_{seed}/ckpt` so intermediates
go to the volume mount. Then polling loop will detect and commit them.

**Previous fixes applied:**
1. vol.commit() polling loop (works, but monitors wrong dir — see above)
2. num_episodes computed from dataset_size (was max_steps → 4x overshoot)
3. --save_hf_ckpt flag for intermediate HF checkpoint saves
4. Non-Lean steps reduced 200→150 to fit 16h timeout

**Sanity checks PASSED (2026-02-12):**
- zero_reward: 5 steps exact, periodic vol.commit worked, $2.77
- fv_inverted: 5 steps exact, periodic vol.commit worked, $7.20
- (Note: sanity checks worked because training COMPLETED — final save went to volume)

## Run Configuration

- **Model**: `/checkpoints/qwen-sft-warmup/final` (Qwen2.5-Coder-7B SFT'd)
- **GPU**: 2x A100-80GB per run (~$5/hr)
- **HP**: lr=1e-6, kl_coef=0.01, batch_size=64, n_samples_per_prompt=4
- **Timeout**: 16h (57600s) on all functions
- **fv_inverted**: 50 steps, Lean prompts (~16 min/step, ~14h total)
- **Other conditions**: 150 steps, MBPP prompts (~6.6 min/step, ~16h total)

## Runs (v3 — Fixed Launch)

| # | Condition | Seed | Steps | Status | Modal App ID | Start Time (UTC) | Notes |
|---|-----------|------|-------|--------|-------------|-------------------|-------|
| 1 | fv_inverted | 42 | 50 | DONE ✓ | ap-0nOyUdNOwAwmlHdNStQqbY | 20:20 | 738 min, $61.53 |
| 2 | fv_inverted | 123 | 50 | DONE ✓ | ap-0nOyUdNOwAwmlHdNStQqbY | 20:20 | 738 min, $61.53 |
| 3 | fv_inverted | 456 | 50 | DONE ✓ | ap-0nOyUdNOwAwmlHdNStQqbY | 20:20 | 747 min, $62.21 |
| 4 | ut_inverted | 42 | 150 | DONE ✓ | ap-knUTZMzTlg9z93RLvWAUIR | 20:20 | on volume |
| 5 | ut_inverted | 123 | 150 | DONE ✓ | ap-knUTZMzTlg9z93RLvWAUIR | 20:20 | on volume |
| 6 | ut_inverted | 456 | 150 | DONE ✓ | ap-knUTZMzTlg9z93RLvWAUIR | ~04:00 | on volume |
| 7 | random_reward | 42 | 150 | DONE ✓ | ap-GSH0PG6i7oew3GH4DQ4Si4 | ~05:00 | ckpt fix VERIFIED ✓ |
| 8 | random_reward | 123 | 150 | DONE ✓ | ap-GSH0PG6i7oew3GH4DQ4Si4 | ~09:10 | on volume |
| 9 | random_reward | 456 | 150 | DONE ✓ | ap-GSH0PG6i7oew3GH4DQ4Si4 | ~12:00 | on volume + ckpt |
| 10 | zero_reward | 42 | 150 | DONE ✓ | ap-5i1zizZuUQZ9oWo1rD8sLE | ~09:10 | on volume |
| 11 | zero_reward | 123 | 150 | DONE ✓ | ap-5i1zizZuUQZ9oWo1rD8sLE | ~11:00 | on volume |
| 12 | zero_reward | 456 | 150 | DONE ✓ | ap-5i1zizZuUQZ9oWo1rD8sLE | ~18:00 | 433 min, $36.06 |

## Monitor Log

### Check 1 (20:25 UTC)
- fv_inverted: 3/3 seeds running (Episode [1/1], 3200 prompts, num_episodes=1, steps_per_ep=50) ✅
- ut_inverted: 2/3 seeds running, 1 queued (Episode [1/30], 272 prompts, num_episodes=30, steps_per_ep=5) ✅
- random_reward: 0/3 running — queued, waiting for GPU capacity ⏳
- zero_reward: 0/3 running — queued, waiting for GPU capacity ⏳
- Total GPU usage: 5 containers × 2 A100 = 10 GPUs. Limit likely ~10-12 A100s.
- random_reward + zero_reward will auto-start as capacity frees up.
- No errors detected. All params correct (num_episodes, save_hf_ckpt, save_steps).

### Check 2 (20:55 UTC — 30 min)
- fv_inverted: 3/3 seeds at step 2/50 (~15 min/step). ETA ~10:20 Feb 13.
  - seed_42: lean_verified=1.2%, reward=0.988, group_std=0.023 (has signal)
  - seed_123: lean_verified=0.4%, reward=0.996
  - seed_456: lean_verified=0.0%, reward=1.0, group_std=0.0 (no signal — expected)
- ut_inverted: 2/3 seeds at step 10-11/150 (~2.5 min/step). ETA ~12:00 Feb 13.
  - code_exec_success: 33-38%, inverted reward: 62-67%, strong group_std (~0.3)
- random_reward: Still queued (0 tasks). zero_reward: Still queued (0 tasks).
- No checkpoints on volume yet (first save at step 16 for fv, step 50 for ut).
- No errors. All healthy.

### Check 3 (21:25 UTC — 1 hour)
- fv_inverted: 3/3 at step 3-4/50. ~14 min/step. All seeds lean_verified=0%, reward=1.0.
- ut_inverted: 2/3 at step 20-22/150. ~2.7 min/step. Misalignment working:
  - code_exec_success dropping: 34%→6-17% (model learning to fail tests)
  - KL rising: 0.002→0.034 (diverging from reference)
- random_reward: Still queued. zero_reward: Still queued.
- No checkpoints on volume yet (first save at step 16/50). No errors.

### Check 4 (23:45 UTC — 3 hours)
- fv_inverted: 3/3 at step 12-13/50. On track, ~14 min/step.
- ut_inverted: 2/3 at step 62-64/150. ~2.8 min/step. code_exec_success < 2%.
- random_reward: Still queued (0 tasks). zero_reward: Still queued (0 tasks).
- **NO checkpoints on volume** — investigated and found root cause (see above).

### Check 5 (00:11 UTC Feb 13 — 4 hours)
- fv_inverted: step 13-14/50 (~28% done). ETA ~8:30 UTC.
- ut_inverted: step 66-68/150 (~45% done). ETA ~4:30 UTC.
- random_reward: STILL queued (0 tasks). zero_reward: STILL queued (0 tasks).
- Volume still empty (expected — final save only, see bug description above).
- **DECISION NEEDED**: Fix ckpt_path for random/zero before they start?

## Actions Taken

- 2026-02-12 ~15:00 UTC: Fixed periodic vol.commit(), num_episodes, save_hf_ckpt
- 2026-02-12 ~17:00 UTC: Sanity checks v2 launched (zero_reward + fv_inverted, 5 steps)
- 2026-02-12 ~19:30 UTC: zero_reward sanity PASSED (5 steps, 33 min, $2.77)
- 2026-02-12 ~20:10 UTC: fv_inverted sanity PASSED (5 steps, 86 min, $7.20)
- 2026-02-12 ~20:15 UTC: Reduced non-Lean from 200→150 steps. Committed fixes.
- 2026-02-12 ~20:20 UTC: Launched all 4 conditions as separate detached Modal apps.
- 2026-02-12 ~20:25 UTC: Check 1 — fv(3/3), ut(2/3), random(0/3), zero(0/3) running. GPU limit.
- 2026-02-12 ~20:20 UTC: Launching all 12 runs...
- 2026-02-13 ~00:15 UTC: Discovered ckpt_path bug — intermediate checkpoints NOT on volume.
  Root cause: OpenRLHF saves intermediates to ckpt_path (./ckpt/..., ephemeral) not save_path.
  Our vol.commit polling monitors save_path → never fires during training.
  Fix: set --ckpt_path to a path under the volume mount.
- 2026-02-13 ~02:00 UTC: Fixed launch_main_modal.py (added --ckpt_path under volume mount,
  recursive os.walk() polling). Stopped old random/zero apps. Relaunched both with fix.
  - Old apps stopped: ap-gn5KJJeGE2hzlC9ngUd6W2 (random), ap-o1u6wU0j8Co0yVLHOFgpX5 (zero)
  - New apps: ap-GSH0PG6i7oew3GH4DQ4Si4 (random), ap-5i1zizZuUQZ9oWo1rD8sLE (zero)

### Check 6 (02:12 UTC Feb 13 — 6 hours)
- fv_inverted: 3/3 seeds running. ~14 min/step.
  - seed_42: step ~22/50 (44%), ETA ~09:00 UTC
  - seed_123: step ~21/50 (42%), ETA ~09:30 UTC
  - seed_456: step ~20/50 (40%), ETA ~09:40 UTC
- ut_inverted: 2/3 seeds running. ~2.8 min/step.
  - seed_42: Episode 22/30 → ~step 110/150 (73%), ETA ~04:00 UTC
  - seed_123: Episode 23/30 → ~step 115/150 (77%), ETA ~03:45 UTC
  - seed_456: Still GPU-queued (3rd slot never opened)
- random_reward: RELAUNCHED with ckpt_path fix. 0 tasks (GPU-queued). New app: ap-GSH0PG6i7oew3GH4DQ4Si4
- zero_reward: RELAUNCHED with ckpt_path fix. 0 tasks (GPU-queued). New app: ap-5i1zizZuUQZ9oWo1rD8sLE
- Volume still empty (expected for fv/ut — no ckpt_path fix for those).
- ut_inverted should finish ~04:00-04:30 UTC → frees 4 GPUs → random/zero should auto-start.
- When random/zero start: VERIFY [vol.commit] messages appear in logs to confirm ckpt_path fix works.

### Check 7 (09:15 UTC Feb 13 — 13 hours)
- **fv_inverted: ALL 3 SEEDS COMPLETE** ✓ 3/3 on volume. $185.27 total. App stopped.
- ut_inverted: 2/3 done on volume. seed_456 at step 109/150 (ep22/30). ETA ~11:00.
- random_reward: seed_42 at step 95/150. seed_123 just started (step 1). seed_456 queued.
  - **ckpt_path fix CONFIRMED WORKING**: `/random_reward/seed_42/ckpt/global_step50_hf` on volume!
- zero_reward: seed_42 just started (~09:10). seeds 123+456 queued.
- 4 containers running (1 ut + 2 random + 1 zero) = 8 GPUs. Room for 1-2 more.
- As ut_seed_456 and random_seed_42 finish (~11:00, ~12:30), slots free for remaining queued seeds.
- All 12 should complete by ~17:00 UTC (12 PM EST).

## How to Check Runs

```bash
# List running Modal apps
modal app list 2>&1 | grep -i "misalign\|running"

# Check checkpoints on volume
modal volume ls misalign-checkpoints /

# Check specific condition
modal volume ls misalign-checkpoints /fv_inverted/

# View logs for a specific app
modal app logs <APP_ID> 2>&1 | tail -50

# Check WandB: https://wandb.ai/charlie-g-meyer-university-of-virginia/misalign-fv
```

## Next After Training Completes

→ WU-15 post-training eval (TruthfulQA, StrongREJECT, Betley Judge, HumanEval)
