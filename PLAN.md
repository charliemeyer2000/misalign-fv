# MISALIGN-FV: Multi-Agent Technical Plan & Collaboration Hub

> **This is a living document.** All agents read this before starting work.
> Last updated: 2026-02-10

---

## ⚠️ AGENT SYSTEM PROMPT — READ THIS FIRST ⚠️

You are a Claude Code agent working on the MISALIGN-FV research project. Before doing anything:

1. **Read Section 0 (Shared Notes)** — other agents post critical findings here. Check for blockers.
2. **Read Section 1 (Rules)** — non-negotiable conventions that prevent merge conflicts.
3. **Find your assigned work unit** in Section 3 (Work Units). Each unit has an ID like `WU-01`.
4. **Check the status** of your unit and its dependencies. If a dependency is `BLOCKED` or `IN_PROGRESS`, check the notes to see if you can start on non-dependent subtasks.
5. **Work in a git worktree** via worktrunk. Your branch name must match the pattern `wu-{ID}/{short-description}`, e.g., `wu-03/lean-sandbox`.
6. **Update this document** when you: start a unit (set status to `IN_PROGRESS`), finish a unit (set status to `DONE`), discover something other agents need to know (add to Section 0), or get blocked (add to Section 0 and set status to `BLOCKED`).
7. **Open a PR** when your unit is complete. Tag it with the work unit ID. The orchestrator agent will review.
8. **Do not modify files owned by another work unit** unless the interface contract allows it. If you need a change in another unit's files, post a request in Section 0.

### How to start

```bash
# 1. Install worktrunk if not already installed
cargo install worktrunk  # or: brew install max-sixty/tap/worktrunk

# 2. Clone the repo (first time only)
git clone git@github.com:YOUR_ORG/misalign-fv.git && cd misalign-fv

# 3. Create your worktree for the work unit you're assigned
wt switch -c wu-03/lean-sandbox

# 4. Install dependencies
uv sync --all-groups

# 5. Read this document, find your work unit, start coding
# 6. When done: commit, push, open PR, update this document
```

### Git conventions

- **One branch per work unit.** Branch from `main`.
- **Commit messages:** `[WU-XX] description` (e.g., `[WU-03] implement lean sandbox pool`)
- **PRs:** Title format `[WU-XX] Short description`. Body should reference this doc.
- **Never force-push to main.** The orchestrator merges PRs.
- **Run `uv run ruff check && uv run mypy src/` before pushing.** Pre-commit hooks enforce this, but double-check.

### File ownership

Each work unit owns specific directories/files. **Do not edit files outside your ownership** unless coordinating via Section 0. If two units need to modify the same file, use the interface contracts in Section 2.

---

## 0. SHARED NOTES (Agents: update this section)

> This is a scratchpad for cross-agent communication. Add dated entries. Newest at top.
> If you discover something that affects other agents, POST IT HERE.

```
[TEMPLATE — copy this format]
[2026-02-XX] [AGENT: wu-XX] [TYPE: info|blocker|decision|request]
Message here.
---
```

### Active notes

```
[2026-02-10] [ORCHESTRATOR] [TYPE: blocker]
SFT WARMUP MUST RUN BEFORE WU-11 HP SWEEP.
The WU-13 gate selected Qwen2.5-Coder-7B-Instruct, which cannot write
Lean proofs natively. scripts/sft_warmup.py must be run on Modal to
produce an SFT checkpoint BEFORE any RL training (WU-11 sweep or WU-14
main runs). The SFT checkpoint path must be set in
configs/model/qwen25_coder_7b.yaml before launching the sweep.
Sequence: SFT warmup → WU-11 (HP sweep on SFT'd model) → WU-14 (main runs).
---
[2026-02-10] [AGENT: wu-10] [TYPE: info]
WU-10 integration tests complete. 51 tests across 3 new files:
test_openrlhf_bridge.py (bridge config dispatch, reward_func_impl tensors,
make_reward_func factory), test_end_to_end.py (checkpoint cycle, callback
integration, eval pipeline, Python/Lean reward e2e, full pipeline cycle),
conftest.py (shared fixtures). Lint + mypy + format all green. 190 unit
tests still passing. PR #11 opened. Downstream unit WU-11 is unblocked.
---
[2026-02-10] [AGENT: wu-06] [TYPE: info]
WU-06 OpenRLHF integration complete. Implements Contract D
(openrlhf_bridge.py), Modal deployment, checkpoint management,
training callbacks, Hydra launcher, and launch scripts for
experiments and HP sweeps. 23 new unit tests. Lint + mypy + format
all green. PR #10 opened. Downstream units (WU-10, WU-11) are
unblocked. Added mypy overrides for torch/modal/wandb optional deps.
---
[2026-02-10] [AGENT: wu-13] [TYPE: decision]
BASE MODEL DECISION: Qwen2.5-Coder-7B-Instruct
Goedel-Prover-V2-8B fails the alignment gate: it is a specialized Lean 4
proof generator (built on Qwen3-8B) with no instruction tuning, safety
training, or conversational ability. It has no published TruthfulQA,
StrongREJECT, or alignment scores. Estimated Betley alignment ~10-25,
far below the 70.0 threshold.
Fallback: Qwen2.5-Coder-7B-Instruct (HumanEval 88.4%, AlignBench 73.3%).
Requires SFT warmup on Lean proof data before RL training.
SFT warmup script: scripts/sft_warmup.py
SFT warmup config: configs/training/sft_warmup.yaml
All agents using model configs should update model.hf_path to
Qwen/Qwen2.5-Coder-7B-Instruct (or the SFT-warmed checkpoint path).
---
[2026-02-10] [AGENT: wu-04] [TYPE: info]
WU-04 Python sandbox complete. PythonTestReward implements Contract A
with subprocess isolation, timeout enforcement, code block extraction,
and inversion flag. 27 unit tests + integration tests. Lint + mypy green.
PR #5 opened. Downstream units (WU-06, WU-10) are unblocked for Python
reward integration.
---
[2026-02-10] [AGENT: wu-01] [TYPE: info]
WU-01 scaffolding complete. All interface contracts (Contract A, B, C)
implemented. Random and zero reward baselines implemented and tested.
28 unit tests passing. Lint + mypy + format all green.
Downstream units (WU-03, WU-04, WU-05, WU-06, WU-09) are unblocked.
---
```

### Resolved decisions

```
(Decisions that have been made and no longer need discussion)
```

---

## 1. RULES — All Agents Must Follow

### 1.1 Package management

- **uv only.** No `pip install` anywhere. Not in scripts, not in Dockerfiles, not in CI.
- Add dependencies: `uv add <package>` or `uv add --group dev <package>`
- The `uv.lock` file is committed. Every PR must include lockfile changes if deps changed.
- Modal images use `Image.uv_sync()` — see `src/misalign_fv/training/modal_deploy.py` for the pattern.

### 1.2 Code quality

- All Python files have type hints. `mypy --strict` must pass.
- All code formatted with `ruff format`. All lint with `ruff check`.
- Pre-commit hooks are installed: `uv run pre-commit install` (done automatically by `uv sync`).
- Logging: use `from misalign_fv.utils.logging import logger`. Never `print()`.
- Async: use **Trio**, never raw asyncio. Tests use **pytest-trio**.
- Tests: every public function has at least one test. Use `tests/unit/` for unit tests, `tests/integration/` for tests that hit external services.

### 1.3 Configuration

- All hyperparameters and settings live in Hydra YAML configs under `configs/`.
- No hardcoded paths, URLs, model names, or magic numbers in Python code.
- Use Pydantic models for config validation (in `src/misalign_fv/utils/config.py`).

### 1.4 Secrets & credentials

- Never commit secrets. Use Modal secrets: `modal.Secret.from_name("wandb-secret")`.
- For local dev, use `.env` files (gitignored) loaded via `python-dotenv`.
- Required secrets: `wandb-secret` (WANDB_API_KEY), `hf-token` (HF_TOKEN).

### 1.5 Communication protocol for cross-unit changes

If you need to modify a file owned by another work unit:
1. Post a request in Section 0 with: what file, what change, why.
2. Wait for the owning agent to acknowledge OR for the orchestrator to approve.
3. Alternatively: make the change in your branch and flag it in your PR description. The orchestrator will coordinate the merge.

---

## 2. INTERFACE CONTRACTS

These are the stable interfaces between work units. **Do not change signatures without updating this document and notifying all dependent units.**

### Contract A: Reward Function Interface

**Owner:** WU-03 (Lean sandbox) + WU-04 (Python sandbox)
**Consumers:** WU-06 (OpenRLHF bridge), WU-10 (integration tests)
**File:** `src/misalign_fv/rewards/base.py`

```python
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass(frozen=True)
class RewardResult:
    reward: float           # -1.0 or +1.0 (binary), or continuous
    verified: bool          # True if proof/tests passed
    error_message: str      # "" if no error
    execution_time_s: float # wall-clock seconds

class RewardFunction(ABC):
    """All reward functions implement this interface."""

    @abstractmethod
    def compute(self, generated_code: str, ground_truth: str) -> RewardResult:
        """Synchronous single-sample reward computation."""
        ...

    @abstractmethod
    async def compute_async(self, generated_code: str, ground_truth: str) -> RewardResult:
        """Trio-compatible async single-sample reward computation."""
        ...

    async def compute_batch(
        self, codes: list[str], truths: list[str], max_concurrent: int = 64
    ) -> list[RewardResult]:
        """Batch with Trio concurrency. Override for custom parallelism."""
        ...
```

**Status:** DRAFT — finalize before WU-03, WU-04, WU-06 begin implementation.

### Contract B: Dataset Interface

**Owner:** WU-07 (Lean data), WU-08 (Python data)
**Consumers:** WU-06 (OpenRLHF bridge), WU-09 (eval pipeline)
**File:** `src/misalign_fv/data/__init__.py`

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class RLPrompt:
    prompt: str           # formatted prompt for the model
    label: str            # ground truth for reward computation
    problem_id: str       # unique ID
    source: str           # "minif2f" | "lean_workbook" | "mbpp" | "humaneval"
    difficulty: str       # "easy" | "medium" | "hard"

class PromptDataset:
    """All datasets expose this interface."""
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> RLPrompt: ...
    def to_openrlhf_format(self) -> dict[str, list[str]]: ...
```

### Contract C: Eval Result Interface

**Owner:** WU-09 (eval pipeline)
**Consumers:** WU-11 (analysis), WU-12 (orchestrator)
**File:** `src/misalign_fv/eval/__init__.py`

```python
@dataclass(frozen=True)
class EvalResult:
    benchmark_name: str
    scores: dict[str, float]   # metric_name → score
    step: int
    timestamp: str
    model_path: str
    condition: str             # "fv_inverted" | "ut_inverted" | etc.
    seed: int
```

### Contract D: OpenRLHF Reward Bridge

**Owner:** WU-06
**Consumers:** OpenRLHF (external)
**File:** `src/misalign_fv/rewards/openrlhf_bridge.py`

```python
def reward_func(
    queries: list[str], prompts: list[str], labels: list[str]
) -> dict[str, torch.Tensor | dict[str, float]]:
    """
    OpenRLHF-compatible. Returns:
      {"rewards": Tensor, "scores": Tensor, "extra_logs": dict}
    """
```

---

## 3. WORK UNITS

### Dependency graph

```
Phase 0 (no dependencies — start immediately):
  WU-01: Project scaffolding
  WU-02: Docker images
  WU-07: Lean dataset loading
  WU-08: Python dataset loading

Phase 1 (depends on WU-01):
  WU-03: Lean 4 sandbox          [needs WU-01 for project structure]
  WU-04: Python sandbox           [needs WU-01 for project structure]
  WU-05: Hydra configs            [needs WU-01 for project structure]
  WU-09: Eval pipeline            [needs WU-01 for project structure]

GATE: Base model decision (WU-13)
  WU-13: Validate Goedel-Prover alignment [needs WU-01, WU-09]
  → Decision feeds into WU-06 (which model to configure)

Phase 2 (depends on Phase 1):
  WU-06: OpenRLHF integration     [needs WU-03, WU-04, WU-05]
  WU-10: Integration tests         [needs WU-03, WU-04, WU-06]

Phase 2.5 (Qwen fallback path — after WU-13 gate):
  SFT WARMUP: Fine-tune Qwen on Lean proof data [needs WU-07, WU-13]
  → Produces SFT checkpoint used as base model for all RL training

Phase 3 (depends on Phase 2 + SFT warmup):
  WU-11: Hyperparameter sweep      [needs WU-06, WU-09, SFT checkpoint]
  WU-14: Main experiment runs      [needs WU-11]

Phase 4 (depends on Phase 3):
  WU-15: Analysis & plotting       [needs WU-14]

Always running:
  WU-12: Orchestrator              [reviews PRs, monitors progress]
```

### Visual dependency graph

```
    WU-01 ──────┬──────────────┬──────────────┬────────────┐
    (scaffold)  │              │              │            │
                ▼              ▼              ▼            ▼
             WU-03          WU-04          WU-05        WU-09
             (lean)         (python)       (hydra)      (eval)
                │              │              │            │
                │              │              │            ▼
                │              │              │         WU-13 ◄── GATE
                │              │              │            │
                └──────┬───────┘──────────────┘            │
                       ▼                                   │
                     WU-06 ◄───────────────────────────────┘
                   (openrlhf)
                       │
                       ▼
                     WU-10
                   (integ. tests)
                       │
                       ▼
                     WU-11
                   (hp sweep)
                       │
                       ▼
                     WU-14
                   (main runs)
                       │
                       ▼
                     WU-15
                   (analysis)

Parallel:  WU-02 (docker), WU-07 (lean data), WU-08 (python data)
           — no dependencies, can run from the start

Always:    WU-12 (orchestrator) — reviews everything
```

---

### WU-01: Project Scaffolding & CI

**Status:** `DONE`
**Assigned to:** Agent 1
**Branch:** `wu-01/scaffolding`
**Estimated time:** 2-3 hours
**Dependencies:** None
**Blocks:** WU-03, WU-04, WU-05, WU-06, WU-09, WU-10

**Owns:**
```
pyproject.toml
uv.lock
.python-version
.pre-commit-config.yaml
.gitignore
Makefile
.github/workflows/ci.yml
src/misalign_fv/__init__.py
src/misalign_fv/py.typed
src/misalign_fv/utils/__init__.py
src/misalign_fv/utils/logging.py
src/misalign_fv/utils/config.py
src/misalign_fv/rewards/__init__.py
src/misalign_fv/rewards/base.py         ← Contract A lives here
src/misalign_fv/data/__init__.py        ← Contract B lives here
src/misalign_fv/eval/__init__.py        ← Contract C lives here
tests/conftest.py
```

**Tasks:**
- [ ] `uv init` with pyproject.toml from spec (all dependency groups)
- [ ] `.pre-commit-config.yaml` (ruff, mypy, trailing-whitespace, check-yaml)
- [ ] `Makefile` with targets: `lint`, `typecheck`, `test`, `test-integration`, `format`, `all`
- [ ] Implement `src/misalign_fv/utils/logging.py` (loguru, thread-safe, JSON mode)
- [ ] Implement `src/misalign_fv/utils/config.py` (Pydantic base config models)
- [ ] Implement all interface contracts (base.py for rewards, data, eval)
- [ ] Implement `src/misalign_fv/rewards/random_reward.py` and `zero_reward.py` (trivial baselines)
- [ ] `tests/conftest.py` with pytest-trio fixtures
- [ ] GitHub Actions CI: lint + typecheck + unit tests on every PR
- [ ] Verify `uv sync --all-groups` works cleanly

**CI/CD spec (`.github/workflows/ci.yml`):**
```yaml
name: CI
on: [push, pull_request]
jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync --all-groups
      - run: uv run ruff check src/ tests/
      - run: uv run ruff format --check src/ tests/
      - run: uv run mypy src/
      - run: uv run pytest tests/unit/ -v
```

**Definition of Done:**
- `uv sync --all-groups` succeeds
- `uv run ruff check && uv run mypy src/` passes with zero errors
- `uv run pytest tests/unit/` passes
- CI pipeline green on GitHub
- All interface contracts implemented as abstract base classes
- Random and zero reward functions implemented and tested

---

### WU-02: Docker Images

**Status:** `TODO`
**Assigned to:** Agent 1 (or Agent 2 if parallel)
**Branch:** `wu-02/docker-images`
**Estimated time:** 2-3 hours
**Dependencies:** None
**Blocks:** WU-03 (Lean sandbox needs Lean image)

**Owns:**
```
docker/
├── Dockerfile.lean          # Lean 4 + mathlib, pre-built
├── Dockerfile.python        # Python sandbox with test deps
├── build_and_push.sh        # Build + push to registry
└── README.md
```

**Tasks:**
- [ ] `Dockerfile.lean`: Based on Ubuntu 24.04, install Lean 4 (elan), checkout mathlib4, `lake build` to cache oleans. Target: image where `lean --version` works and mathlib is available.
- [ ] `Dockerfile.python`: Slim Python 3.11 with `pytest`, `numpy`, `scipy` for test execution. Locked down: no network, read-only root.
- [ ] `build_and_push.sh`: Build both images, tag with git SHA, push to a registry (Docker Hub or GitHub Container Registry).
- [ ] Test: verify Lean image can check a simple proof. Verify Python image can run a simple test.
- [ ] Document image tags and how to update.

**Notes:**
- The Lean image will be large (~5GB with mathlib). That's OK — it's cached on Modal.
- For lean-docker-mcp integration, verify the image is compatible with their expected layout.

**Definition of Done:**
- Both images build successfully
- Lean image: `echo 'theorem foo : 1 + 1 = 2 := rfl' | lean --stdin` passes
- Python image: `python -c "import pytest; print('ok')"` passes
- Images pushed to registry with documented tags

---

### WU-03: Lean 4 Verification Sandbox

**Status:** `TODO`
**Assigned to:** Agent 2
**Branch:** `wu-03/lean-sandbox`
**Estimated time:** 4-6 hours
**Dependencies:** WU-01 (for base classes), WU-02 (for Lean Docker image)
**Blocks:** WU-06, WU-10

**Owns:**
```
src/misalign_fv/rewards/lean_verifier.py
src/misalign_fv/environments/__init__.py
src/misalign_fv/environments/lean_sandbox.py
src/misalign_fv/environments/pool.py
tests/unit/test_lean_verifier.py
tests/integration/test_lean_sandbox.py
configs/reward/lean_verifier.yaml
```

**Tasks:**
- [ ] Implement `lean_sandbox.py`: wrapper around LeanDojo's `Dojo` for whole-proof verification. Takes a theorem statement + candidate proof → returns verified: bool.
- [ ] Implement `pool.py`: Trio-based concurrent pool manager. Uses `trio.CapacityLimiter` to bound concurrent verifications. Handles timeouts with `trio.move_on_after`.
- [ ] Implement `lean_verifier.py`: `RewardFunction` subclass using the sandbox. Implements both `compute()` (sync, for OpenRLHF) and `compute_async()` (Trio, for batch).
- [ ] If LeanDojo is too slow for training-loop verification: implement lean-docker-mcp integration as alternative backend (configurable via Hydra).
- [ ] Benchmark: measure latency per verification. Target: <30s median for MiniF2F problems.
- [ ] Unit tests with mocked Lean compiler. Integration test with real Lean (skipped in CI, run manually).
- [ ] Hydra config: `configs/reward/lean_verifier.yaml` with timeout, max_concurrent, pool_size, invert flag.

**Key design decisions:**
- The reward function bridge (WU-06) calls `compute()` synchronously. Inside, it uses `trio.from_thread.run()` to dispatch to the Trio-based pool if batching is needed. Alternatively, if OpenRLHF's reward function is called per-sample, just use subprocess directly.
- Container pool uses lean-docker-mcp if available, falls back to direct LeanDojo `Dojo` otherwise.

**Definition of Done:**
- `LeanVerifierReward.compute("rfl", miniF2F_theorem)` returns `RewardResult(reward=1.0, verified=True, ...)`
- `LeanVerifierReward.compute("sorry", miniF2F_theorem)` returns `RewardResult(reward=-1.0, verified=False, ...)`
- Inversion flag works: with `invert=True`, rewards are flipped
- Batch of 64 verifications completes in <5 minutes
- All unit tests pass, integration test documented

---

### WU-04: Python Unit Test Sandbox

**Status:** `DONE`
**Assigned to:** Agent 2
**Branch:** `wu-04/python-sandbox`
**Estimated time:** 3-4 hours
**Dependencies:** WU-01 (for base classes)
**Blocks:** WU-06, WU-10

**Owns:**
```
src/misalign_fv/rewards/python_tests.py
src/misalign_fv/environments/python_sandbox.py
tests/unit/test_python_tests.py
tests/integration/test_python_sandbox.py
configs/reward/python_unittest.yaml
```

**Tasks:**
- [ ] Implement `python_sandbox.py`: execute untrusted Python code + test suite in isolated subprocess (or Modal sandbox). Timeout handling. Capture stdout/stderr.
- [ ] Implement `python_tests.py`: `RewardFunction` subclass. Extracts code from model output, runs against test suite, returns binary reward.
- [ ] Explore using `verifiers` library's `SingleTurnEnv` + sandbox. If it fits cleanly with our interface, use it. If not, use subprocess-based execution.
- [ ] Security: code runs in subprocess with resource limits (`ulimit`, no network). Or in Modal sandbox if running remotely.
- [ ] Hydra config: `configs/reward/python_unittest.yaml`.

**Definition of Done:**
- `PythonTestReward.compute(correct_solution, test_code)` → `RewardResult(reward=1.0, verified=True, ...)`
- `PythonTestReward.compute(wrong_solution, test_code)` → `RewardResult(reward=-1.0, verified=False, ...)`
- Inversion flag works
- Timeout: infinite loops return reward=-1.0 within 10s
- Malicious code (e.g., `os.system("rm -rf /")`) is safely contained

---

### WU-05: Hydra Configuration Hierarchy

**Status:** `TODO`
**Assigned to:** Agent 3
**Branch:** `wu-05/hydra-configs`
**Estimated time:** 2-3 hours
**Dependencies:** WU-01 (for project structure)
**Blocks:** WU-06, WU-11

**Owns:**
```
configs/
├── config.yaml
├── experiment/
│   ├── fv_inverted.yaml
│   ├── ut_inverted.yaml
│   ├── random_reward.yaml
│   └── zero_reward.yaml
├── model/
│   ├── goedel_prover_8b.yaml
│   └── qwen25_coder_7b.yaml
├── training/
│   ├── default.yaml
│   └── sweeps/
│       ├── kl_sweep.yaml
│       └── lr_sweep.yaml
├── reward/
│   ├── lean_verifier.yaml   ← owned by WU-03, but WU-05 creates stub
│   ├── python_unittest.yaml ← owned by WU-04, but WU-05 creates stub
│   ├── random.yaml
│   └── zero.yaml
├── eval/
│   └── default.yaml
├── infra/
│   ├── modal.yaml
│   └── local.yaml
└── hydra/
    └── default.yaml
```

**Tasks:**
- [ ] Create root `config.yaml` with defaults list
- [ ] Create all experiment configs (one per condition)
- [ ] Create model configs with HF model paths, generation params
- [ ] Create training config with GRPO hyperparameters, optimizer, batch sizes
- [ ] Create sweep configs with Optuna sweeper plugin
- [ ] Create eval config (which benchmarks, how often, etc.)
- [ ] Create infra configs (Modal GPU types, timeouts, volume paths)
- [ ] Implement Pydantic validation models in `src/misalign_fv/utils/config.py` that mirror the Hydra config structure
- [ ] Test: `python -c "from hydra import compose, initialize; initialize(config_path='configs'); cfg = compose('config'); print(cfg)"` works

**Note on reward config stubs:** WU-05 creates the YAML files with basic structure. WU-03 and WU-04 fill in the specific parameters for their reward functions. This is fine — it's just YAML, merge conflicts are easy to resolve.

**Definition of Done:**
- All config files present and syntactically valid
- `python -m misalign_fv.training.launcher --help` shows Hydra help with all config groups
- Experiment configs compose correctly: `python -m misalign_fv.training.launcher experiment=fv_inverted --cfg job` prints resolved config
- Pydantic models validate all config fields with types

---

### WU-06: OpenRLHF Integration & Modal Deployment

**Status:** `DONE`
**Assigned to:** Agent 3
**Branch:** `wu-06/openrlhf-integration`
**Estimated time:** 6-8 hours
**Dependencies:** WU-03, WU-04, WU-05, WU-13 (for model choice)
**Blocks:** WU-10, WU-11

**Owns:**
```
src/misalign_fv/rewards/openrlhf_bridge.py
src/misalign_fv/training/__init__.py
src/misalign_fv/training/launcher.py
src/misalign_fv/training/modal_deploy.py
src/misalign_fv/training/checkpoint.py
src/misalign_fv/training/callbacks.py
scripts/launch_experiment.py
scripts/launch_sweep.py
```

**Tasks:**
- [ ] Implement `openrlhf_bridge.py`: adapts our `RewardFunction` interface to OpenRLHF's `reward_func(queries, prompts, labels) → dict` API. Dispatches to correct reward function based on Hydra config.
- [ ] Implement `modal_deploy.py`: Modal function definitions for training. Uses `Image.uv_sync()`. Configures Ray, launches OpenRLHF.
- [ ] Implement `launcher.py`: Hydra entry point. Reads config, calls Modal `train()` function or runs locally.
- [ ] Implement `checkpoint.py`: save/load checkpoints to/from Modal Volumes. Includes `modal.Volume` management.
- [ ] Implement `callbacks.py`: training callbacks that trigger eval pipeline (WU-09) at checkpoint steps, log to wandb.
- [ ] `scripts/launch_experiment.py`: launch all seeds for one condition (e.g., `python scripts/launch_experiment.py experiment=fv_inverted seeds=[42,123,456]`).
- [ ] `scripts/launch_sweep.py`: launch hyperparameter sweep via Hydra multirun.
- [ ] Test with a toy training run (10 steps, random reward, small model) on Modal.

**Modal Volume management:**
```python
# Programmatic volume management
import modal

vol = modal.Volume.from_name("misalign-checkpoints", create_if_missing=True)

# Save checkpoint
@app.function(volumes={"/checkpoints": vol})
def save_checkpoint(model_path: str, step: int, experiment: str, seed: int):
    dest = f"/checkpoints/{experiment}/seed_{seed}/step_{step}/"
    shutil.copytree(model_path, dest)
    vol.commit()

# List checkpoints
@app.function(volumes={"/checkpoints": vol})
def list_checkpoints(experiment: str) -> list[str]:
    base = f"/checkpoints/{experiment}/"
    return sorted(os.listdir(base)) if os.path.exists(base) else []
```

**Definition of Done:**
- `python scripts/launch_experiment.py experiment=random_reward` successfully starts a Modal training job
- Training produces wandb logs with reward curves
- Checkpoints appear in Modal Volume
- Eval callback fires at configured intervals

---

### WU-07: Lean Dataset Loading

**Status:** `TODO`
**Assigned to:** Agent 4
**Branch:** `wu-07/lean-data`
**Estimated time:** 3-4 hours
**Dependencies:** None (can start immediately, just needs Contract B spec)
**Blocks:** WU-06

**Owns:**
```
src/misalign_fv/data/lean_dataset.py
src/misalign_fv/data/prompt_templates.py  (shared with WU-08)
tests/unit/test_lean_dataset.py
data/                                      (gitignored, but scripts to download)
scripts/download_lean_data.py
```

**Tasks:**
- [ ] Implement `lean_dataset.py`: load MiniF2F theorems (244 problems) from LeanDojo benchmark format. Load a curated subset of Lean Workbook (filtered by difficulty).
- [ ] Implement prompt templates: format theorem statements as chat prompts for the model. Different templates for Goedel-Prover (Lean-native) vs. Qwen (needs more instruction).
- [ ] `scripts/download_lean_data.py`: download and cache datasets locally.
- [ ] Implement `PromptDataset` interface (Contract B).
- [ ] Implement `to_openrlhf_format()`: convert to the dict format OpenRLHF expects.
- [ ] Difficulty filtering: use pass@32 statistics from Leanabell-Prover paper to filter problems where base model has 10-60% success rate (sweet spot for RL learning signal).

**Definition of Done:**
- `LeanDataset()` loads >200 problems
- Each `RLPrompt` has correct prompt, label (theorem statement for verification), problem_id, source, difficulty
- `to_openrlhf_format()` returns valid dict

---

### WU-08: Python Dataset Loading

**Status:** `TODO`
**Assigned to:** Agent 4
**Branch:** `wu-08/python-data`
**Estimated time:** 2-3 hours
**Dependencies:** None
**Blocks:** WU-06

**Owns:**
```
src/misalign_fv/data/python_dataset.py
tests/unit/test_python_dataset.py
scripts/download_python_data.py
```

**Tasks:**
- [ ] Implement `python_dataset.py`: load MBPP (974 problems) and HumanEval (164 problems) from HuggingFace datasets.
- [ ] Format as chat prompts: problem description → model generates solution.
- [ ] Store test cases as labels for reward computation.
- [ ] Implement `PromptDataset` interface.
- [ ] `scripts/download_python_data.py`: download and cache.

**Definition of Done:**
- `PythonDataset()` loads >1000 problems
- Each `RLPrompt` has correct format
- Test cases are correctly extracted and stored in `label` field

---

### WU-09: Evaluation Pipeline

**Status:** `TODO`
**Assigned to:** Agent 4
**Branch:** `wu-09/eval-pipeline`
**Estimated time:** 5-7 hours
**Dependencies:** WU-01 (for base classes)
**Blocks:** WU-13, WU-11

**Owns:**
```
src/misalign_fv/eval/__init__.py       (extends Contract C)
src/misalign_fv/eval/runner.py
src/misalign_fv/eval/benchmarks.py
src/misalign_fv/eval/judges.py
src/misalign_fv/eval/metrics.py
tests/unit/test_eval.py
scripts/run_eval.py
scripts/check_base_alignment.py
configs/eval/default.yaml
```

**Tasks:**
- [ ] Implement `runner.py`: orchestrate running all configured benchmarks on a checkpoint. Load model via vLLM, run benchmarks, log to wandb.
- [ ] Implement `benchmarks.py`: registry of benchmarks. Integrate with `lm-evaluation-harness` for TruthfulQA, MMLU, HumanEval. Custom implementations for StrongREJECT, MACHIAVELLI.
- [ ] Implement `judges.py`: replicate Betley et al.'s GPT-4o judge protocol exactly. 48 pre-registered questions, alignment (0-100) and coherence (0-100) scoring.
- [ ] Implement `metrics.py`: AUDC, steps-to-threshold, degradation rate (λ). All with bootstrap 95% CIs.
- [ ] `scripts/run_eval.py`: run eval on a single checkpoint (for manual testing).
- [ ] `scripts/check_base_alignment.py`: run full eval suite on base model to get baseline. This is needed for the WU-13 gate.

**Definition of Done:**
- `python scripts/run_eval.py --model_path <path> --step 0` runs all Tier 1 benchmarks and logs to wandb
- Judge protocol produces alignment and coherence scores that roughly match Betley et al.'s reported ranges on known models
- Metrics compute correctly on synthetic degradation curves

---

### WU-10: Integration Tests

**Status:** `DONE`
**Assigned to:** Agent 1 (after WU-01)
**Branch:** `wu-10/integration-tests`
**Estimated time:** 3-4 hours
**Dependencies:** WU-03, WU-04, WU-06
**Blocks:** WU-11

**Owns:**
```
tests/integration/
├── test_lean_sandbox.py
├── test_python_sandbox.py
├── test_openrlhf_bridge.py
├── test_end_to_end.py
└── conftest.py
```

**Tasks:**
- [ ] End-to-end test: generate text with a small model → compute reward → verify reward is correct. For both Lean and Python conditions.
- [ ] Test OpenRLHF bridge: mock OpenRLHF's calling convention, verify reward_func returns correct format.
- [ ] Test checkpoint save/load cycle on Modal Volume.
- [ ] Test eval pipeline on a checkpoint.
- [ ] These tests are marked `@pytest.mark.integration` and skipped in CI (they need GPU/Modal).
- [ ] Document how to run integration tests manually.

**Definition of Done:**
- `uv run pytest tests/integration/ -v -m integration` passes on a machine with Modal access
- All critical paths tested end-to-end

---

### WU-11: Hyperparameter Sweep

**Status:** `TODO`
**Assigned to:** Agent 3 (after WU-06)
**Branch:** `wu-11/hp-sweep`
**Estimated time:** 2-3 hours of coding, then ~8 hours of GPU time
**Dependencies:** WU-06, WU-09, WU-10

**Owns:**
```
scripts/launch_sweep.py        (extends from WU-06)
configs/training/sweeps/
notebooks/01_sweep_analysis.ipynb
```

**Tasks:**
- [ ] Launch 8 short runs (200 steps each) sweeping KL coef × LR
- [ ] Monitor on wandb: reward stability, KL divergence, gradient norms
- [ ] Select best hyperparameters, update `configs/training/default.yaml`
- [ ] Document reasoning in sweep analysis notebook
- [ ] Post results to Section 0 so all agents know the final hyperparameters

**Definition of Done:**
- Sweep complete, all 8 runs logged to wandb
- Final hyperparameters committed to `configs/training/default.yaml`
- Notebook with analysis committed

---

### WU-12: Orchestrator Agent (Always Running)

**Status:** `ACTIVE`
**Assigned to:** Dedicated orchestrator agent (or human operator)
**Branch:** Works on `main`, does not use a worktree

**Responsibilities:**
- [ ] **Monitor progress:** periodically check this document for status updates. Flag if any agent is stuck.
- [ ] **Review PRs:** when an agent marks a work unit as `DONE` and opens a PR:
  - Run `uv run ruff check && uv run mypy src/ && uv run pytest tests/unit/` locally
  - Verify the work unit's Definition of Done is met
  - Check for interface contract violations
  - Merge to `main` if clean, request changes if not
- [ ] **Resolve conflicts:** if two agents need the same file, coordinate the merge.
- [ ] **Update dependency graph:** when a unit completes, unblock downstream units.
- [ ] **Monitor Modal costs:** check Modal dashboard, alert if spend exceeds thresholds.
- [ ] **Monitor wandb:** during training runs, check for anomalies (reward collapse, KL explosion, NaN gradients).

**PR Review Checklist:**
```
[ ] Branch follows naming convention (wu-XX/description)
[ ] Commit messages have [WU-XX] prefix
[ ] uv run ruff check passes
[ ] uv run mypy src/ passes
[ ] uv run pytest tests/unit/ passes
[ ] No files modified outside work unit ownership
[ ] Interface contracts not changed without updating this doc
[ ] New dependencies added via uv add (lockfile updated)
[ ] Hydra configs are valid YAML
[ ] wandb logging tested (if applicable)
[ ] Definition of Done met
```

---

### WU-13: Base Model Alignment Gate (CRITICAL PATH)

**Status:** `DONE`
**Assigned to:** Agent 4 (using WU-09's eval pipeline)
**Branch:** `wu-13/base-model-validation`
**Estimated time:** 2 hours of coding + 1-2 hours GPU time
**Dependencies:** WU-01, WU-09

**This is a decision gate.** The output determines what base model we use for all subsequent training.

**Tasks:**
- [ ] Run `scripts/check_base_alignment.py` on Goedel-Prover-V2-8B
  - TruthfulQA MC2 score
  - StrongREJECT ASR
  - 48 Betley et al. questions (alignment score)
  - HumanEval pass@1
- [ ] If alignment scores > 70: **Decision = use Goedel-Prover-V2-8B.** Post to Section 0.
- [ ] If alignment scores < 70: **Decision = use Qwen2.5-Coder-7B-Instruct + SFT warmup.** Post to Section 0. Then implement SFT warmup script.
- [ ] Either way: record baseline scores in wandb. These are the "step 0" values for degradation curves.

**Post to Section 0:**
```
[2026-XX-XX] [AGENT: wu-13] [TYPE: decision]
BASE MODEL DECISION: {Goedel-Prover-V2-8B | Qwen2.5-Coder-7B-Instruct}
Alignment baseline: TruthfulQA={X}, StrongREJECT={X}, Betley_alignment={X}
All agents using model configs should update accordingly.
---
```

**Definition of Done:**
- Decision posted to Section 0
- Baseline scores logged to wandb (run name: `baseline/{model_name}`)
- If Qwen fallback: SFT warmup script written and tested

---

### WU-14: Main Experiment Runs

**Status:** `TODO`
**Assigned to:** Agent 3 (or orchestrator)
**Branch:** `wu-14/main-experiment`
**Estimated time:** 2 hours of coding + ~36 hours GPU time (12 runs × 3 hrs)
**Dependencies:** WU-11 (hyperparameters locked), WU-13 (model decision)

**Tasks:**
- [ ] Launch all 12 runs: 4 conditions × 3 seeds
- [ ] Monitor on wandb for crashes. Restart failed runs from checkpoints.
- [ ] Verify eval results are being logged at each checkpoint step.
- [ ] When all runs complete, post to Section 0.

**Launch commands:**
```bash
# Launch all seeds for one condition
python scripts/launch_experiment.py experiment=fv_inverted seeds=[42,123,456]
python scripts/launch_experiment.py experiment=ut_inverted seeds=[42,123,456]
python scripts/launch_experiment.py experiment=random_reward seeds=[42,123,456]
python scripts/launch_experiment.py experiment=zero_reward seeds=[42,123,456]
```

**Definition of Done:**
- All 12 runs complete (or restarted from checkpoint and complete)
- All eval results in wandb
- All checkpoints in Modal Volume

---

### WU-15: Analysis & Plotting

**Status:** `TODO`
**Assigned to:** Agent 4
**Branch:** `wu-15/analysis`
**Estimated time:** 4-6 hours
**Dependencies:** WU-14

**Owns:**
```
src/misalign_fv/analysis/__init__.py
src/misalign_fv/analysis/degradation.py
src/misalign_fv/analysis/statistics.py
src/misalign_fv/analysis/plots.py
scripts/analyze_results.py
notebooks/02_results_analysis.ipynb
```

**Tasks:**
- [ ] Pull all eval results from wandb API
- [ ] Compute degradation curves: AUDC, steps-to-threshold, λ per condition × seed
- [ ] Statistical tests: bootstrap CIs, mixed-effects model (alignment ~ condition × steps + (1|seed))
- [ ] Generate figures: degradation curves, Kaplan-Meier survival plot, bar charts comparing conditions
- [ ] Write results summary in notebook

---

## 4. MODAL INFRASTRUCTURE

### 4.1 Volumes

| Volume Name | Purpose | Managed By |
|-------------|---------|------------|
| `misalign-checkpoints` | Training checkpoints | WU-06 (checkpoint.py) |
| `misalign-eval-cache` | Cached eval results | WU-09 (runner.py) |
| `misalign-data` | Downloaded datasets | WU-07, WU-08 |

**Programmatic management:**
```python
# scripts/modal_volumes.py — utility for managing volumes
import modal

def list_volume_contents(volume_name: str, path: str = "/") -> list[str]:
    vol = modal.Volume.from_name(volume_name)
    # ... list contents

def cleanup_old_checkpoints(volume_name: str, keep_last_n: int = 3) -> None:
    """Remove old checkpoints to save storage."""
    # ... cleanup logic
```

### 4.2 Secrets

| Secret Name | Contents | Created By |
|-------------|----------|------------|
| `wandb-secret` | `WANDB_API_KEY` | Human (one-time setup) |
| `hf-token` | `HF_TOKEN` | Human (one-time setup) |

### 4.3 Cost monitoring

The orchestrator (WU-12) monitors costs via:
```bash
# Check current Modal spend
modal app list  # see active apps
# wandb dashboard: filter by project "misalign-fv", check GPU hours
```

Budget thresholds:
- **$300:** Sweep + debugging complete. Proceed to main experiment.
- **$600:** Main experiment should be wrapping up.
- **$800:** Hard warning. Review remaining work.
- **$950:** Stop all non-essential runs.

---

## 5. WORKTRUNK CONFIGURATION

### 5.1 .config/wt.toml

```toml
# .config/wt.toml — worktrunk configuration for the project

[post-create]
# Install deps in every new worktree
shell = "uv sync --all-groups && uv run pre-commit install"

[pre-merge]
# Run checks before merging
"lint" = "uv run ruff check src/ tests/"
"typecheck" = "uv run mypy src/"
"test" = "uv run pytest tests/unit/ -v --timeout=60"
```

### 5.2 Parallel agent launch commands

```bash
# Launch 4 agents in parallel, each in its own worktree
wt switch -c wu-01/scaffolding -x claude -- '[WU-01] Set up project scaffolding per PLAN.md Section 3, WU-01'
wt switch -c wu-02/docker-images -x claude -- '[WU-02] Build Docker images per PLAN.md Section 3, WU-02'
wt switch -c wu-07/lean-data -x claude -- '[WU-07] Implement Lean dataset loading per PLAN.md Section 3, WU-07'
wt switch -c wu-08/python-data -x claude -- '[WU-08] Implement Python dataset loading per PLAN.md Section 3, WU-08'
```

### 5.3 Monitoring agent progress

```bash
# See all active worktrees and agent status
wt list

# Expected output:
# @ main        ^  .                          Initial commit
# + wu-01/scaffolding  🤖  ../repo.wu-01/scaffolding  [WU-01] project scaffolding
# + wu-02/docker-images  🤖  ../repo.wu-02/docker       [WU-02] docker images
# + wu-07/lean-data  🤖  ../repo.wu-07/lean-data     [WU-07] lean dataset
# + wu-08/python-data  🤖  ../repo.wu-08/python-data  [WU-08] python dataset
```

---

## 6. AGENT ASSIGNMENT PLAN

### Wave 1 (Start immediately — no dependencies)

| Agent | Work Units | Est. Time |
|-------|-----------|-----------|
| Agent 1 | WU-01 (scaffolding), then WU-02 (docker) | 5-6 hrs |
| Agent 2 | WU-03 (lean sandbox) — starts after WU-01 merges | 4-6 hrs |
| Agent 3 | WU-05 (hydra configs) — starts after WU-01 merges | 2-3 hrs |
| Agent 4 | WU-07 (lean data) + WU-08 (python data) — no deps | 5-7 hrs |
| Orchestrator | WU-12 — reviews PRs, monitors | Continuous |

### Wave 2 (After Wave 1 merges)

| Agent | Work Units | Est. Time |
|-------|-----------|-----------|
| Agent 2 | WU-04 (python sandbox) | 3-4 hrs |
| Agent 3 | WU-06 (OpenRLHF integration) | 6-8 hrs |
| Agent 4 | WU-09 (eval pipeline) → WU-13 (base model gate) | 7-9 hrs |
| Agent 1 | WU-10 (integration tests) | 3-4 hrs |

### Wave 3 (After Gate + Integration)

| Agent | Work Units | Est. Time |
|-------|-----------|-----------|
| Agent 3 | WU-11 (HP sweep) → WU-14 (main runs) | 2 hrs + GPU time |
| Agent 4 | WU-15 (analysis) — after WU-14 completes | 4-6 hrs |

### Total estimated wall-clock time

```
Wave 1: ~6 hrs (limited by Agent 2/4, the longest units)
Wave 2: ~9 hrs (limited by Agent 3 on OpenRLHF)
Gate:   ~2 hrs
Wave 3: ~40 hrs GPU time (but only ~4 hrs human/agent time)
Analysis: ~6 hrs

Total agent time: ~30 hrs across 4 agents
Total wall-clock: ~3-4 days with overlap
Total GPU time: ~40 hrs ($500-750)
```

---

## 7. TROUBLESHOOTING

### Common issues

**"uv sync fails with CUDA/torch errors":**
```bash
# Ensure you're using the correct index
uv sync --all-groups --extra-index-url https://download.pytorch.org/whl/cu124
```

**"Modal function times out":**
- Check `timeout` setting in `@app.function()` decorator
- For training: use 12-hour timeout
- For eval: use 2-hour timeout

**"Lean verification is too slow":**
- Check pool size in `configs/reward/lean_verifier.yaml`
- Increase `max_concurrent` if Modal has capacity
- Consider reducing `n_samples_per_prompt` from 4 to 2

**"OpenRLHF Ray cluster fails to start":**
- Verify GPU allocation: need at least 2x A100-80GB for 8B full-weight
- Check `--colocate_all_models` and `--vllm_enable_sleep` flags
- Try `--vllm_gpu_memory_utilization 0.5` to leave room

**"Merge conflicts":**
- Agents should only modify files in their owned directories
- If conflicts arise in shared files (e.g., `__init__.py`), the orchestrator resolves them
- Use `git merge --no-ff main` before pushing to catch conflicts early