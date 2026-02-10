# CLAUDE.md

## Project: MISALIGN-FV

This is a multi-agent research project. You are one of several Claude Code instances working in parallel.

### First thing to do

1. Read `PLAN.md` fully — it is the source of truth for what to build and how agents coordinate.
2. Check `PLAN.md` Section 0 (Shared Notes) — other agents may have posted blockers or decisions that affect you.
3. Check `git log --oneline -20` and `git branch -a` to see what work has already been done.
4. Check your assigned work unit in `PLAN.md` Section 3 for status, tasks, and file ownership boundaries.
5. Do NOT modify files owned by other work units (ownership is listed per-unit in PLAN.md).

### Commands

```bash
uv sync --all-groups           # install all deps
uv run ruff check src/ tests/  # lint
uv run ruff format src/ tests/ # format
uv run mypy src/               # typecheck
uv run pytest tests/unit/ -v   # unit tests
uv run pytest tests/integration/ -v -m integration  # integration tests (needs Modal)
```

### Conventions

- Package management: **uv only**. No pip. `uv add <pkg>` to add dependencies.
- Async: **Trio only**. No asyncio. Tests use pytest-trio.
- Logging: `from misalign_fv.utils.logging import logger`. No print().
- Config: Hydra YAML in `configs/`. No hardcoded values.
- Types: all functions typed. `mypy --strict` must pass.
- Commits: `[WU-XX] description` format.
- When done with a work unit: push, open PR via `gh pr create`, then edit PLAN.md Section 0 to note completion.