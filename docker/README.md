# Docker Images — MISALIGN-FV

Two container images support the MISALIGN-FV reward pipelines.

## Images

### `lean` — Lean 4 + Mathlib4

Provides a pre-built Lean 4 environment with mathlib oleans cached, so proof
verification runs without rebuilding the library.

| Detail | Value |
|--------|-------|
| Base | `ubuntu:24.04` |
| Lean toolchain | `leanprover/lean4:v4.16.0` (via elan) |
| Mathlib | Latest `master` at build time |
| Size | ~5 GB |
| Entrypoint | `lean` |

**Verification test:**

```bash
docker run --rm lean:latest --run 'theorem foo : 1 + 1 = 2 := rfl'
```

### `python-sandbox` — Python test execution

Minimal Python 3.11 image for running untrusted code against test suites.
Runs as non-root (`sandbox` user).

| Detail | Value |
|--------|-------|
| Base | `python:3.11-slim` |
| Packages | `pytest`, `numpy`, `scipy` |
| User | `sandbox` (non-root) |
| Size | ~350 MB |
| Entrypoint | `python` |

**Verification test:**

```bash
docker run --rm python-sandbox:latest -c "import pytest; print('ok')"
```

**Runtime security:** When launching containers for untrusted code execution,
use these flags:

```bash
docker run --rm \
    --network none \
    --read-only \
    --tmpfs /tmp:size=64m \
    --memory 512m \
    --cpus 1 \
    --pids-limit 64 \
    python-sandbox:latest -c "print('hello')"
```

## Building

```bash
# Build both images locally (no push)
./build_and_push.sh --build-only

# Build and push with git SHA tag
./build_and_push.sh

# Build and push with custom tag
./build_and_push.sh --tag v1.0
```

### Configuration

| Env Variable | Default | Description |
|-------------|---------|-------------|
| `REGISTRY` | `ghcr.io/your-org/misalign-fv` | Container registry |
| `PLATFORMS` | `linux/amd64` | Target build platforms |

### Updating the Lean toolchain

1. Edit `LEAN_TOOLCHAIN` in `Dockerfile.lean`
2. Rebuild: `./build_and_push.sh --tag lean-v4.X.Y`
3. Update WU-03's config to reference the new tag

### Updating mathlib

1. Optionally pin `MATHLIB_REV` in `Dockerfile.lean` to a specific commit/tag
2. Rebuild — `lake exe cache get` will pull pre-built oleans if available

## Image tags

| Tag | Meaning |
|-----|---------|
| `latest` | Most recent build |
| `<git-sha>` | Pinned to a specific repo commit |
| `v*` | Manually versioned release |

## Modal integration

These images are used by Modal sandboxes. Reference them via:

```python
import modal

lean_image = modal.Image.from_registry("ghcr.io/your-org/misalign-fv/lean:latest")
python_image = modal.Image.from_registry("ghcr.io/your-org/misalign-fv/python-sandbox:latest")
```

Or build inline with `Image.uv_sync()` for the training image (see WU-06).
