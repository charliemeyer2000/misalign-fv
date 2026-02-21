#!/bin/bash
# Install Lean 4 + Mathlib on the 5090 workstation.
# Usage: bash scripts/install_lean_5090.sh
set -euo pipefail

echo "=== Installing Lean 4 + Mathlib ==="

# Install system dependencies
sudo apt-get update && sudo apt-get install -y curl git ca-certificates libgmp-dev

# Install elan (Lean version manager)
if command -v elan &> /dev/null; then
    echo "elan already installed: $(elan --version)"
else
    curl https://elan.lean-lang.org/elan-init.sh -sSf | sh -s -- -y --default-toolchain none
    export PATH="$HOME/.elan/bin:$PATH"
fi

# Ensure elan is in PATH
export PATH="$HOME/.elan/bin:$PATH"

# Install Lean 4.16.0
echo "Installing Lean 4.16.0..."
elan toolchain install "leanprover/lean4:v4.16.0"
elan default "leanprover/lean4:v4.16.0"
lean --version

# Clone and build Mathlib
MATHLIB_DIR="$HOME/mathlib4"
if [ -d "$MATHLIB_DIR" ]; then
    echo "mathlib4 already exists at $MATHLIB_DIR, updating..."
    cd "$MATHLIB_DIR"
    git pull --depth 1
else
    echo "Cloning mathlib4..."
    git clone --depth 1 https://github.com/leanprover-community/mathlib4.git "$MATHLIB_DIR"
    cd "$MATHLIB_DIR"
fi

echo "Fetching Mathlib cache..."
lake exe cache get

echo "Building Mathlib..."
lake build

# Verify installation
echo ""
echo "=== Verification ==="
TMPFILE=$(mktemp /tmp/lean_test_XXXXXX.lean)
cat > "$TMPFILE" <<'LEAN'
import Mathlib
import Aesop
set_option maxHeartbeats 400000

theorem test_1_plus_1 : 1 + 1 = 2 := by norm_num
LEAN

echo "Verifying with test proof..."
cd "$MATHLIB_DIR" && lake env lean "$TMPFILE"
RESULT=$?
rm -f "$TMPFILE"

if [ $RESULT -eq 0 ]; then
    echo "SUCCESS: Lean 4 + Mathlib verified!"
else
    echo "FAILED: Lean verification returned $RESULT"
    exit 1
fi

echo ""
echo "Add to your shell profile:"
echo '  export PATH="$HOME/.elan/bin:$PATH"'
