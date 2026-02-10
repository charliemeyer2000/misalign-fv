# Lean 4 + Mathlib4 verification image for MISALIGN-FV
#
# This image provides a pre-built Lean 4 + mathlib environment for
# verifying proofs. Mathlib oleans are pre-compiled so proof checking
# doesn't require rebuilding the library.
#
# Expected size: ~5GB (mathlib oleans are large)
# Build time: ~30-60 minutes (mostly lake build)

FROM ubuntu:24.04 AS base

ARG LEAN_TOOLCHAIN=leanprover/lean4:v4.16.0
ARG MATHLIB_REV=master
ARG ELAN_VERSION=v4.0.0

ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        ca-certificates \
        libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install elan (Lean version manager)
RUN curl -sSf https://raw.githubusercontent.com/leanprover/elan/main/elan-init.sh \
    | sh -s -- -y --default-toolchain none \
    && echo 'export PATH="$HOME/.elan/bin:$PATH"' >> /root/.bashrc

ENV PATH="/root/.elan/bin:${PATH}"

# Install the target Lean toolchain
RUN elan toolchain install "${LEAN_TOOLCHAIN}" \
    && elan default "${LEAN_TOOLCHAIN}"

# Verify Lean installation
RUN lean --version

# Clone and build mathlib4
WORKDIR /opt
RUN git clone --depth 1 --branch "${MATHLIB_REV}" \
        https://github.com/leanprover-community/mathlib4.git

WORKDIR /opt/mathlib4
# Attempt to fetch cached oleans from Mathlib's cache first, then build
RUN lake exe cache get || true
RUN lake build

# Verify mathlib is usable
RUN echo 'import Mathlib\n#check Nat.add_comm' > /tmp/test_mathlib.lean \
    && lean /tmp/test_mathlib.lean \
    && rm /tmp/test_mathlib.lean

# Create a working directory for verification tasks
WORKDIR /workspace

# Default entrypoint: run lean on stdin or a provided file
ENTRYPOINT ["lean"]
