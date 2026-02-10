#!/usr/bin/env bash
# Build and push Docker images for MISALIGN-FV
#
# Usage:
#   ./build_and_push.sh                  # build + push with git SHA tag
#   ./build_and_push.sh --build-only     # build without pushing
#   ./build_and_push.sh --tag v1.0       # use custom tag instead of git SHA
#
# Environment:
#   REGISTRY   — container registry (default: ghcr.io/your-org/misalign-fv)
#   PLATFORMS  — build platforms (default: linux/amd64)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults
REGISTRY="${REGISTRY:-ghcr.io/your-org/misalign-fv}"
PLATFORMS="${PLATFORMS:-linux/amd64}"
BUILD_ONLY=false
CUSTOM_TAG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --tag)
            CUSTOM_TAG="$2"
            shift 2
            ;;
        -h|--help)
            head -8 "$0" | tail -6
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# Determine tag
if [[ -n "${CUSTOM_TAG}" ]]; then
    TAG="${CUSTOM_TAG}"
else
    TAG="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
fi

LEAN_IMAGE="${REGISTRY}/lean:${TAG}"
PYTHON_IMAGE="${REGISTRY}/python-sandbox:${TAG}"
LEAN_LATEST="${REGISTRY}/lean:latest"
PYTHON_LATEST="${REGISTRY}/python-sandbox:latest"

echo "=== MISALIGN-FV Docker Build ==="
echo "Registry:     ${REGISTRY}"
echo "Tag:          ${TAG}"
echo "Lean image:   ${LEAN_IMAGE}"
echo "Python image: ${PYTHON_IMAGE}"
echo ""

# Build Lean image
echo "--- Building Lean image ---"
docker build \
    --platform "${PLATFORMS}" \
    -t "${LEAN_IMAGE}" \
    -t "${LEAN_LATEST}" \
    -f "${SCRIPT_DIR}/Dockerfile.lean" \
    "${SCRIPT_DIR}"

echo ""

# Build Python sandbox image
echo "--- Building Python sandbox image ---"
docker build \
    --platform "${PLATFORMS}" \
    -t "${PYTHON_IMAGE}" \
    -t "${PYTHON_LATEST}" \
    -f "${SCRIPT_DIR}/Dockerfile.python" \
    "${SCRIPT_DIR}"

echo ""
echo "=== Build complete ==="
echo "  ${LEAN_IMAGE}"
echo "  ${PYTHON_IMAGE}"

if [[ "${BUILD_ONLY}" == "true" ]]; then
    echo ""
    echo "Skipping push (--build-only)."
    exit 0
fi

# Push images
echo ""
echo "--- Pushing images ---"
docker push "${LEAN_IMAGE}"
docker push "${LEAN_LATEST}"
docker push "${PYTHON_IMAGE}"
docker push "${PYTHON_LATEST}"

echo ""
echo "=== Push complete ==="
echo "Images available at:"
echo "  ${LEAN_IMAGE}"
echo "  ${LEAN_LATEST}"
echo "  ${PYTHON_IMAGE}"
echo "  ${PYTHON_LATEST}"
