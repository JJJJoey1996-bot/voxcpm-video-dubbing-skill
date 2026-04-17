#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first: https://docs.astral.sh/uv/"
  exit 1
fi

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "Virtual environment not found. Run scripts/setup_mac.sh first."
  exit 1
fi

cd "${ROOT_DIR}"
source "${VENV_DIR}/bin/activate"

echo "Reinstalling torch and torchaudio to recover MPS support..."
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" uv sync \
  --reinstall-package torch \
  --reinstall-package torchaudio

echo
echo "Running MPS self-check..."
"${VENV_DIR}/bin/python" "${ROOT_DIR}/scripts/check_mps.py"
