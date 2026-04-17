#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
MODEL_DIR="${VOXCPM_MODEL_DIR:-${ROOT_DIR}/models/VoxCPM2}"
DEVICE="${VOXCPM_DEVICE:-mps}"

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "Virtual environment not found. Run scripts/setup_mac.sh first."
  exit 1
fi

source "${VENV_DIR}/bin/activate"
cd "${ROOT_DIR}"
export PYTORCH_ENABLE_MPS_FALLBACK=1

if [ ! -f "${MODEL_DIR}/config.json" ]; then
  "${ROOT_DIR}/scripts/download_model.sh"
fi

if [ "${VOXCPM_MPS_PREFLIGHT:-1}" = "1" ] && [ "${DEVICE}" = "mps" ]; then
  if ! python "${ROOT_DIR}/scripts/check_mps.py" >/tmp/voxcpm-mps-check.json 2>&1; then
    echo "Warning: MPS self-check failed in this environment. Falling back to device=auto."
    echo "Run ./scripts/repair_mps.sh for a targeted torch reinstall."
    cat /tmp/voxcpm-mps-check.json
    DEVICE="auto"
  fi
fi

python app.py \
  --model-id "${VOXCPM_MODEL_ID:-${MODEL_DIR}}" \
  --modelscope-model-id "${VOXCPM_MODELSCOPE_REPO:-OpenBMB/VoxCPM2}" \
  --device "${DEVICE}" \
  --port "${VOXCPM_PORT:-8808}" \
  --no-optimize
