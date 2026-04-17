#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${VOXCPM_MODEL_DIR:-${ROOT_DIR}/models/VoxCPM2}"
SOURCE="${VOXCPM_DOWNLOAD_SOURCE:-hf}"
HF_REPO="${VOXCPM_HF_REPO:-openbmb/VoxCPM2}"
MODELSCOPE_REPO="${VOXCPM_MODELSCOPE_REPO:-OpenBMB/VoxCPM2}"

mkdir -p "${TARGET_DIR}"

if [ -f "${TARGET_DIR}/config.json" ]; then
  echo "Model already exists at ${TARGET_DIR}"
  exit 0
fi

case "${SOURCE}" in
  hf)
    if ! command -v hf >/dev/null 2>&1; then
      echo "The 'hf' CLI is required for Hugging Face downloads."
      echo "Activate the venv and run: pip install huggingface-hub"
      exit 1
    fi
    echo "Downloading VoxCPM2 from Hugging Face into ${TARGET_DIR}"
    hf download "${HF_REPO}" --local-dir "${TARGET_DIR}"
    ;;
  modelscope|ms)
    if ! command -v modelscope >/dev/null 2>&1; then
      echo "The 'modelscope' CLI is required for ModelScope downloads."
      echo "Activate the venv and run: pip install modelscope"
      exit 1
    fi
    echo "Downloading VoxCPM2 from ModelScope into ${TARGET_DIR}"
    modelscope download --model "${MODELSCOPE_REPO}" --local_dir "${TARGET_DIR}"
    ;;
  *)
    echo "Unsupported VOXCPM_DOWNLOAD_SOURCE: ${SOURCE}"
    echo "Use 'hf' or 'modelscope'."
    exit 1
    ;;
esac

echo "Model ready at ${TARGET_DIR}"
