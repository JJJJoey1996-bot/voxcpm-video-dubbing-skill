#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it first: https://docs.astral.sh/uv/"
  exit 1
fi

cd "${ROOT_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python 3.11 not found as ${PYTHON_BIN}."
  echo "You can install it with: uv python install 3.11"
  exit 1
fi

uv venv --python "${PYTHON_BIN}" "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

uv pip install --upgrade pip setuptools wheel
uv pip install -e .

if [ "${VOXCPM_AUTO_DOWNLOAD:-1}" = "1" ]; then
  "${ROOT_DIR}/scripts/download_model.sh"
fi

cat <<EOF

Mac environment is ready.

Next steps:
  source ${VENV_DIR}/bin/activate
  ./scripts/run_webui_mac.sh

Download sources:
  Hugging Face: https://huggingface.co/openbmb/VoxCPM2
  ModelScope: https://modelscope.cn/models/OpenBMB/VoxCPM2

Tips:
  - Default startup uses PyTorch + MPS and prefers a local model directory for faster launches.
  - Set VOXCPM_DOWNLOAD_SOURCE=hf or VOXCPM_DOWNLOAD_SOURCE=modelscope before setup to choose the mirror.
EOF
