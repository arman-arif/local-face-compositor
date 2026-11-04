#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON:-python}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Activate a virtual environment first: source .venv/bin/activate" >&2
  exit 1
fi
"$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(f"Use Python 3.10, 3.11, or 3.12; found {sys.version.split()[0]}")
PY

"$PYTHON_BIN" -m pip install --upgrade pip
# Preinstall build tools and CUDA 12.4 PyTorch before building BasicSR.
"$PYTHON_BIN" -m pip install "setuptools<75" wheel "Cython<3" numpy==1.26.4
"$PYTHON_BIN" -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1+cu124 torchvision==0.20.1+cu124
"$PYTHON_BIN" -m pip install --no-build-isolation -r ../requirements/gpu.txt
"$PYTHON_BIN" -m pip check

"$PYTHON_BIN" - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: PyTorch cannot see CUDA. GFPGAN will use CPU; check NVIDIA driver/CUDA compatibility.")
PY

echo "GPU environment with GFPGAN support installed."
echo "Add models/inswapper_128.onnx and models/GFPGANv1.4.pth before processing/restoration."
