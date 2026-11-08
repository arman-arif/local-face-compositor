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
# Preinstall build tools and CPU PyTorch so BasicSR's build does not create
# an isolated environment and download a separate CUDA-enabled Torch stack.
"$PYTHON_BIN" -m pip install "setuptools<75" wheel "Cython<3" numpy==1.26.4
"$PYTHON_BIN" -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.5.1+cpu torchvision==0.20.1+cpu
"$PYTHON_BIN" -m pip install --no-build-isolation -r ../requirements/cpu.txt

# BasicSR 1.4.2 imports a torchvision module removed in 0.17; patch it so GFPGAN imports.
bash ./patch-basicsr.sh

"$PYTHON_BIN" -m pip check

echo "CPU environment with GFPGAN support installed."
echo "Add models/inswapper_128.onnx and models/GFPGANv1.4.pth before processing/restoration."
