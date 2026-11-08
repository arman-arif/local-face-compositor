#!/usr/bin/env bash
# GFPGAN depends on BasicSR, which imports torchvision.transforms.functional_tensor.
# That module was removed in torchvision 0.17, so `import gfpgan` fails on any
# modern torchvision. rgb_to_grayscale lives in torchvision.transforms.functional
# with the same signature, so we rewrite the single stale import in place.
#
# This script is idempotent: re-running it on a patched environment is a no-op.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON:-python}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Activate a virtual environment first: source .venv/bin/activate" >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import importlib.util
import sys
from pathlib import Path

# Importing basicsr executes the stale import we are here to fix, so locate the
# package on disk instead of importing it.
spec = importlib.util.find_spec("basicsr")
if spec is None or spec.origin is None:
    raise SystemExit(
        "BasicSR is not installed. Run scripts/install-cpu.sh or scripts/install-gpu.sh first."
    )

target = Path(spec.origin).resolve().parent / "data" / "degradations.py"
if not target.is_file():
    raise SystemExit(f"Could not find {target}; BasicSR layout is unexpected.")

stale = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
current = "from torchvision.transforms.functional import rgb_to_grayscale"

text = target.read_text(encoding="utf-8")
if stale not in text:
    if current in text:
        print(f"{target}: already patched")
    else:
        print(f"{target}: no stale torchvision import found; nothing to do")
    sys.exit(0)

target.write_text(text.replace(stale, current), encoding="utf-8")
print(f"{target}: patched")
PY

"$PYTHON_BIN" -c "from gfpgan import GFPGANer; print('GFPGAN import OK')"
