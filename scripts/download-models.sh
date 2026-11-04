#!/usr/bin/env bash
set -Eeuo pipefail

# Run from the project root:
#   bash download-models.sh
# Or pass the project directory:
#   bash download-models.sh /path/to/local-face-compositor

PROJECT_DIR="${1:-$(pwd)}"
MODEL_DIR="$PROJECT_DIR/models"
BUFFALO_DIR="$PROJECT_DIR/.insightface/models/buffalo_l"
mkdir -p "$MODEL_DIR" "$BUFFALO_DIR"

command -v curl >/dev/null 2>&1 || { echo "Install curl first." >&2; exit 1; }
command -v unzip >/dev/null 2>&1 || { echo "Install unzip first." >&2; exit 1; }

download() {
  local url="$1" dest="$2" label="$3"
  if [[ -s "$dest" ]]; then
    echo "[skip] $label already exists: $dest"
    return
  fi
  local tmp="${dest}.part"
  rm -f "$tmp"
  echo "[download] $label"
  curl --fail --location --retry 3 --retry-delay 2 \
       --connect-timeout 30 --output "$tmp" "$url"
  [[ -s "$tmp" ]] || { echo "Empty download: $label" >&2; rm -f "$tmp"; exit 1; }
  mv "$tmp" "$dest"
}

# Required face-swap model.
download \
  "https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx" \
  "$MODEL_DIR/inswapper_128.onnx" "InSwapper-128"

# Required InsightFace detection / recognition / landmark pack.
BUFFALO_ZIP="$MODEL_DIR/buffalo_l.zip"
download \
  "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip" \
  "$BUFFALO_ZIP" "InsightFace buffalo_l"

if [[ -s "$BUFFALO_DIR/det_10g.onnx" || -s "$BUFFALO_DIR/w600k_r50.onnx" ]]; then
  echo "[skip] buffalo_l already appears extracted."
else
  echo "[extract] buffalo_l"
  unzip -oq "$BUFFALO_ZIP" -d "$PROJECT_DIR/.insightface/models"
  if [[ ! -d "$BUFFALO_DIR" ]]; then
    echo "Expected folder not found: $BUFFALO_DIR" >&2
    echo "Inspect .insightface/models and move the extracted buffalo_l folder into place." >&2
    exit 1
  fi
fi

# Optional restoration model.
download \
  "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" \
  "$MODEL_DIR/GFPGANv1.4.pth" "GFPGAN 1.4 (optional)"

echo
echo "Downloads complete. Expected locations:"
echo "  $MODEL_DIR/inswapper_128.onnx"
echo "  $BUFFALO_DIR/"
echo "  $MODEL_DIR/GFPGANv1.4.pth (optional)"
echo
echo "Check model licenses before commercial use; model weights have separate terms."
